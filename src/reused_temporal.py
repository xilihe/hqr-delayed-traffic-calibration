"""Small same-input traffic models; Mamba means the official mamba_ssm.Mamba.

Inputs are preprocessed packet length, log-IAT and optional direction channels.
Time2Vec-like features encode channel 1; they do NOT calibrate the SSM's physical
time step or replace Mamba's learned delta. No pretrained weights are downloaded.
Masks describe a nonempty valid prefix followed by right padding. This contract
keeps causal states and pooled representations independent of trailing padding.

Official constructor checked 2026-09-08: https://github.com/state-spaces/mamba
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    num_classes: int
    input_dim: int = 3
    d_model: int = 64
    n_layers: int = 2
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    dropout: float = 0.1
    backbone: str = "mamba"
    moe_experts: int = 0
    moe_top_k: int = 1

    def __post_init__(self):
        for name in ("num_classes", "input_dim", "d_model", "n_layers", "d_state", "d_conv", "expand"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.input_dim < 2:
            raise ValueError("input_dim must include length and IAT channels")
        if not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("dropout must be finite and in [0, 1)")
        if self.backbone not in {"mamba", "cnn", "gru", "transformer"}:
            raise ValueError("backbone must be mamba, cnn, gru or transformer")
        if type(self.moe_experts) is not int or self.moe_experts < 0:
            raise ValueError("moe_experts must be a nonnegative integer")
        if type(self.moe_top_k) is not int or self.moe_top_k < 1:
            raise ValueError("moe_top_k must be a positive integer")
        if self.moe_experts and self.moe_top_k > self.moe_experts:
            raise ValueError("moe_top_k cannot exceed moe_experts")


class CausalConv1d(nn.Module):
    """Left padding only; shape [B,L,C] is preserved along the packet axis."""

    def __init__(self, in_channels, out_channels, kernel_size, groups=1):
        super().__init__()
        self.left_padding = kernel_size - 1
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, groups=groups)

    def forward(self, x):
        x = F.pad(x.transpose(1, 2), (self.left_padding, 0))
        return self.conv(x).transpose(1, 2)


class MultiScaleTemporalEmbedding(nn.Module):
    """Shared embedding for every backbone, using causal kernels 1, 3 and 5."""

    def __init__(self, config):
        super().__init__()
        width = config.d_model
        self.linear = nn.Linear(config.input_dim, width)
        self.scales = nn.ModuleList(CausalConv1d(config.input_dim, width, k) for k in (1, 3, 5))
        self.fuse = nn.Linear(3 * width, width)
        self.time_affine = nn.Linear(1, width)
        self.norm = nn.LayerNorm(width)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, mask):
        # masked_fill, rather than multiplication, also removes NaNs in padding.
        clean = x.masked_fill(~mask.unsqueeze(-1), 0.0)
        local = self.fuse(torch.cat([F.gelu(conv(clean)) for conv in self.scales], dim=-1))
        time = self.time_affine(clean[..., 1:2])
        time = torch.cat((time[..., :1], torch.sin(time[..., 1:])), dim=-1)
        hidden = self.norm(self.linear(clean) + local + time)
        # A shared, length-independent position encoding also makes the
        # Transformer baseline explicitly sensitive to packet order.
        length, width = hidden.shape[1:]
        positions = torch.arange(length, device=x.device, dtype=torch.float32).unsqueeze(1)
        rates = torch.exp(torch.arange(0, width, 2, device=x.device, dtype=torch.float32)
                          * (-math.log(10000.0) / width))
        angles = positions * rates
        positional = torch.zeros(length, width, device=x.device, dtype=torch.float32)
        positional[:, 0::2] = torch.sin(angles)
        positional[:, 1::2] = torch.cos(angles[:, :width // 2])
        hidden = self.dropout(hidden + positional.to(hidden.dtype))
        return hidden.masked_fill(~mask.unsqueeze(-1), 0.0)


class _MambaLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        try:
            from mamba_ssm import Mamba
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "backbone='mamba' requires the official mamba-ssm package and its "
                "compatible runtime. No GRU or approximate SSM fallback is used. "
                "Check https://github.com/state-spaces/mamba for this environment."
            ) from exc
        self.norm = nn.LayerNorm(config.d_model)
        # Preserve the official initialization, notably dt_proj.bias, A and D.
        self.mixer = Mamba(d_model=config.d_model, d_state=config.d_state,
                           d_conv=config.d_conv, expand=config.expand)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, mask):
        try:
            mixed = self.mixer(self.norm(x))
        except (RuntimeError, NotImplementedError) as exc:
            raise RuntimeError(
                "Official Mamba forward failed; verify PyTorch, device and the "
                "mamba-ssm/selective-scan runtime. No alternative backbone was substituted."
            ) from exc
        return (x + self.dropout(mixed)).masked_fill(~mask.unsqueeze(-1), 0.0)


class _CNNLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.d_model
        self.norm = nn.LayerNorm(width)
        self.conv = CausalConv1d(width, width, config.d_conv, groups=width)
        self.feedforward = nn.Sequential(nn.Linear(width, config.expand * width), nn.GELU(),
                                         nn.Dropout(config.dropout), nn.Linear(config.expand * width, width))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, mask):
        update = self.feedforward(self.conv(self.norm(x)))
        return (x + self.dropout(update)).masked_fill(~mask.unsqueeze(-1), 0.0)


class _GRULayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm = nn.LayerNorm(config.d_model)
        self.gru = nn.GRU(config.d_model, config.d_model, batch_first=True)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, mask):
        update, _ = self.gru(self.norm(x))
        return (x + self.dropout(update)).masked_fill(~mask.unsqueeze(-1), 0.0)


class _TransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        heads = next(h for h in (4, 2, 1) if config.d_model % h == 0)
        self.layer = nn.TransformerEncoderLayer(
            config.d_model, heads, config.expand * config.d_model, dropout=config.dropout,
            activation="gelu", batch_first=True, norm_first=True)

    def forward(self, x, mask):
        causal = torch.ones(x.shape[1], x.shape[1], device=x.device, dtype=torch.bool).triu(1)
        hidden = self.layer(x, src_mask=causal, src_key_padding_mask=~mask)
        return hidden.masked_fill(~mask.unsqueeze(-1), 0.0)


class SparseFlowMoE(nn.Module):
    """Top-k routing AFTER pooling, not token routing and not a pretrained MoE.

    Only selected expert/sample pairs are evaluated. Full-softmax probabilities
    weight selected experts without top-k renormalization, so top-1 retains a
    task-loss gradient for its gate. aux_loss is E*sum(mean_probability*load),
    with detached hard assignment fractions; the balanced reference value is 1.
    """

    def __init__(self, config):
        super().__init__()
        width = config.d_model
        self.top_k = config.moe_top_k
        self.router = nn.Linear(width, config.moe_experts, bias=False)
        self.experts = nn.ModuleList(
            nn.Sequential(nn.Linear(width, config.expand * width), nn.GELU(),
                          nn.Dropout(config.dropout), nn.Linear(config.expand * width, width))
            for _ in range(config.moe_experts))

    def forward(self, x):
        probabilities = F.softmax(self.router(x).float(), dim=-1)
        gates, indices = probabilities.topk(self.top_k, dim=-1)
        mixed = torch.zeros_like(x)
        for expert_id, expert in enumerate(self.experts):
            rows, slots = torch.where(indices == expert_id)
            if rows.numel():
                values = expert(x.index_select(0, rows))
                weighted = values * gates[rows, slots].unsqueeze(-1).to(values.dtype)
                mixed = mixed.index_add(0, rows, weighted.to(mixed.dtype))
        load = F.one_hot(indices, len(self.experts)).float().mean(dim=(0, 1))
        aux_loss = len(self.experts) * (probabilities.mean(dim=0) * load.detach()).sum()
        return mixed, aux_loss


class TrafficModel(nn.Module):
    """Returns logits, pooled embedding, normalized projection and scalar aux_loss."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        config.__post_init__()
        self.config = config
        self.embedding_layer = MultiScaleTemporalEmbedding(config)
        layer_type = {"mamba": _MambaLayer, "cnn": _CNNLayer,
                      "gru": _GRULayer, "transformer": _TransformerLayer}[config.backbone]
        self.layers = nn.ModuleList(layer_type(config) for _ in range(config.n_layers))
        self.pool_norm = nn.LayerNorm(config.d_model)
        self.moe = SparseFlowMoE(config) if config.moe_experts else None
        self.moe_norm = nn.LayerNorm(config.d_model) if self.moe is not None else nn.Identity()
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.d_model, config.num_classes)
        self.projection_head = nn.Sequential(nn.Linear(config.d_model, config.d_model), nn.GELU(),
                                             nn.Linear(config.d_model, config.d_model))

    def forward(self, x, mask):
        if x.ndim != 3 or x.shape[-1] != self.config.input_dim or not x.is_floating_point():
            raise ValueError("x must be floating point [B,L,input_dim]")
        if mask.dtype != torch.bool or mask.shape != x.shape[:2] or mask.device != x.device:
            raise ValueError("mask must be bool [B,L] on x.device")
        if x.shape[0] == 0 or x.shape[1] == 0 or not bool(mask.any(dim=1).all()):
            raise ValueError("every sample must contain at least one valid packet")
        if bool((mask[:, 1:] & ~mask[:, :-1]).any()):
            raise ValueError("mask must be a contiguous valid prefix followed by right padding")
        hidden = self.embedding_layer(x, mask)
        for layer in self.layers:
            hidden = layer(hidden, mask)
        counts = mask.sum(dim=1, keepdim=True).float()
        pooled = (hidden.float().sum(dim=1) / counts).to(hidden.dtype)
        embedding = self.pool_norm(pooled)
        aux_loss = embedding.sum() * 0.0
        if self.moe is not None:
            update, aux_loss = self.moe(embedding)
            embedding = self.moe_norm(embedding + self.dropout(update))
        logits = self.classifier(embedding)
        projection = F.normalize(self.projection_head(embedding).float(), p=2, dim=-1, eps=1e-12)
        return {"logits": logits, "embedding": embedding,
                "projection": projection, "aux_loss": aux_loss}

    def parameter_counts(self):
        """Unique parameter counts, not FLOPs or a batch-level speed estimate.

        active_per_sample = all shared parameters + k equally sized experts.
        A batch can activate more experts in total, up to the whole expert bank.
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        expert_sizes = ([sum(p.numel() for p in expert.parameters()) for expert in self.moe.experts]
                        if self.moe is not None else [])
        expert_total = sum(expert_sizes)
        active = total - expert_total + (self.config.moe_top_k * expert_sizes[0] if expert_sizes else 0)
        return {"total": total, "trainable": trainable, "shared": total - expert_total,
                "expert_total": expert_total, "expert_per_expert": expert_sizes[0] if expert_sizes else 0,
                "active_per_sample": active}
