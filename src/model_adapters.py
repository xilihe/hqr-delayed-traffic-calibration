"""Adapt the previous project's models to the shared four-channel packet input."""
import torch
from torch import nn
from reused_cnn import SmallCNN
from reused_temporal import ModelConfig, TrafficModel


class PacketTransformer(nn.Module):
    def __init__(self, classes, parameters):
        super().__init__()
        self.network = TrafficModel(ModelConfig(num_classes=classes, input_dim=3,
                                               backbone='transformer', **parameters))

    def forward_features(self, inputs):
        # The reused temporal model expects [size, time, direction], [B,L,C].
        packets = inputs[:, [2, 0, 1], :].transpose(1, 2).contiguous()
        output = self.network(packets, inputs[:, 3, :].bool())
        return output['logits'], output['embedding']

    def forward(self, inputs):
        return self.forward_features(inputs)[0]


def build_model(classes, config):
    kind = config.get('model_type', 'cnn')
    if kind == 'cnn':
        return SmallCNN(classes)
    if kind == 'transformer':
        return PacketTransformer(classes, config['model_parameters'])
    raise ValueError('Unknown model type')


def forward_features(model, inputs):
    if isinstance(model, SmallCNN):
        mask = inputs[:, 3:4, :]
        embedding = (model.body(inputs) * mask).sum(2) / mask.sum(2).clamp_min(1)
        return model.head(embedding), embedding
    return model.forward_features(inputs)
