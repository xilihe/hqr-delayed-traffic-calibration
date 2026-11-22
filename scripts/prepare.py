"""Fit a reused CNN and cache scores on the development dates only."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import sys
import time
from configio import study_config

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import tables
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cesnet_adapter import canonical_inputs, fit_preprocessor, transform
from reused_cnn import SmallCNN
from model_adapters import build_model, forward_features
from geometry import fit_geometry, distance


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selected_rows(database, paths, known, cap, seed):
    """Sample each source class across all source days, without later labels."""
    candidates = {c: [] for c in known}
    for path in paths:
        labels = database.get_node(path).col("APP")
        for c in known:
            candidates[c].extend((path, int(i)) for i in np.flatnonzero(labels == c))
    chosen = {}
    rng = random.Random(seed)
    for c in known:
        for path, index in rng.sample(candidates[c], min(cap, len(candidates[c]))):
            chosen.setdefault(path, []).append(index)
    return {p: sorted(ids) for p, ids in chosen.items()}


def inputs_at(database, path, indices):
    table = database.get_node(path)
    coordinates = np.asarray(indices, dtype=np.int64)
    packets = table.read_coordinates(coordinates, field="PPI")
    lengths = table.read_coordinates(coordinates, field="PPI_LEN")
    labels = table.read_coordinates(coordinates, field="APP")
    x = canonical_inputs(packets, lengths)
    return x.astype(np.float32), labels.astype(np.int64)


def natural_calibration_rows(database, paths, known, cap, seed):
    candidates = []
    for path in paths:
        labels = database.get_node(path).col("APP")
        candidates.extend((path, int(i)) for i in np.flatnonzero(np.isin(labels, known)))
    chosen = {}
    for path, index in random.Random(seed).sample(candidates, min(cap, len(candidates))):
        chosen.setdefault(path, []).append(index)
    return {p: sorted(indices) for p, indices in chosen.items()}


def signatures(inputs):
    values = np.ascontiguousarray(inputs, dtype="<f4")
    values[values == 0] = 0
    return [hashlib.sha256(row.tobytes()).digest() for row in values]


def infer(model, inputs, device, geometry=None):
    if not len(inputs):
        scores = {"prediction": np.empty(0, dtype=np.int64),
                  "negative_energy": np.empty(0, dtype=np.float32),
                  "one_minus_msp": np.empty(0, dtype=np.float32)}
        if geometry is not None:
            scores["mahalanobis"] = np.empty(0, dtype=np.float64)
        return scores
    result = []
    embeddings = []
    with torch.inference_mode():
        for start in range(0, len(inputs), 1024):
            batch = torch.from_numpy(inputs[start:start + 1024]).to(device)
            if geometry is None:
                result.append(model(batch).cpu())
            else:
                logits, features = forward_features(model, batch)
                result.append(logits.cpu())
                embeddings.append(features.cpu().numpy())
    logits = torch.cat(result)
    if not torch.isfinite(logits).all():
        raise RuntimeError("Non-finite classifier output")
    scores = {"prediction": logits.argmax(1).numpy(),
              "negative_energy": (-torch.logsumexp(logits, dim=1)).numpy(),
              "one_minus_msp": (1 - logits.softmax(1).max(1).values).numpy()}
    if geometry is not None:
        scores["mahalanobis"] = distance(np.concatenate(embeddings), geometry["centers"], geometry["precision"])
        if not np.isfinite(scores["mahalanobis"]).all():
            raise RuntimeError("Non-finite geometry scores")
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", default="configs/pilot-v1.json")
    args = parser.parse_args()
    root = args.root
    config_path, config, output = study_config(root, args.config)
    config_bytes = config_path.read_bytes()
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    runner_sha256 = sha(__file__)
    inventory = json.loads((root / "outputs/inventory.json").read_text())
    dataset = root / "data/CESNET-QUICEXT-25-XS.h5"
    if sha(dataset) != inventory["sha256"]:
        raise ValueError("Dataset hash differs from the acquired version")
    output.mkdir(parents=True, exist_ok=False)
    scores_dir = output / "scores"
    scores_dir.mkdir()
    torch.set_num_threads(4)
    torch.manual_seed(config["training_seed"])
    np.random.seed(config["training_seed"])
    random.seed(config["training_seed"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Training job did not receive the scheduled GPU")
    table_dates = {t["path"]: t["path"].rsplit("/D", 1)[1] for t in inventory["tables"]}
    source = [p for p, d in table_dates.items() if config["source_start"] <= d <= config["source_end"]]
    calibration = [p for p, d in table_dates.items() if config["calibration_start"] <= d <= config["calibration_end"]]
    development = [p for p, d in table_dates.items() if config["development_start"] <= d <= config["development_end"]]
    opened = source + calibration + development
    if any(table_dates[p] >= config["protected_start"] for p in opened):
        raise ValueError("Attempt to open a protected future flow table")
    if not source or not calibration or not development:
        raise ValueError("Temporal roles do not have available tables")
    background = {int(value) for name, value in inventory["apps"].items() if name.casefold().endswith("background")}
    if len(background) != config["expected_background_group_count"]:
        raise ValueError("Official background taxonomy differs from the pre-result protocol; inspect metadata before class selection")
    with tables.open_file(dataset, mode="r") as database:
        counts = Counter()
        for path in source:
            codes, frequencies = np.unique(database.get_node(path).col("APP"), return_counts=True)
            counts.update(dict(zip(map(int, codes), map(int, frequencies))))
        eligible = sorted(c for c, n in counts.items() if c not in background and n >= config["minimum_source_class_rows"])
        (output / "source-support.json").write_text(json.dumps({
            "source_counts": dict(counts), "eligible_codes": eligible,
            "fields_read": ["APP"], "source_tables": source,
            "development_predictions_read": False,
        }, indent=2) + "\n")
        if len(eligible) < config["known_class_count"] + 5:
            raise ValueError(f"Source support leaves only {len(eligible)} eligible classes; review the pre-result design")
        known = sorted(random.Random(config["class_selection_seed"]).sample(eligible, config["known_class_count"]))
        mapping = {code: index for index, code in enumerate(known)}
        source_rows = selected_rows(database, source, known, config["source_cap_per_class"], config["source_sampling_seed"])
        (output / "source-rows.json").write_text(json.dumps(source_rows) + "\n")
        train_parts = [inputs_at(database, path, ids) for path, ids in source_rows.items()]
        train_raw = np.concatenate([p[0] for p in train_parts])
        truth = np.concatenate([p[1] for p in train_parts])
        del train_parts
        source_signatures = set(signatures(train_raw))
        preprocessing = fit_preprocessor(train_raw)
        train = transform(train_raw, preprocessing).astype(np.float32)
        del train_raw
        labels = np.asarray([mapping[int(c)] for c in truth], dtype=np.int64)
        model = build_model(len(known), config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
        tensor_x = torch.from_numpy(train)
        tensor_y = torch.from_numpy(labels)
        count_array = np.bincount(labels, minlength=len(known)).astype(float)
        weights = count_array ** -0.5
        weights /= np.sum(weights * count_array) / len(labels)
        loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
        generator = torch.Generator().manual_seed(config["training_seed"])
        history = []
        started = time.monotonic()
        for epoch in range(config["epochs"]):
            model.train()
            order = torch.randperm(len(labels), generator=generator)
            loss_sum = 0.0
            for start in range(0, len(labels), config["batch_size"]):
                ids = order[start:start + config["batch_size"]]
                xb, yb = tensor_x[ids].to(device), tensor_y[ids].to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(model(xb), yb)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite training loss")
                loss.backward()
                optimizer.step()
                loss_sum += loss.item() * len(ids)
            event = {"epoch": epoch + 1, "loss": loss_sum / len(labels), "elapsed_seconds": time.monotonic() - started}
            history.append(event)
            print(json.dumps(event), flush=True)
        model.eval()
        torch.save({"state_dict": model.state_dict(), "known_codes": known, "preprocessing": preprocessing,
                    "model_type": config.get("model_type", "cnn"), "model_parameters": config.get("model_parameters", {})}, output / "model.pt")
        geometry = None
        if "mahalanobis" in config["rejection_scores"]:
            features = []
            with torch.inference_mode():
                for start in range(0, len(train), 1024):
                    _, embedding = forward_features(model, torch.from_numpy(train[start:start+1024]).to(device))
                    features.append(embedding.cpu().numpy())
            geometry = fit_geometry(np.concatenate(features), labels, len(known), config["covariance_ridge_fraction"])
            np.savez_compressed(output / "geometry-state.npz", **geometry)
            del features
        del train, tensor_x, tensor_y
        calibration_rows = natural_calibration_rows(database, calibration, known, config["calibration_total_cap"], config["calibration_sampling_seed"])
        calibration_signatures = set()
        total_calibration = 0
        calibration_source_overlap = 0
        calibration_repeated_inputs = 0
        for path, indices in calibration_rows.items():
            raw, target = inputs_at(database, path, indices)
            digests = signatures(raw)
            calibration_source_overlap += sum(d in source_signatures for d in digests)
            for digest in digests:
                calibration_repeated_inputs += int(digest in calibration_signatures)
                calibration_signatures.add(digest)
            values = infer(model, transform(raw, preprocessing).astype(np.float32), device, geometry)
            values.update(labels=target, known=np.ones(len(target), dtype=bool),
                          background=np.zeros(len(target), dtype=bool),
                          indices=np.asarray(indices), codes=np.asarray(known))
            np.savez_compressed(scores_dir / f"cal-{table_dates[path]}.npz", **values)
            total_calibration += len(target)
        if total_calibration < 1000:
            raise ValueError("Insufficient chronological calibration support")
        development_counts = []
        for path in development:
            table = database.get_node(path)
            date = table_dates[path]
            seed = int.from_bytes(hashlib.sha256(date.encode()).digest()[:4], "big")
            rng = np.random.default_rng(seed)
            indices = np.sort(rng.choice(table.nrows, min(config["development_cap_per_day"], table.nrows), replace=False))
            raw, target = inputs_at(database, path, indices)
            duplicate = np.asarray([d in source_signatures or d in calibration_signatures for d in signatures(raw)])
            values = infer(model, transform(raw, preprocessing).astype(np.float32), device, geometry)
            values.update(labels=target, known=np.isin(target, known), background=np.isin(target, list(background)),
                          indices=indices, codes=np.asarray(known), source_overlap=duplicate)
            np.savez_compressed(scores_dir / f"dev-{date}.npz", **values)
            development_counts.append({"date": date, "rows": len(target), "source_overlap": int(duplicate.sum())})
            print(json.dumps({"scored_date": date, "rows": len(target)}), flush=True)
    if sha(config_path) != config_sha256 or sha(__file__) != runner_sha256:
        raise RuntimeError("Configuration or running source changed during execution")
    report = {"status": "DEVELOPMENT_SCORES_COMPLETE", "config_sha256": config_sha256,
              "runner_sha256": runner_sha256, "dataset_sha256": inventory["sha256"],
              "model_sha256": sha(output / "model.pt"),
              "geometry_sha256": sha(output / "geometry-state.npz") if geometry is not None else None,
              "geometry_fit_role": "SOURCE_FITTING_ONLY" if geometry is not None else None,
              "model_type": config.get("model_type", "cnn"), "device": str(device), "torch": torch.__version__,
              "python": platform.python_version(), "numpy": np.__version__, "tables": tables.__version__,
              "cuda": torch.version.cuda, "gpu_name": torch.cuda.get_device_name(device),
              "source_rows_sha256": sha(output / "source-rows.json"),
              "known_codes": known, "source_counts": dict(counts), "training_rows": len(labels),
              "calibration_rows": total_calibration, "calibration_source_overlap_rows": calibration_source_overlap,
              "calibration_repeated_input_rows": calibration_repeated_inputs,
              "calibration_policy": "Distinct later flow rows retained, including equal packet prefixes; exact-input overlap reported separately.",
              "training_history": history, "development": development_counts,
              "flow_tables_read": opened, "protected_flow_tables_read": [],
              "result_claim_allowed": "DEVELOPMENT_ONLY_SINGLE_TRAINING_SEED"}
    (output / "prepare-complete.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "known_classes": len(known), "calibration_rows": total_calibration}), flush=True)


if __name__ == "__main__":
    main()
