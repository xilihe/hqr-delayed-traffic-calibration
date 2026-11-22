"""Evaluate cached development scores with identical delayed-label audit streams."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from configio import study_config

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from replay import run_replay, metrics, quantile
from fast_metrics import PackedDay


def records(path, score):
    date = path.stem.split("-", 1)[1]
    with np.load(path, allow_pickle=False) as archive:
        # NpzFile loads and decompresses a member on every __getitem__.
        # Materialize each array once before constructing per-flow records.
        data = {key: archive[key] for key in archive.files}
        codes = data["codes"]
        return [dict(id=f"{date}:{int(data['indices'][i])}", date=date,
                     predicted=int(codes[data["prediction"][i]]), label=int(data["labels"][i]),
                     known=bool(data["known"][i]), score=float(data[score][i]),
                     kind="known" if data["known"][i] else "background" if data["background"][i] else "heldout",
                     source_disjoint=not bool(data["source_overlap"][i]) if "source_overlap" in data else True)
                for i in range(len(data["labels"]))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", default="configs/pilot-v1.json")
    args = parser.parse_args()
    root = args.root
    config_path, config, folder = study_config(root, args.config)
    prepared = json.loads((folder / "prepare-complete.json").read_text())
    if prepared["config_sha256"] != hashlib.sha256(config_path.read_bytes()).hexdigest():
        raise ValueError("Prepared score configuration identity differs")
    config["known_codes"] = prepared["known_codes"]
    if prepared["protected_flow_tables_read"]:
        raise ValueError("Protected future data were accessed during development")
    summary = []
    retrospective = []
    for score in config["rejection_scores"]:
        initial = [r for p in sorted((folder / "scores").glob("cal-*.npz")) for r in records(p, score)]
        daily = {p.stem.split("-", 1)[1]: records(p, score) for p in sorted((folder / "scores").glob("dev-*.npz"))}
        if any(d >= config["protected_start"] for d in daily):
            raise ValueError("Protected future date in replay")
        for date, rows in daily.items():
            known_scores = [r["score"] for r in rows if r["known"]]
            if not known_scores:
                continue
            cutoff = quantile(known_scores, config["alpha"])
            result = metrics(rows, {c: cutoff for c in config["known_codes"]})
            result.update(date=date, score=score, method="retrospective_current_day_global",
                          deployable=False, future_labels_used=True,
                          interpretation="Headroom diagnostic for global thresholds with the fixed score; not a prospective method or an upper bound for all classwise rules.")
            retrospective.append(result)
        packed = {date:PackedDay(rows,config["known_codes"],date) for date,rows in daily.items()}
        for budget in config["label_budgets_per_day"]:
            for delay in config["label_delays_days"]:
                for seed in config["label_sampling_seeds"]:
                    results = run_replay(initial, daily, config, budget, delay, seed, metric_engines=packed)
                    name = f"{score}-budget{budget}-delay{delay}-seed{seed}"
                    with gzip.open(folder / (name + ".jsonl.gz"), "wt") as output:
                        for result in results:
                            output.write(json.dumps(result, allow_nan=False) + "\n")
                    for method in config["methods"]:
                        selected = [r for r in results if r["method"] == method]
                        fields = ["known_frr", "known_frr_excess", "known_macro_frr", "known_correct_accepted", "heldout_macro_detection", "background_macro_detection"]
                        row = dict(score=score, method=method, budget=budget, delay=delay, sampling_seed=seed,
                                   evaluated_days=len(selected), days_exceeding_alpha=sum(r["known_frr_exceeds_alpha"] for r in selected))
                        for field in fields:
                            values = [r[field] for r in selected if r[field] is not None]
                            row["mean_daily_" + field] = sum(values) / len(values) if values else None
                        summary.append(row)
                    print(json.dumps({"completed": name}), flush=True)
    with (folder / "summary.csv").open("w") as output:
        writer = csv.DictWriter(output, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    (folder / "retrospective-headroom.json").write_text(json.dumps(retrospective, indent=2, allow_nan=False) + "\n")
    (folder / "evaluation-complete.json").write_text(json.dumps({
        "status": "DEVELOPMENT_REPLAY_COMPLETE", "training_seeds": [config["training_seed"]],
        "label_sampling_seeds": config["label_sampling_seeds"], "summary_cells": len(summary),
        "formal_confirmation_performed": False,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
