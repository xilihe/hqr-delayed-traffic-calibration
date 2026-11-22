"""Recount saved development decisions using NumPy and verify feedback timing."""
import argparse
from datetime import datetime, timedelta
from functools import lru_cache
import gzip
import hashlib
import json
from pathlib import Path
import random

import numpy as np
from configio import study_config


@lru_cache(maxsize=512)
def date(value):
    return datetime.strptime(value, "%Y%m%d").date()


def recount(cached, score_name, thresholds, isolated=False):
    codes = cached["codes"]
    prediction = codes[cached["prediction"]]
    by_class = np.asarray([np.inf if thresholds[str(int(c))] is None else thresholds[str(int(c))] for c in codes], dtype=np.float64)
    cutoffs = by_class[cached["prediction"]]
    rejected = cached[score_name].astype(np.float64) > cutoffs
    mask = ~cached["source_overlap"] if isolated else np.ones(len(prediction), dtype=bool)
    rows = []
    for label in np.unique(cached["labels"][mask]):
        selected = (cached["labels"] == label) & mask
        first = int(np.flatnonzero(selected)[0])
        known = bool(cached["known"][first])
        kind = "known" if known else "background" if cached["background"][first] else "heldout"
        rows.append(dict(known=known, kind=kind, label=int(label), n=int(selected.sum()),
                         rejected=int(np.count_nonzero(selected & rejected)),
                         correct_accepted=int(np.count_nonzero(selected & ~rejected & (prediction == cached["labels"]) & cached["known"]))))
    return sorted(rows, key=lambda r: (r["known"], r["kind"], r["label"]))


def independent_cutoffs(initial_scores, initial_prediction, recent_scores, recent_prediction, classes, method, alpha, pseudocount):
    """Reconstruct numeric thresholds with NumPy order selection, not replay.py."""
    def order(values):
        values = np.asarray(values, dtype=np.float64)
        rank = int(np.ceil((values.size + 1) * (1-alpha)))
        if rank > values.size:
            return np.inf
        return float(np.partition(values, rank-1)[rank-1])
    initial_global = order(initial_scores)
    if method == 'fixed_global':
        return {str(c): initial_global for c in classes}
    if method == 'fixed_classwise':
        scores, prediction, global_cutoff = initial_scores, initial_prediction, initial_global
    else:
        scores, prediction = recent_scores, recent_prediction
        global_cutoff = order(scores) if len(scores) else initial_global
    if method == 'rolling_global':
        return {str(c): global_cutoff for c in classes}
    if method not in {'fixed_classwise','rolling_classwise','pooled_classwise'}:
        raise ValueError('Unknown method in independent threshold audit')
    result = {}
    for c in classes:
        selected = scores[prediction == c]
        local = order(selected) if len(selected) else global_cutoff
        if not np.isfinite(local):
            local = global_cutoff
        if method == 'pooled_classwise' and np.isfinite(global_cutoff):
            n = len(selected)
            local = (n*local + pseudocount*global_cutoff)/(n+pseudocount)
        result[str(c)] = local
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", default="configs/pilot-v1.json")
    args = parser.parse_args()
    root = args.root
    config_path, config, folder = study_config(root, args.config)
    prepared = json.loads((folder / "prepare-complete.json").read_text())
    if prepared["config_sha256"] != hashlib.sha256(config_path.read_bytes()).hexdigest():
        raise ValueError("Prepared data were produced under a different configuration")
    if prepared["protected_flow_tables_read"]:
        raise ValueError("Protected future flow rows were accessed")
    if any(p.rsplit("/D", 1)[1] >= config["protected_start"] for p in prepared["flow_tables_read"]):
        raise ValueError("Protected future table appears in the read ledger")
    cached = {}
    for path in sorted((folder / "scores").glob("dev-*.npz")):
        current_date = path.stem.split("-", 1)[1]
        if current_date >= config["protected_start"]:
            raise ValueError("Future score cache was exposed during development")
        with np.load(path, allow_pickle=False) as data:
            cached[current_date] = {key: data[key].copy() for key in data.files}
        known = np.isin(cached[current_date]["labels"], prepared["known_codes"])
        if not np.array_equal(known, cached[current_date]["known"]):
            raise ValueError("Known/unknown coding differs from the frozen source classes")
    decisions = 0
    files = []
    for score in config["rejection_scores"]:
        calibration = []
        calibration_prediction = []
        for path in sorted((folder / 'scores').glob('cal-*.npz')):
            with np.load(path, allow_pickle=False) as data:
                calibration.append(data[score].astype(np.float64))
                calibration_prediction.append(data['codes'][data['prediction']])
        initial_scores = np.concatenate(calibration)
        initial_prediction = np.concatenate(calibration_prediction)
        for budget in config["label_budgets_per_day"]:
            for delay in config["label_delays_days"]:
                for seed in config["label_sampling_seeds"]:
                    path = folder / f"{score}-budget{budget}-delay{delay}-seed{seed}.jsonl.gz"
                    with gzip.open(path, "rt") as source:
                        results = [json.loads(line) for line in source]
                    if len(results) != len(cached) * len(config["methods"]):
                        raise ValueError("Replay is missing daily method decisions")
                    requested = []
                    requested_by_date = {}
                    for current_date in sorted(cached):
                        data = cached[current_date]
                        today = date(current_date)
                        latest = today-timedelta(days=delay)
                        oldest = today-timedelta(days=config["window_days"])
                        dates_available = [d for d in sorted(requested_by_date) if oldest <= date(d) <= latest]
                        available = [(d,i) for d in dates_available for i in requested_by_date[d]]
                        identifiers = [f"{d}:{int(cached[d]['indices'][i])}" for d, i in available]
                        expected_history = hashlib.sha256(json.dumps(identifiers, separators=(",", ":")).encode()).hexdigest()
                        records = [r for r in results if r["date"] == current_date]
                        if sorted(r["method"] for r in records) != sorted(config["methods"]):
                            raise ValueError("Methods differ across daily replay cells")
                        known_indices = [(d,i) for d,i in available if cached[d]['known'][i]]
                        recent_scores = np.asarray([cached[d][score][i] for d,i in known_indices], dtype=np.float64)
                        recent_prediction = np.asarray([cached[d]['codes'][cached[d]['prediction'][i]] for d,i in known_indices])
                        for record in records:
                            expected_cutoffs = independent_cutoffs(initial_scores, initial_prediction, recent_scores, recent_prediction,
                                                                  prepared['known_codes'], record['method'], config['alpha'], config['pooling_pseudocount'])
                            observed_cutoffs = {c: np.inf if value is None else value for c,value in record['thresholds'].items()}
                            cutoff_codes=sorted(expected_cutoffs)
                            if expected_cutoffs.keys() != observed_cutoffs.keys() or not np.allclose(
                                    [expected_cutoffs[c] for c in cutoff_codes], [observed_cutoffs[c] for c in cutoff_codes], rtol=1e-12, atol=1e-12):
                                raise ValueError('Thresholds differ from independent order-statistic reconstruction')
                            if (record["history_sha256"] != expected_history or record["labels_requested_before_today"] != len(requested)
                                    or record["labels_available_in_window"] != len(available)):
                                raise ValueError("Audit budget, stream or delayed-arrival ledger differs")
                            known_count = len(known_indices)
                            if sum(record["recent_known_support_by_predicted_class"].values()) != known_count:
                                raise ValueError("Per-class feedback support does not add up to the known audit labels")
                            counts = recount(data, score, record["thresholds"])
                            if counts != record["class_counts"]:
                                raise ValueError("Saved decision counts do not match cached scores and thresholds")
                            disjoint = recount(data, score, record["thresholds"], isolated=True)
                            if disjoint != record["source_disjoint_metrics"]["class_counts"]:
                                raise ValueError("Source-disjoint sensitivity counts differ")
                            decisions += len(data["labels"])
                        token = int.from_bytes(hashlib.sha256(f"{seed}:{current_date}".encode()).digest()[:8], "big")
                        indices = random.Random(token).sample(range(len(data["labels"])), min(budget, len(data["labels"])))
                        requested.extend((current_date, i) for i in sorted(indices))
                        requested_by_date[current_date] = sorted(indices)
                    files.append({"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
                    print(json.dumps({"audited": path.name}), flush=True)
    report = {"status": "PASS_INDEPENDENT_NUMPY_RECOUNT_AND_FEEDBACK_LEDGER",
              "decision_evaluations": decisions, "files": files, "protected_flow_tables_read": [],
              "checks": ["Configuration and source class identity", "Protected-date boundaries",
                         "Shared requests, label cost and calendar-day arrival", "Independent order-statistic threshold reconstruction", "All per-class decision counts",
                         "Source-disjoint decision counts"],
              "scope": "Independent arithmetic path; same researcher and same cached model predictions. Model accuracy and novel-method efficacy require separate scientific interpretation."}
    (folder / "audit-complete.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "files": len(files)}), flush=True)


if __name__ == "__main__":
    main()
