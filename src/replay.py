"""Past-only rejection calibration. Larger scores mean more unfamiliar traffic."""
from collections import Counter, defaultdict, deque
from datetime import datetime, timedelta
from functools import lru_cache
import hashlib
import json
import math
import random


@lru_cache(maxsize=512)
def day(value):
    return datetime.strptime(value, "%Y%m%d").date()


def quantile(scores, alpha):
    """Split-calibration order statistic, with conservative finite-sample rounding."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    values = sorted(scores)
    if not values:
        raise ValueError("Cannot calibrate an empty population")
    if not all(math.isfinite(x) for x in values):
        raise ValueError("Calibration scores must be finite")
    rank = math.ceil((len(values) + 1) * (1 - alpha))
    return values[rank - 1] if rank <= len(values) else math.inf


def thresholds(initial, recent, classes, method, alpha, pseudocount, initial_global=None):
    global_initial = quantile([r["score"] for r in initial], alpha) if initial_global is None else initial_global
    if method == "fixed_global":
        return {c: global_initial for c in classes}
    if method == "fixed_classwise":
        known_recent, global_recent = initial, global_initial
    else:
        known_recent = [r for r in recent if r["known"]]
        global_recent = quantile([r["score"] for r in known_recent], alpha) if known_recent else global_initial
    if method == "rolling_global":
        return {c: global_recent for c in classes}
    if method not in {"fixed_classwise", "rolling_classwise", "pooled_classwise"}:
        raise ValueError(f"Unknown calibration method {method}")
    grouped = defaultdict(list)
    for row in known_recent:
        grouped[row["predicted"]].append(row["score"])
    result = {}
    for category in classes:
        values = grouped[category]
        local = quantile(values, alpha) if values else global_recent
        # A sparse local tail cannot produce a finite order statistic. Use the
        # shared threshold, and retain the support count in the replay record.
        if not math.isfinite(local):
            local = global_recent
        if method == "pooled_classwise" and math.isfinite(global_recent):
            weight = len(values) / (len(values) + pseudocount)
            local = weight * local + (1 - weight) * global_recent
        result[category] = local
    return result


def metrics(rows, cutoffs):
    counts = defaultdict(lambda: {"n": 0, "rejected": 0, "correct_accepted": 0})
    for row in rows:
        rejected = row["score"] > cutoffs[row["predicted"]]
        key = (row["known"], row["kind"], row["label"])
        cell = counts[key]
        cell["n"] += 1
        cell["rejected"] += int(rejected)
        cell["correct_accepted"] += int(not rejected and row["known"] and row["predicted"] == row["label"])
    groups = [{"known": k[0], "kind": k[1], "label": k[2], **v} for k, v in sorted(counts.items())]
    known = [r for r in groups if r["known"]]
    heldout = [r for r in groups if r["kind"] == "heldout"]
    background = [r for r in groups if r["kind"] == "background"]

    def micro(cells, field):
        total = sum(r["n"] for r in cells)
        return sum(r[field] for r in cells) / total if total else None

    def macro(cells, field):
        return sum(r[field] / r["n"] for r in cells) / len(cells) if cells else None

    return {
        "known_n": sum(r["n"] for r in known),
        "known_frr": micro(known, "rejected"),
        "known_macro_frr": macro(known, "rejected"),
        "known_correct_accepted": micro(known, "correct_accepted"),
        "heldout_macro_detection": macro(heldout, "rejected"),
        "background_macro_detection": macro(background, "rejected"),
        "class_counts": groups,
    }


def label_requests(rows, budget, seed, date):
    """A random audit of all flows, independent of scores, labels and decisions."""
    token = f"{seed}:{date}".encode()
    random_seed = int.from_bytes(hashlib.sha256(token).digest()[:8], "big")
    indices = random.Random(random_seed).sample(range(len(rows)), min(budget, len(rows)))
    return [rows[i] for i in sorted(indices)]


def run_replay(initial, daily, config, budget, delay, seed, metric_engines=None):
    if budget < 0 or delay < 1 or config["window_days"] < 1 or config["pooling_pseudocount"] <= 0:
        raise ValueError("Invalid label budget, delay, rolling window or pooling parameter")
    if not initial or not all(r["known"] for r in initial):
        raise ValueError("Initial calibration must contain known flows only")
    dates = sorted(daily)
    classes = sorted(config.get("known_codes", {r["predicted"] for r in initial} | {r["label"] for r in initial}))
    if not classes:
        raise ValueError("No source classes")
    if dates and max(day(r["date"]) for r in initial) >= day(dates[0]):
        raise ValueError("Initial calibration overlaps or follows the replay")
    requests = []
    pending, active = deque(), deque()
    initial_global = quantile([r["score"] for r in initial], config["alpha"])
    fixed = {method:thresholds(initial, [], classes, method, config["alpha"], config["pooling_pseudocount"], initial_global)
             for method in config["methods"] if method.startswith("fixed_")}
    output = []
    for date in dates:
        today = day(date)
        rows = daily[date]
        engine = metric_engines[date] if metric_engines is not None else None
        if engine is not None:
            engine.validate(rows, classes, date)
        elif any(r["date"] != date or r["predicted"] not in classes or not math.isfinite(r["score"]) for r in rows):
            raise ValueError("Invalid daily scores or unknown predicted class")
        arrived_before = today-timedelta(days=delay)
        oldest = today-timedelta(days=config["window_days"])
        while pending and day(pending[0]["date"]) <= arrived_before:
            active.append(pending.popleft())
        while active and day(active[0]["date"]) < oldest:
            active.popleft()
        available = list(active)
        ids = [r["id"] for r in available]
        support = Counter(r["predicted"] for r in available if r["known"])
        history_sha = hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
        for method in config["methods"]:
            cutoffs = fixed[method] if method in fixed else thresholds(initial, available, classes, method, config["alpha"], config["pooling_pseudocount"], initial_global)
            result = engine.measure(cutoffs) if engine is not None else metrics(rows, cutoffs)
            result.update(date=date, method=method, budget=budget, delay=delay, sampling_seed=seed,
                          labels_requested_before_today=len(requests), labels_available_in_window=len(available),
                          known_labels_available=sum(r["known"] for r in available), history_sha256=history_sha,
                          recent_known_support_by_predicted_class={str(c): support[c] for c in classes},
                          deployment_queries_before_today=0 if method.startswith("fixed_") else len(requests),
                          thresholds={str(k): v if math.isfinite(v) else None for k, v in cutoffs.items()})
            result["known_frr_exceeds_alpha"] = result["known_frr"] is not None and result["known_frr"] > config["alpha"]
            result["known_frr_excess"] = max(0, result["known_frr"] - config["alpha"]) if result["known_frr"] is not None else None
            if engine is not None:
                result["source_disjoint_metrics"] = engine.measure(cutoffs, isolated=True)
            else:
                disjoint_rows = [r for r in rows if r.get("source_disjoint", True)]
                result["source_disjoint_metrics"] = metrics(disjoint_rows, cutoffs)
            output.append(result)
        # Evaluation labels above never enter calibration. Only the preselected
        # audit labels enter requests and they become usable after their delay.
        chosen = label_requests(rows, budget, seed, date)
        requests.extend(chosen)
        pending.extend(chosen)
    return output
