"""Label-budgeted, delayed ACI comparator with a rolling known-score reference.

Adaptation of Gibbs and Candes (JMLR 2024), equation (4). Each revealed known
label generates one update; errors use the threshold at the original prediction.
No original full-feedback coverage guarantee is claimed for this adaptation.
"""
from collections import deque
from datetime import timedelta
import math
from replay import day, label_requests, metrics, quantile


def extended_quantile(scores, level):
    if not math.isfinite(level):
        raise ValueError('Non-finite adaptive miscoverage level')
    if level <= 0:
        return math.inf
    if level >= 1:
        return -math.inf
    return quantile(scores, level)


def encode_cutoff(value):
    return value if math.isfinite(value) else ('+inf' if value > 0 else '-inf')


def run_delayed_aci(initial, daily, config, budget, delay, seed, gamma, metric_engines=None):
    if not initial or not all(r['known'] for r in initial):
        raise ValueError('Initial calibration must contain known flows')
    if budget < 0 or delay < 1 or gamma <= 0 or not math.isfinite(gamma):
        raise ValueError('Invalid feedback budget, delay or ACI step size')
    dates = sorted(daily)
    if dates and max(day(r['date']) for r in initial) >= day(dates[0]):
        raise ValueError('Calibration must precede replay')
    classes = sorted(config['known_codes'])
    requested, pending, recent_known = [], deque(), deque()
    alpha = config['alpha']
    adaptive_alpha = alpha
    known_revealed = rejected_revealed = 0
    output = []
    for date in dates:
        today = day(date)
        arrived = []
        latest = today-timedelta(days=delay)
        while pending and day(pending[0]['date']) <= latest:
            arrived.append(pending.popleft())
        # The archived known flags are consulted only after the reveal boundary.
        for row in arrived:
            if row['known']:
                known_revealed += 1
                rejected_revealed += row['rejected_at_prediction']
                recent_known.append(row)
        # Algebraically identical to one ACI step per revealed known label;
        # cumulative counts avoid rounding drift at integer quantile ranks.
        adaptive_alpha = alpha + gamma * (alpha * known_revealed - rejected_revealed)
        oldest = today-timedelta(days=config['window_days'])
        while recent_known and day(recent_known[0]['date']) < oldest:
            recent_known.popleft()
        recent = list(recent_known)
        scores = [r['score'] for r in recent] or [r['score'] for r in initial]
        cutoff = extended_quantile(scores, adaptive_alpha)
        cutoffs = dict.fromkeys(classes, cutoff)
        rows = daily[date]
        engine = metric_engines[date] if metric_engines is not None else None
        if engine is not None:
            engine.validate(rows,classes,date)
        elif any(r['date'] != date or r['predicted'] not in classes or not math.isfinite(r['score']) for r in rows):
            raise ValueError('Invalid daily scores or predicted classes')
        result = engine.measure(cutoffs) if engine is not None else metrics(rows, cutoffs)
        result.update(date=date, method='delayed_aci_global', gamma=gamma,
                      budget=budget, delay=delay, sampling_seed=seed,
                      adaptive_alpha=adaptive_alpha, cutoff=encode_cutoff(cutoff),
                      deployment_queries_before_today=len(requested),
                      revealed_known_labels=known_revealed,
                      known_labels_in_window=len(recent),
                      known_frr_exceeds_alpha=result['known_frr'] is not None and result['known_frr'] > alpha,
                      known_frr_excess=max(0, result['known_frr']-alpha) if result['known_frr'] is not None else None,
                      source_disjoint_metrics=engine.measure(cutoffs,isolated=True) if engine is not None else metrics([r for r in rows if r.get('source_disjoint', True)], cutoffs))
        output.append(result)
        for row in label_requests(rows, budget, seed, date):
            stored = dict(row, rejected_at_prediction=int(row['score'] > cutoff))
            requested.append(stored)
            pending.append(stored)
    return output
