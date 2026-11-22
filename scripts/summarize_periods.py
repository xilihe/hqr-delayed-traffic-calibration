"""Aggregate saved decision counts by period without changing any decision.

Daily-macro averages and period-level class macros are different estimands.
Both are retained, including the identical-decision input-overlap sensitivity.
"""
import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
import statistics
from configio import study_config


def aggregate(records, isolated=False):
    groups = defaultdict(lambda: dict(n=0, rejected=0, correct_accepted=0))
    daily = []
    for record in records:
        view = record['source_disjoint_metrics'] if isolated else record
        daily.append(view)
        if sum(c['n'] for c in view['class_counts'] if c['known']) != view['known_n']:
            raise ValueError('Saved known-flow total differs from class counts')
        for cell in view['class_counts']:
            key = (cell['known'], cell['kind'], cell['label'])
            if not (0 <= cell['rejected'] <= cell['n'] and 0 <= cell['correct_accepted'] <= cell['n']):
                raise ValueError('Invalid saved count')
            for field in groups[key]:
                groups[key][field] += cell[field]
    result = dict(evaluated_days=len(daily))
    for kind in ['known', 'heldout', 'background']:
        cells = [c for (_, k, _), c in groups.items() if k == kind and c['n']]
        n = sum(c['n'] for c in cells)
        result[kind + '_n'] = n
        result[kind + '_classes_present'] = len(cells)
        result['period_' + kind + '_micro_rejection'] = sum(c['rejected'] for c in cells) / n if n else None
        result['period_' + kind + '_macro_rejection'] = statistics.mean(c['rejected'] / c['n'] for c in cells) if cells else None
    n = result['known_n']
    result['period_known_correct_accepted'] = sum(c['correct_accepted'] for (known, _, _), c in groups.items() if known) / n if n else None
    for field in ['known_frr', 'known_macro_frr', 'known_correct_accepted', 'heldout_macro_detection', 'background_macro_detection']:
        values = [r[field] for r in daily if r[field] is not None]
        result['mean_daily_' + field] = statistics.mean(values) if values else None
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--config', default='configs/transformer-v1.json')
    args = parser.parse_args()
    config_path, config, folder = study_config(args.root, args.config)
    if not (folder / 'audit-complete.json').exists():
        raise ValueError('Independent replay audit required before aggregation')
    paths = sorted(folder.glob('*.jsonl.gz')) + sorted((folder / 'aci-comparison').glob('*.jsonl.gz'))
    summary, provenance = [], []
    for path in paths:
        with gzip.open(path, 'rt') as stream:
            records = [json.loads(line) for line in stream]
        if not records:
            raise ValueError('Empty saved replay')
        by_method = defaultdict(list)
        for record in records:
            by_method[record['method']].append(record)
        for method, selected in by_method.items():
            if len({r['date'] for r in selected}) != len(selected):
                raise ValueError('Repeated method/date')
            score = path.name.split('-budget', 1)[0]
            first = selected[0]
            identity = dict(score=score, method=method, gamma=first.get('gamma', ''),
                            budget=first['budget'], delay=first['delay'], sampling_seed=first['sampling_seed'])
            if 'gamma' in path.name and not identity['gamma']:
                identity['gamma'] = path.name.split('-gamma', 1)[1].removesuffix('.jsonl.gz')
            for isolated in [False, True]:
                summary.append(dict(**identity, view='input_disjoint' if isolated else 'all', **aggregate(selected, isolated)))
        provenance.append(dict(path=str(path.relative_to(args.root)), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    if not summary:
        raise ValueError('No saved replay traces')
    out = folder / 'period-summary.csv'
    with out.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader(); writer.writerows(summary)
    report = dict(status='SAVED_DECISION_COUNT_AGGREGATION_COMPLETE', summary_cells=len(summary),
                  source_config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(), input_traces=provenance,
                  output_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
                  estimands={'period_macro': 'Sum each true-class count over all dates, calculate each class rejection proportion, then weight present classes equally.',
                             'mean_daily_macro': 'Calculate the macro over true classes present on each date, then weight observed dates equally.'},
                  formal_confirmation_performed=False)
    (folder / 'period-summary-provenance.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(status=report['status'], summary_cells=len(summary))), flush=True)


if __name__ == '__main__':
    main()
