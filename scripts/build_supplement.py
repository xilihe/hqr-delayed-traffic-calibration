"""Create supplementary tables directly from audited confirmation estimates."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
a_path = ROOT / 'outputs/confirmation-analysis-v1/analysis.json'
a = json.loads(a_path.read_text())
apps = {int(v): k for k, v in json.loads((ROOT / 'outputs/inventory.json').read_text())['apps'].items()}
out = ROOT / 'manuscript/tables'
metrics = ['known_frr', 'background_macro_detection', 'heldout_macro_detection', 'known_correct_accepted']
labels = {'raw:fixed_global': 'Fixed global', 'raw:fixed_classwise': 'Fixed classwise', 'raw:rolling_global': 'Rolling global', 'raw:rolling_classwise': 'Rolling classwise', 'raw:pooled_classwise': 'Threshold shrinkage', 'raw:delayed_aci_global': 'Delayed ACI', 'shift:rolling_global': 'HQR', 'reference:rolling_global': 'Multi-anchor'}
order = ['shift:rolling_global', 'raw:rolling_global', 'raw:delayed_aci_global', 'raw:rolling_classwise', 'raw:pooled_classwise', 'raw:fixed_classwise', 'raw:fixed_global', 'reference:rolling_global']
outputs = []
def write(name, lines):
    p = out / name
    
    for i in range(len(lines)-1):
        if lines[i].startswith(r'\caption') and lines[i].endswith(r'\\') and lines[i+1].startswith(r'\label'):
            lines[i] = lines[i][:-2] + lines[i+1] + r'\\'
            lines[i+1] = ''
    p.write_text('\n'.join(lines) + '\n')
    outputs.append({'path': str(p.relative_to(ROOT)), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()})
def pct(v):
    return f'{100*v:.2f}'
lines = [r'\begin{table}[p]', r'\centering\footnotesize', r'\caption{Primary operating rates by source-model seed. Each row averages the three common audit streams on the same confirmation traffic. Rates are in \%. Known false rejection and correct-and-accepted rates average dates; unknown detection uses period-level true-class macros.}', r'\label{tab:model-seeds}', r'\begin{tabular}{rlrrrr}', r'\toprule', r'Model seed & Calibration & \shortstack{Known\\FRR} & \shortstack{Background\\detection} & \shortstack{Held-out\\detection} & \shortstack{Correct and\\accepted} \\', r'\midrule']
for seed in [42, 43, 44]:
    for i, method in enumerate(order):
        row = next(r for r in a['model_sensitivity'] if r['model_seed'] == seed and r['method'] == method and r['view'] == 'all')
        lines.append((str(seed) if i == 0 else '') + ' & ' + labels[method] + ' & ' + ' & '.join(pct(row[k]) for k in metrics) + r' \\')
    lines.append(r'\addlinespace')
lines += [r'\bottomrule\end{tabular}', r'\end{table}']
write('model-seed-results.tex', lines)
lines = [r'\footnotesize', r'\begin{longtable}{lllrrr}', r'\caption{Paired primary contrasts under different calendar-block lengths. Differences and percentile 95\% intervals use percentage points and 2,000 resamples. Every contrast is HQR minus the named comparator. All means use the nine fixed model/audit cells. The input-disjoint view keeps the same decisions and requests.}\\', r'\label{tab:block-sensitivity}', r'\toprule', r'View & Comparator & Outcome & Block & Difference & 95\% interval \\', r'\midrule\endfirsthead', r'\multicolumn{6}{l}{\tablename~\thetable\ (continued)}\\', r'\toprule View & Comparator & Outcome & Block & Difference & 95\% interval \\ \midrule\endhead', r'\bottomrule\endfoot']
metric_labels = ['Known FRR', 'Background', 'Held-out', 'Correct/accepted']
for view in ['all', 'input_disjoint']:
    for comparator in ['raw:rolling_global', 'reference:rolling_global']:
        for metric, ml in zip(metrics, metric_labels):
            for b in [7, 28, 56]:
                r = next(r for r in a['contrasts'] if r['view'] == view and r['comparator'] == comparator and r['metric'] == metric and r['block_days'] == b)
                lo, hi = r['interval_95']
                lines.append(('All' if view == 'all' else 'Disjoint') + ' & ' + labels[comparator] + ' & ' + ml + ' & ' + str(b) + ' & ' + f"{100*r['difference']:+.3f}" + ' & [' + f'{100*lo:+.3f}, {100*hi:+.3f}' + r'] \\')
            lines.append(r'\addlinespace')
lines += [r'\end{longtable}', r'\normalsize']
write('block-sensitivity.tex', lines)
lines = [r'\footnotesize', r'\begin{longtable}{rlrrrr}', r'\caption{Class-specific primary rejection rates for all official labels. Each rate (\%) is the mean of nine fixed model/audit cells, calculated from class counts aggregated over confirmation. Known-class rejection is a false rejection; held-out and background rejection is detection. Counts are unique sampled flows, shared across methods. Changes are HQR minus rolling global in percentage points; this is an exploratory breakdown.}\\', r'\label{tab:class-outcomes}', r'\toprule', r'Code / role & Service & Flows & Rolling global & HQR & Change \\', r'\midrule\endfirsthead', r'\multicolumn{6}{l}{\tablename~\thetable\ (continued)}\\', r'\toprule Code / role & Service & Flows & Rolling global & HQR & Change \\ \midrule\endhead', r'\bottomrule\endfoot']
rows=[]
for kind in ['known', 'heldout', 'background']:
    for code in sorted(apps):
        r = next(x for x in a['class_detection'] if x['method'] == 'raw:rolling_global' and x['label'] == code and x['view'] == 'all')
        if r['kind'] != kind:
            continue
        h = next(x for x in a['class_detection'] if x['method'] == 'shift:rolling_global' and x['label'] == code and x['view'] == 'all')
        assert r['unique_flow_n'] == h['unique_flow_n']
        role = {'known': 'K', 'heldout': 'H', 'background': 'B'}[kind]
        lines.append(str(code) + ' / ' + role + r' & \texttt{\detokenize{' + apps[code] + '}} & ' + f"{r['unique_flow_n']:,}" + ' & ' + pct(r['mean_rejection']) + ' & ' + pct(h['mean_rejection']) + ' & ' + f"{100*(h['mean_rejection']-r['mean_rejection']):+.2f}" + r' \\')
        rows.append(dict(code=code, role=kind, service=apps[code], unique_flows=r['unique_flow_n'], rolling_global_rejection=r['mean_rejection'], hqr_rejection=h['mean_rejection']))
    lines.append(r'\addlinespace')
assert len(rows) == 53 and sum(r['unique_flows'] for r in rows) == 1417414
lines += [r'\end{longtable}', r'\normalsize']
write('class-outcomes.tex', lines)
with (out / 'class-outcomes-source-data.csv').open('w') as f:
    w=csv.DictWriter(f, fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(ROOT / 'research/verification/supplement-sources.json').write_text(json.dumps({'status': 'SUPPLEMENT_TABLES_FROM_AUDITED_ESTIMATES', 'analysis_sha256': hashlib.sha256(a_path.read_bytes()).hexdigest(), 'classes':len(rows), 'unique_flows':sum(r['unique_flows'] for r in rows), 'outputs':outputs},indent=2)+'\n')
print('Created supplementary seed, block and all-class tables: 53 classes, 1417414 unique flows.')
