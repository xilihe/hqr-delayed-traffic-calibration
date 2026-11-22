"""Generate manuscript/supplement tables from audited confirmation sources."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import statistics

LABELS={'raw:fixed_global':'Fixed global','raw:fixed_classwise':'Fixed classwise','raw:rolling_global':'Rolling global','raw:rolling_classwise':'Rolling classwise','raw:pooled_classwise':'Threshold shrinkage','raw:delayed_aci_global':'Delayed ACI','shift:rolling_global':'HQR','reference:rolling_global':'Multi-anchor'}
METRICS={'known_frr':('Known false rejection','mean_daily_known_frr'),'background_macro_detection':('Background detection','period_background_macro_rejection'),'heldout_macro_detection':('Held-out detection','period_heldout_macro_rejection'),'known_correct_accepted':('Correct and accepted','mean_daily_known_correct_accepted')}
ORDER=['shift:rolling_global','raw:rolling_global','raw:delayed_aci_global','raw:rolling_classwise','raw:pooled_classwise','raw:fixed_classwise','raw:fixed_global','reference:rolling_global']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fixed(value,signed=False):
    percent=value*100
    if abs(percent)<.005:percent=0.
    return format(percent,'+.2f' if signed else '.2f')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);args=parser.parse_args();root=args.root
    base=root/'outputs/confirmation-analysis-v1';analysis_path=base/'analysis.json';analysis=json.loads(analysis_path.read_text());protocol_path=root/'research/frozen/confirmation-v1.json';protocol=json.loads(protocol_path.read_text())
    if analysis['protocol_sha256']!=sha(protocol_path):raise ValueError('Analysis does not match frozen protocol')
    if sha(base/'operating-grid.csv')!=analysis['output_hashes']['operating-grid.csv']:raise ValueError('Operating grid changed after analysis')
    with (base/'operating-grid.csv').open() as stream:grid=list(csv.DictReader(stream))
    def selected(method,score,budget,delay,view='all'):
        variant,rule=method.split(':',1)
        rows=[r for r in grid if r['variant']==variant and r['method']==rule and r['score']==score and int(r['budget'])==budget and int(r['delay'])==delay and r['view']==view]
        if len(rows)!=9:raise ValueError('Expected three model and three common audit replications')
        return rows
    checks=[];primary=protocol['primary']
    # Cross-check two arithmetic paths: NumPy weighted class counts in analysis
    # versus the standard-library period aggregator in each evaluation CSV.
    for method in ORDER:
        for view in ['all','input_disjoint']:
            rows=selected(method,primary['score'],primary['budget'],primary['delay'],view)
            for metric,(_,field) in METRICS.items():
                expected=statistics.mean(float(r[field]) for r in rows);actual=analysis['summaries'][method+'|'+view][metric]
                if abs(expected-actual)>1e-12:raise ValueError('Independent point-estimate aggregation differs: '+method+' '+metric)
                checks.append(dict(method=method,view=view,metric=metric,absolute_difference=abs(expected-actual)))
    out=root/'manuscript/tables';out.mkdir(parents=True,exist_ok=True);outputs=[]
    def write(name,lines):
        path=out/name;path.write_text('\n'.join(lines)+'\n');outputs.append(dict(path=str(path.relative_to(root)),sha256=sha(path)))
    lines=[r'\begin{table}[tb]',r'\centering',r'\small',r'\caption{Frozen primary comparison: energy score, a daily budget of 32 labels and a seven-day delay. Rates (\%) average the three source-model seeds and three common audit streams on the same traffic. Known false rejection and correct-and-accepted rates average dates; unknown detection uses period-level true-class macros. Static rules request zero deployment labels.}',r'\label{tab:primary}',r'\setlength{\tabcolsep}{4pt}',r'\begin{tabular}{lrrrr}',r'\toprule',r'Calibration & \shortstack{Known\\FRR} & \shortstack{Background\\detection} & \shortstack{Held-out\\detection} & \shortstack{Correct and\\accepted} \\',r'\midrule']
    for method in ORDER:
        p=analysis['summaries'][method+'|all'];label=r'\textbf{HQR}' if method.startswith('shift:') else LABELS[method]
        lines.append(label+' & '+' & '.join(fixed(p[k]) for k in METRICS)+r' \\')
    lines.extend([r'\bottomrule',r'\end{tabular}',r'\end{table}']);write('primary-results.tex',lines)
    lines=[r'\begin{table}[tb]',r'\centering',r'\small',r'\caption{Paired primary contrasts in percentage points. Differences are HQR minus the comparator; intervals are the 2.5th and 97.5th percentiles of 2,000 paired 28-calendar-day block resamples. Date weights are shared across methods and all fixed model/audit replications.}',r'\label{tab:contrasts}',r'\begin{tabular}{llr}',r'\toprule',r'Comparator & Outcome & Difference [95\% interval] \\',r'\midrule']
    for comparator in ['raw:rolling_global','reference:rolling_global']:
        for i,(metric,(label,_)) in enumerate(METRICS.items()):
            row=next(r for r in analysis['contrasts'] if r['comparator']==comparator and r['metric']==metric and r['view']=='all' and r['block_days']==28)
            interval=', '.join(fixed(v,True) for v in row['interval_95']);name=LABELS[comparator] if i==0 else ''
            lines.append(name+' & '+label+' & $'+fixed(row['difference'],True)+r'\;['+interval+r']$ \\')
        if comparator=='raw:rolling_global':lines.append(r'\addlinespace')
    lines.extend([r'\bottomrule',r'\end{tabular}',r'\end{table}']);write('paired-contrasts.tex',lines)
    # Full-grid tables retain all frozen settings, without post-hoc selection.
    for score in protocol['scores']:
        lines=[r'\begin{table}[p]',r'\centering',r'\footnotesize',r'\caption{Complete frozen grid for '+score.replace('_',' ')+r'. Rates are in \%; each cell is the mean of the same nine fixed model/audit replications. See the machine-readable operating grid for every individual replication and the input-disjoint view.}',r'\label{tab:grid-'+score.replace('_','-')+r'}',r'\setlength{\tabcolsep}{3pt}',r'\begin{tabular}{rrlrrrr}',r'\toprule',r'\shortstack{Daily\\budget} & \shortstack{Delay\\(days)} & Calibration & \shortstack{Known\\FRR} & \shortstack{Background\\detection} & \shortstack{Held-out\\detection} & \shortstack{Correct and\\accepted} \\',r'\midrule']
        for budget in protocol['budgets']:
            for delay in protocol['delays']:
                for i,method in enumerate(ORDER):
                    rows=selected(method,score,budget,delay)
                    values=[statistics.mean(float(r[field]) for r in rows) for _,field in METRICS.values()]
                    lines.append((str(budget) if i==0 else '')+' & '+(str(delay) if i==0 else '')+' & '+LABELS[method]+' & '+' & '.join(fixed(v) for v in values)+r' \\')
                lines.append(r'\addlinespace')
        lines.extend([r'\bottomrule',r'\end{tabular}',r'\end{table}']);write('full-grid-'+score+'.tex',lines)
    cohort=[]
    for entry in protocol['models']:
        path=root/entry['confirmation_run']/'score-complete.json';report=json.loads(path.read_text())
        if report['protocol_sha256']!=analysis['protocol_sha256']:raise ValueError('Cohort evidence uses another protocol')
        row={key:sum(r[key] for r in report['confirmation_tables_read']) for key in ['rows','known_n','heldout_n','background_n','input_overlap']};cohort.append(row)
    if any(c!=cohort[0] for c in cohort):raise ValueError('Confirmation populations differ across source models')
    historical_path=root/'outputs/pilot-transformer-v1/prepare-complete.json';historical=json.loads(historical_path.read_text())
    development_path=root/'outputs/pilot-transformer-v1/period-summary.csv'
    with development_path.open() as stream:development=next(r for r in csv.DictReader(stream) if r['view']=='all' and r['score']=='negative_energy' and r['method']=='rolling_global' and r['budget']=='32' and r['delay']=='7' and r['sampling_seed']=='101')
    source_n=historical['training_rows'];calibration_n=historical['calibration_rows'];development_n=[int(development[k]) for k in ['known_n','heldout_n','background_n']]
    counts=cohort[0];lines=[r'\begin{table}[tb]',r'\centering',r'\small',r'\caption{Chronological data roles. Counts refer to sampled flow rows. Historical mapping and initial risk calibration partition the 30,000-row historical pool; all methods use the same overall prelabelled pool. Confirmation labels are revealed to calibration only through the frozen audit schedule.}',r'\label{tab:cohorts}',r'\setlength{\tabcolsep}{4pt}',r'\begin{tabular}{lrrrr}',r'\toprule',r'Role & Flows & Known & Held-out & Background \\',r'\midrule']
    lines += [f'Source fitting & {source_n:,} & {source_n:,} & 0 & 0 '+r'\\',f'Historical pool & {calibration_n:,} & {calibration_n:,} & 0 & 0 '+r'\\','Development & '+f'{sum(development_n):,} & '+' & '.join(f'{v:,}' for v in development_n)+r' \\']
    lines.append('Confirmation & '+' & '.join(f'{counts[k]:,}' for k in ['rows','known_n','heldout_n','background_n'])+r' \\');lines.extend([r'\bottomrule',r'\end{tabular}',r'\end{table}']);write('cohort-counts.tex',lines)
    report=dict(status='RESULT_TABLES_GENERATED_WITH_INDEPENDENT_POINT_ESTIMATE_AGREEMENT',protocol_sha256=analysis['protocol_sha256'],inputs=[dict(path=str(p.relative_to(root)),sha256=sha(p)) for p in [analysis_path,protocol_path,base/'operating-grid.csv',historical_path,development_path]],point_estimate_cross_checks=checks,outputs=outputs,confirmation_population=counts,qa_status='PENDING_FINAL_LATEX_RENDER')
    (out/'source-mapping.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(status=report['status'],cross_checks=len(checks),tables=len(outputs))),flush=True)


if __name__=='__main__':main()
