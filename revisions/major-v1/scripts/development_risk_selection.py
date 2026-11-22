"""Choose risk-grid alpha only from development known-FRR, then look up future.

Supplementary post-confirmation design, not original frozen selection.
Confirmation score files are not loaded during development selection.
"""
from pathlib import Path
import csv,json,os,sys,statistics
os.environ.setdefault('OPENBLAS_NUM_THREADS','2');os.environ.setdefault('OMP_NUM_THREADS','2')
import numpy as np
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'revisions/major-v1/results/development-risk-selection';OUT.mkdir(exist_ok=False)
sys.path.insert(0,str(ROOT/'src'))
from frozen_protocol import verified_protocol
from array_replay import load_variant,replay,summarize
P,_=verified_protocol(ROOT);plan=json.loads((ROOT/'revisions/major-v1/analysis-plan.json').read_text());alphas=plan['risk_curve']['nominal_alpha_grid'];rows=[]
configs={'raw':[('fixed_global','Fixed global'),('fixed_classwise','Fixed classwise'),('rolling_global','Rolling global')],'shift':[('fixed_global','Static HQR'),('rolling_global','Dynamic HQR')],'reference':[('rolling_global','Multi-anchor')]}
for entry in P['models']:
 for variant,methods in configs.items():
  initial,daily=load_variant(ROOT,entry,variant,'negative_energy',include_confirmation=False);assert len(daily)==85
  for method,label in methods:
   for alpha in alphas:
    for audit in ([0] if method.startswith('fixed_') else P['audit_seeds']):
     counts,_=replay(initial,daily,P['known_codes'],method,alpha,32,7,audit);rows.append(dict(method=label,model_seed=entry['seed'],audit_seed=audit,nominal_alpha=alpha,**summarize(counts,P['known_codes'],{0,51,52})))
 with (OUT/'development-grid-progress.json').open('w') as f:json.dump({'model_completed':entry['seed'],'rows':len(rows)},f)
means=[];decisions=[]
for label in dict.fromkeys(x['method'] for x in rows):
 curve=[]
 for alpha in alphas:
  group=[r for r in rows if r['method']==label and r['nominal_alpha']==alpha];curve.append(dict(method=label,nominal_alpha=alpha,development_known_frr=statistics.mean(r['known_frr'] for r in group)))
 means+=curve
 for target in [.03,.05,.07]:
  chosen=min(curve,key=lambda v:(abs(v['development_known_frr']-target),v['nominal_alpha']));decisions.append(dict(method=label,target_known_frr=target,selected_nominal_alpha=chosen['nominal_alpha'],achieved_development_frr=chosen['development_known_frr']))
# This decision file is written before reading any confirmation result table.
(OUT/'development-selected-rules.json').write_text(json.dumps({'status':'DEVELOPMENT_ONLY_RISK_RULES_SELECTED','criterion':'Minimize absolute mean development known-FRR deviation; ties choose lower nominal alpha.','confirmation_scores_or_labels_loaded_for_selection':False,'decisions':decisions},indent=2)+'\n')
with (OUT/'development-grid.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
with (ROOT/'revisions/major-v1/results/risk-curves-v2/risk-grid-means.csv').open() as f:confirmation=list(csv.DictReader(f))
result=[]
for decision in decisions:
 row=next(v for v in confirmation if v['method']==decision['method'] and float(v['nominal_alpha'])==decision['selected_nominal_alpha']);result.append(dict(**decision,**{'confirmation_'+k:float(v) for k,v in row.items() if k not in ['method','nominal_alpha','fixed_replication_cells']}))
with (OUT/'development-selected-confirmation.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(result[0]));w.writeheader();w.writerows(result)
(OUT/'complete.json').write_text(json.dumps({'status':'POST_CONFIRMATION_DEVELOPMENT_SELECTED_RULE_COMPARISON_COMPLETE','development_rows':len(rows),'selected_rules':len(decisions),'confirmatory_claim':False,'selection_used_confirmation_labels':False,'primary_original_nominal5_results_retained':True},indent=2)+'\n');print('Development-only risk selection and fixed-rule confirmation lookup complete.',flush=True)
