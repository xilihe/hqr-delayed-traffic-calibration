"""History-calibrated risk grids and explicitly descriptive oracle FRR matching."""
from pathlib import Path
from collections import defaultdict
import csv,gzip,json,os,statistics,sys
os.environ.setdefault('OPENBLAS_NUM_THREADS','2');os.environ.setdefault('OMP_NUM_THREADS','2')
import numpy as np
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'revisions/major-v1/results/risk-curves-v2'
sys.path.insert(0,str(ROOT/'src'))
from frozen_protocol import verified_protocol
from array_replay import load_variant,replay,summarize
P,seal=verified_protocol(ROOT);plan=json.loads((ROOT/'revisions/major-v1/analysis-plan.json').read_text())
classes=P['known_codes'];background={0,51,52};alphas=plan['risk_curve']['nominal_alpha_grid'];OUT.mkdir(exist_ok=False)
expected={};date_list=sorted(p.rsplit('/D',1)[1] for p in P['confirmation_tables']);date_index={d:i for i,d in enumerate(date_list)}
with gzip.open(ROOT/'outputs/confirmation-analysis-v1/primary-daily-counts.jsonl.gz','rt') as f:
 for line in f:
  row=json.loads(line)
  if row['view']!='all':continue
  key=(row['model_seed'],row['audit_seed'],row['method']);cube=expected.setdefault(key,np.zeros((238,53,3),dtype=np.int64))
  for c in row['class_counts']:cube[date_index[row['date']],c['label']]=[c['n'],c['rejected'],c['correct_accepted']]
rows=[];stored={};audit_checks=[]
configs={'raw':[('fixed_global','Fixed global','raw:fixed_global'),('fixed_classwise','Fixed classwise','raw:fixed_classwise'),('rolling_global','Rolling global','raw:rolling_global')],'shift':[('fixed_global','Static HQR',None),('rolling_global','Dynamic HQR','shift:rolling_global')],'reference':[('rolling_global','Multi-anchor','reference:rolling_global')]}
for entry in P['models']:
 for variant in ['raw','shift','reference']:
  initial,daily=load_variant(ROOT,entry,variant,'negative_energy');selected=np.asarray([d['date']>=P['confirmation_start'] for d in daily]);assert [d['date'] for d in daily if d['date']>=P['confirmation_start']]==date_list
  for method,label,original_key in configs[variant]:
   seeds=[0] if method.startswith('fixed_') else P['audit_seeds']
   # Execute the original alpha first; it must agree before new risk levels.
   for alpha in sorted(alphas,key=lambda a:(a!=.05,a)):
    for seed in seeds:
     counts,ledger=replay(initial,daily,classes,method,alpha,32,7,seed);future=counts[selected]
     if alpha==.05 and original_key is not None:
      exp=expected[(entry['seed'],101 if seed==0 else seed,original_key)]
      assert np.array_equal(future,exp),(entry['seed'],variant,method,seed,'Original per-class counts differ')
      audit_checks.append(dict(model_seed=entry['seed'],method=label,audit_seed=seed,dates=238,all_class_numerators_match=True))
     name=f"m{entry['seed']}-{variant}-{method}-a{round(alpha*10000):04}-s{seed}"
     stored[name]=future
     rows.append(dict(model_seed=entry['seed'],audit_seed=seed,method=label,nominal_alpha=alpha,deployment_daily_budget=0 if method.startswith('fixed_') else 32,delay_days=0 if method.startswith('fixed_') else 7,**summarize(future,classes,background)))
   print(json.dumps(dict(risk_grid_method_complete=label,model_seed=entry['seed'])),flush=True)
with (OUT/'risk-grid.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
np.savez_compressed(OUT/'risk-grid-class-counts.npz',**stored)
fields=[k for k in rows[0] if k not in ['model_seed','audit_seed','method','nominal_alpha','deployment_daily_budget','delay_days']];means=[]
for label in dict.fromkeys(x['method'] for x in rows):
 for alpha in alphas:
  group=[x for x in rows if x['method']==label and x['nominal_alpha']==alpha];means.append(dict(method=label,nominal_alpha=alpha,fixed_replication_cells=len(group),**{k:statistics.mean(x[k] for x in group) for k in fields}))
with (OUT/'risk-grid-means.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(means[0]));w.writeheader();w.writerows(means)
# Interpolate only adjacent nominal-grid points which bracket the requested
# actual FRR. If several segments bracket it, choose by nominal-risk proximity,
# never by the resulting detection. Do not extrapolate outside observed range.
matched=[]
for label in dict.fromkeys(x['method'] for x in means):
 curve=sorted([x for x in means if x['method']==label],key=lambda x:x['nominal_alpha'])
 for target in [.03,.05,.07]:
  eligible=[]
  for a,b in zip(curve,curve[1:]):
   if min(a['known_frr'],b['known_frr'])<=target<=max(a['known_frr'],b['known_frr']) and a['known_frr']!=b['known_frr']:
    eligible.append((abs((a['nominal_alpha']+b['nominal_alpha'])/2-target),a,b))
  if not eligible:
   matched.append(dict(method=label,target_actual_frr=target,status='OUTSIDE_GRID_OR_NO_IDENTIFIABLE_SEGMENT',analysis_type='DESCRIPTIVE_CONFIRMATION_ORACLE'));continue
  _,a,b=min(eligible,key=lambda x:x[0]);weight=(target-a['known_frr'])/(b['known_frr']-a['known_frr'])
  matched.append(dict(method=label,target_actual_frr=target,status='INTERPOLATED',analysis_type='DESCRIPTIVE_CONFIRMATION_ORACLE',left_nominal_alpha=a['nominal_alpha'],right_nominal_alpha=b['nominal_alpha'],**{k:a[k]+weight*(b[k]-a[k]) for k in fields if k!='known_frr'}))
(OUT/'oracle-frr-matched.json').write_text(json.dumps(matched,indent=2)+'\n')
verified_protocol(ROOT)
report={'status':'POST_CONFIRMATION_RISK_GRID_COMPLETE_ORIGINAL_COUNTS_RECONSTRUCTED','score':'negative_energy','nominal_alpha_grid':alphas,'rows':len(rows),'mean_rows':len(means),'original_primary_per_class_audits':audit_checks,'grid_rules':'Each threshold uses historical or already revealed audit labels; no confirmation label selects a risk level or mapping. Added grid is post-confirmation, not original frozen evidence.','oracle_matching':'Interpolation uses confirmation actual FRR and is descriptive oracle analysis, not deployable method selection. No outcome-maximizing envelope or extrapolation.','static_replications':'Three model cells; no deployment audit stream. Dynamic uses nine fixed model/audit cells on the same traffic.','uncertainty':'Original nominal5% bootstrap retained. New matched-risk values are descriptive point interpolations; no significance or independent-network claims.'}
(OUT/'complete.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='original_primary_per_class_audits'}),flush=True)
