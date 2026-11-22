"""Evaluate the sole predeclared new split with common labels and all outcomes.

Uses already trained new-split models and caches. No model selection, new
partition selection or confirmation-based tuning occurs here.
"""
from pathlib import Path
import csv,gzip,hashlib,json,os,sys,statistics
os.environ.setdefault('OPENBLAS_NUM_THREADS','2');os.environ.setdefault('OMP_NUM_THREADS','2')
import numpy as np
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'revisions/major-v1/results/new-partition-v2';OUT.mkdir(exist_ok=False,parents=True)
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from frozen_protocol import verified_protocol,sha256
from analyse_confirmation import calendar,block_weights,weighted_metrics
from array_replay import load_variant,replay,summarize
from replay import quantile
from independent_array_audit import audit as independent_audit
ORIGINAL,_=verified_protocol(ROOT);P=json.loads((ROOT/'revisions/major-v1/new-partition-plan.json').read_text());known=P['known_codes'];bg={0,51,52};rows=[];primary={};checks=[];independent_decisions=0
methods={'raw':[('fixed_global','Fixed global'),('fixed_classwise','Fixed classwise'),('rolling_global','Rolling global')],'shift':[('fixed_global','Static HQR'),('rolling_global','Dynamic HQR')],'reference':[('rolling_global','Multi-anchor')]}
for e in P['models']:
 source=ROOT/e['raw_run'];prep=json.loads((source/'prepare-complete.json').read_text());scored=json.loads((ROOT/e['confirmation_run']/'score-complete.json').read_text());assert prep['known_codes']==known and scored['known_codes']==known and scored['class_selection_seed']==20261122
 assert scored['plan_sha256']==sha256(ROOT/'revisions/major-v1/new-partition-plan.json');assert sha256(source/'model.pt')==prep['model_sha256']
 for f in scored['files']:assert sha256(ROOT/f['path'])==f['sha256']
 original_entry=ORIGINAL['models'][0]
 for date_record in scored['confirmation_dates']:
  with np.load(ROOT/original_entry['confirmation_run']/'raw/scores'/('test-'+date_record['date']+'.npz'),allow_pickle=False) as f:
   assert hashlib.sha256(f['indices'].astype('<i8').tobytes()).hexdigest()==date_record['indices_sha256']
   assert len(f['indices'])==date_record['rows']
 for variant,configs in methods.items():
  for score in P['scores']:
   initial,daily=load_variant(ROOT,e,variant,score);dates=[x['date'] for x in daily];selected=np.array([d>=ORIGINAL['confirmation_start'] for d in dates]);assert selected.sum()==238
   for method,label in configs:
    # Static rules have one deterministic cell per model; repeat only in
    # matched table rows, never count audit streams as independent samples.
    if method.startswith('fixed_'):
     full,ledger=replay(initial,daily,known,method,.05,0,7,0);future=full[selected];result=summarize(future,known,bg)
     if method=='fixed_global':assert ledger[0]['cutoffs'][0]==quantile(initial['score'].tolist(),.05)
     if score=='negative_energy':
      independent_decisions+=independent_audit(initial,daily,known,method,full,ledger,.05,0,7,0)
      primary[(e['seed'],0,label)]=future
     for budget in P['budgets']:
      for delay in P['delays']:rows.append(dict(model_seed=e['seed'],audit_seed=0,method=label,score=score,comparison_budget=budget,comparison_delay=delay,deployment_daily_budget=0,**result))
    else:
     for budget in P['budgets']:
      for delay in P['delays']:
       for audit_seed in P['audit_seeds']:
        full,ledger=replay(initial,daily,known,method,.05,budget,delay,audit_seed);future=full[selected];result=summarize(future,known,bg)
        rows.append(dict(model_seed=e['seed'],audit_seed=audit_seed,method=label,score=score,comparison_budget=budget,comparison_delay=delay,deployment_daily_budget=budget,**result))
        if score=='negative_energy' and budget==32 and delay==7:
         independent_decisions+=independent_audit(initial,daily,known,method,full,ledger,.05,budget,delay,audit_seed)
         primary[(e['seed'],audit_seed,label)]=future
    print(json.dumps(dict(new_split_evaluated=e['seed'],variant=variant,score=score,method=label)),flush=True)
 # Cross-model cohort identity uses sampled table indices, not only counts.
 indices=[v['indices_sha256'] for v in scored['confirmation_dates']]
 if checks:assert indices==checks[0]['indices_sha256']
 map_state=json.loads((ROOT/e['shift_run']/'reference-maps.json').read_text());assert map_state['fit_rows']+map_state['guard_rows']==30000
 checks.append(dict(seed=e['seed'],known_codes=known,H_rows=map_state['fit_rows'],C_rows=map_state['guard_rows'],source_rows_sha256=prep['source_rows_sha256'],indices_sha256=indices))
with (OUT/'operating-grid.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
metrics=['known_frr','known_correct_accepted','background_macro_detection','background_micro_detection','heldout_macro_detection','heldout_micro_detection','all_unknown_macro_detection','all_unknown_micro_detection'];means=[]
for score in P['scores']:
 for budget in P['budgets']:
  for delay in P['delays']:
   for label in [v[1] for cfgs in methods.values() for v in cfgs]:
    group=[r for r in rows if r['score']==score and r['comparison_budget']==budget and r['comparison_delay']==delay and r['method']==label];assert len(group)==(3 if label.startswith('Fixed') or label=='Static HQR' else 9)
    means.append(dict(score=score,budget=budget,delay=delay,method=label,fixed_cells=len(group),**{k:statistics.mean(r[k] for r in group) for k in metrics}))
with (OUT/'grid-means.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(means[0]));w.writeheader();w.writerows(means)
calendar_dates=calendar(ORIGINAL['confirmation_start'],ORIGINAL['confirmation_end']);where={d:i for i,d in enumerate(calendar_dates)};observed=sorted(ORIGINAL['confirmation_tables']);observed=[d.rsplit('/D',1)[1] for d in observed];cubes={}
for label in [v[1] for cfgs in methods.values() for v in cfgs]:
 cube=np.zeros((9,len(calendar_dates),53,3));cell=0
 for e in P['models']:
  for audit_seed in P['audit_seeds']:
   key=(e['seed'],0 if label.startswith('Fixed') or label=='Static HQR' else audit_seed,label);counts=primary[key]
   for i,date in enumerate(observed):cube[cell,where[date]]=counts[i]
   cell+=1
 cubes[label]=cube
n=cubes['Rolling global'][:,:,:,0];assert all(np.array_equal(v[:,:,:,0],n) for v in cubes.values());assert all(c['source_rows_sha256']==checks[0]['source_rows_sha256'] for c in checks)
masks={kind:np.array([c in known if kind=='known' else c in bg if kind=='background' else c not in set(known)|bg for c in range(53)]) for kind in ['known','background','heldout']};points={m:weighted_metrics(v,np.ones((1,len(calendar_dates))),masks) for m,v in cubes.items()};contrasts=[]
for block in [7,28,56]:
 weights=block_weights(len(calendar_dates),block,2000,20261122);boot={m:weighted_metrics(v,weights,masks) for m,v in cubes.items()}
 for candidate,comparator in [('Dynamic HQR','Rolling global'),('Dynamic HQR','Static HQR'),('Static HQR','Rolling global')]:
  for metric in points[candidate]:
   delta=boot[candidate][metric]-boot[comparator][metric];bounds=np.quantile(delta,[.025,.975]);contrasts.append(dict(candidate=candidate,comparator=comparator,metric=metric,block_days=block,difference=float(points[candidate][metric][0]-points[comparator][metric][0]),interval_95=bounds.tolist()))
np.savez_compressed(OUT/'primary-class-counts.npz',**{m.replace(' ','_'):v for m,v in cubes.items()})
verified_protocol(ROOT)
report={'status':'POST_CONFIRMATION_NEW_PARTITION_EVALUATION_COMPLETE','class_selection_seed':20261122,'known_codes':known,'model_seeds':P['model_seeds'],'audit_seeds':P['audit_seeds'],'rows':len(rows),'mean_rows':len(means),'primary_contrasts':contrasts,'class_and_flow_identity_checks':checks,'independent_network_replication':False,'favorable_partition_screening':False,'original_protocol_unchanged':True,'original_source_models_retrained':False,'new_source_models_trained':3,'independent_primary_decision_checks':independent_decisions}
(OUT/'complete.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ['primary_contrasts','class_and_flow_identity_checks']}),flush=True)
