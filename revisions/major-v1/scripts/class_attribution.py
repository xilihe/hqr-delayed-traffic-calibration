"""User-requested post-hoc attribution for three backgrounds and conncheck.

Predicted-group accounting of actual decisions, not causal mediation.
"""
from pathlib import Path
import csv,json,os,sys
os.environ.setdefault('OPENBLAS_NUM_THREADS','2');os.environ.setdefault('OMP_NUM_THREADS','2')
import numpy as np
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'revisions/major-v1/results/class-attribution';OUT.mkdir(exist_ok=False)
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from frozen_protocol import verified_protocol
from array_replay import load_variant,replay
P,_=verified_protocol(ROOT);known=P['known_codes'];targets=[0,51,52,7];rows=[];months=[];switch_checks=0
for e in P['models']:
 ri,rd=load_variant(ROOT,e,'raw','negative_energy');hi,hd=load_variant(ROOT,e,'shift','negative_energy');offsets=json.loads((ROOT/e['shift_run']/'reference-maps.json').read_text())['maps']['negative_energy']
 for audit_seed in P['audit_seeds']:
  rc,rl=replay(ri,rd,known,'rolling_global',.05,32,7,audit_seed);hc,hl=replay(hi,hd,known,'rolling_global',.05,32,7,audit_seed);tally=np.zeros((53,30,5),dtype=np.int64);monthly={}
  for i,d in enumerate(rd):
   if d['date']<P['confirmation_start']:continue
   assert np.array_equal(d['predicted'],hd[i]['predicted']) and np.array_equal(d['labels'],hd[i]['labels'])
   rr=d['score']>float(rl[i]['cutoffs'][0]);hr=hd[i]['score']>float(hl[i]['cutoffs'][0]);group=d['labels']*30+d['prediction'];values=[np.ones(len(group)),rr,hr,(~rr)&hr,rr&(~hr)]
   day=np.stack([np.bincount(group,weights=v,minlength=53*30).reshape(53,30) for v in values],axis=2).astype(np.int64);tally+=day;monthly.setdefault(d['date'][:6],np.zeros_like(tally));monthly[d['date'][:6]]+=day
   assert np.array_equal(day[:,:,1].sum(1),rc[i,:,1]) and np.array_equal(day[:,:,2].sum(1),hc[i,:,1]);switch_checks+=len(group)
  for c in targets:
   total=int(tally[c,:,0].sum())
   for index,pred in enumerate(known):
    n,r,h,new,removed=map(int,tally[c,index]);assert h-r==new-removed
    rows.append(dict(model_seed=e['seed'],audit_seed=audit_seed,true_code=c,predicted_code=pred,true_class_n=total,predicted_group_n=n,predicted_fraction=n/total if total else None,raw_rejected=r,hqr_rejected=h,additional_rejected=new,additional_accepted=removed,contribution_to_class_difference=(h-r)/total if total else None,historical_offset=float(np.mean(offsets[str(pred)]['x']))))
   for month,array in monthly.items():
    n,r,h=map(int,array[c,:,:3].sum(0));months.append(dict(model_seed=e['seed'],audit_seed=audit_seed,true_code=c,month=month,n=n,raw_rejection=r/n if n else None,hqr_rejection=h/n if n else None,difference=(h-r)/n if n else None))
 print(json.dumps(dict(class_attribution_model_complete=e['seed'])),flush=True)
for name,data in [('predicted-group-decision-switches.csv',rows),('target-class-monthly.csv',months)]:
 with (OUT/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
(OUT/'complete.json').write_text(json.dumps({'status':'POST_CONFIRMATION_TARGET_CLASS_ATTRIBUTION_COMPLETE','targets':targets,'model_audit_cells':9,'independent_switch_count_checks':switch_checks,'causal_claim':False,'selection_rationale':'All three background groups and google-conncheck were specifically required by the user. All other unknown classes remain in full class/LOO tables.','denominator_note':'Each cell has the same true-class unique-flow count; predicted-group membership can differ by source model. Do not sum nine cells as unique observations.'},indent=2)+'\n')
