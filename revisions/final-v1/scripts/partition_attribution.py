"""Same-network role and decision-group diagnostics; not a causal partition test."""
from pathlib import Path
import csv,json,os,sys,statistics
os.environ.setdefault('OPENBLAS_NUM_THREADS','2');os.environ.setdefault('OMP_NUM_THREADS','2')
import numpy as np
R=Path(__file__).resolve().parents[3];V=R/'revisions/final-v1';O=V/'results/partition-attribution';O.mkdir(parents=True,exist_ok=False)
sys.path[:0]=[str(R/'src'),str(R/'scripts')]
from frozen_protocol import verified_protocol
from analyse_confirmation import calendar,block_weights
from array_replay import load_variant,replay
P,_=verified_protocol(R);N=json.loads((R/'revisions/major-v1/new-partition-plan.json').read_text());apps={int(v):k for k,v in json.loads((R/'outputs/inventory.json').read_text())['apps'].items()};bg={0,51,52}
def write(name,rows):
 with (O/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
roles=[]
for c in sorted(apps):
 old='B' if c in bg else 'K' if c in P['known_codes'] else 'H';new='B' if c in bg else 'K' if c in N['known_codes'] else 'H';roles.append(dict(code=c,service=apps[c],original_role=old,second_role=new,changed=old!=new,meta_family=apps[c].startswith('facebook-') or apps[c] in ['instagram','whatsapp']))
write('all-service-roles.csv',roles)
with (R/'revisions/major-v1/results/class-attribution/predicted-group-decision-switches.csv').open() as f:oldrows=[dict(r,partition='20261121') for r in csv.DictReader(f)]
# Recompute only second-partition primary raw/HQR decisions to recover groups.
rows=[];checks=0;nominal={}
with np.load(R/'revisions/major-v1/results/new-partition-v2/primary-class-counts.npz',allow_pickle=False) as f:nominal={k:f[k].astype(np.int64) for k in ['Rolling_global','Dynamic_HQR']}
dates=calendar(P['confirmation_start'],P['confirmation_end']);indices={d:i for i,d in enumerate(dates)}
for mi,e in enumerate(N['models']):
 ri,rd=load_variant(R,e,'raw','negative_energy');hi,hd=load_variant(R,e,'shift','negative_energy');known=N['known_codes'];offsets=json.loads((R/e['shift_run']/'reference-maps.json').read_text())['maps']['negative_energy']
 for ai,audit in enumerate(P['audit_seeds']):
  rc,rl=replay(ri,rd,known,'rolling_global',.05,32,7,audit);hc,hl=replay(hi,hd,known,'rolling_global',.05,32,7,audit);tally=np.zeros((53,30,5),np.int64)
  for t,d in enumerate(rd):
   if d['date']<P['confirmation_start']:continue
   ci=mi*3+ai;di=indices[d['date']];assert np.array_equal(rc[t],nominal['Rolling_global'][ci,di]) and np.array_equal(hc[t],nominal['Dynamic_HQR'][ci,di]);assert np.array_equal(d['indices'],hd[t]['indices'])
   raw=d['score']>float(rl[t]['cutoffs'][0]);hqr=hd[t]['score']>float(hl[t]['cutoffs'][0]);groups=d['labels']*30+d['prediction'];values=[np.ones(len(groups)),raw,hqr,(~raw)&hqr,raw&(~hqr)]
   tally+=np.stack([np.bincount(groups,weights=x,minlength=53*30).reshape(53,30) for x in values],axis=2).astype(np.int64);checks+=len(groups)
  for c in sorted(set(range(53))-set(known)):
   total=int(tally[c,:,0].sum())
   for i,pred in enumerate(known):
    n,r,h,newrej,newacc=map(int,tally[c,i]);assert h-r==newrej-newacc
    rows.append(dict(partition=20261122,model_seed=e['seed'],audit_seed=audit,true_code=c,predicted_code=pred,true_class_n=total,predicted_group_n=n,predicted_fraction=n/total if total else None,raw_rejected=r,hqr_rejected=h,additional_rejected=newrej,additional_accepted=newacc,contribution_to_class_difference=(h-r)/total if total else None,historical_offset=float(np.mean(offsets[str(pred)]['x']))))
 print(json.dumps({'second_partition_attribution_model_complete':e['seed']}),flush=True)
# Existing original group data are complete for all backgrounds and conncheck;
# use them directly rather than rerunning the original partition.
combined=[]
for r in oldrows+rows:
 out={k:(int(r[k]) if k in ['partition','model_seed','audit_seed','true_code','predicted_code','true_class_n','predicted_group_n','raw_rejected','hqr_rejected','additional_rejected','additional_accepted'] else float(r[k]) if r[k] not in ['',None] else None) for k in ['partition','model_seed','audit_seed','true_code','predicted_code','true_class_n','predicted_group_n','predicted_fraction','raw_rejected','hqr_rejected','additional_rejected','additional_accepted','contribution_to_class_difference','historical_offset']};combined.append(out)
write('per-cell-predicted-attribution.csv',combined)
summary=[]
for partition in [20261121,20261122]:
 for true in sorted({r['true_code'] for r in combined if r['partition']==partition}):
  selected=[r for r in combined if r['partition']==partition and r['true_code']==true];unique={r['true_class_n'] for r in selected};assert len(unique)==1;unique=unique.pop()
  for pred in sorted({r['predicted_code'] for r in selected}):
   group=[r for r in selected if r['predicted_code']==pred];assert len(group)==9;n=sum(r['predicted_group_n'] for r in group);raw=sum(r['raw_rejected'] for r in group);hqr=sum(r['hqr_rejected'] for r in group)
   summary.append(dict(partition=partition,true_code=true,true_service=apps[true],unique_true_class_n=unique,predicted_code=pred,predicted_service=apps[pred],mean_predicted_n=n/9,prediction_share=n/(9*unique) if unique else None,raw_within_predicted_rejection=raw/n if n else None,hqr_within_predicted_rejection=hqr/n if n else None,contribution_to_true_class_change=(hqr-raw)/(9*unique) if unique else None,mean_historical_offset=statistics.mean(r['historical_offset'] for r in group),fixed_cells=9))
write('predicted-attribution-summary.csv',summary)
# All class effects for the second partition with conditional paired intervals.
known=set(N['known_codes']);weights=block_weights(len(dates),28,2000,20261122);raw=nominal['Rolling_global'];hqr=nominal['Dynamic_HQR'];n=raw[:,:,:,0];j0=raw[:,:,:,1];j1=hqr[:,:,:,1];total=n.sum(1);point0=np.divide(j0.sum(1),total,out=np.full_like(total,np.nan,dtype=float),where=total>0).mean(0);point1=np.divide(j1.sum(1),total,out=np.full_like(total,np.nan,dtype=float),where=total>0).mean(0)
# Same class denominators in every fixed cell; reduce model numerators first.
assert np.array_equal(n,n[0:1].repeat(9,axis=0));wn=weights@n[0];b0=np.divide(weights@j0.mean(0),wn,out=np.full_like(wn,np.nan),where=wn>0);b1=np.divide(weights@j1.mean(0),wn,out=np.full_like(wn,np.nan),where=wn>0)
class_rows=[]
for c in range(53):
 delta=b1[:,c]-b0[:,c];valid=np.isfinite(delta);lo,hi=np.quantile(delta[valid],[.025,.975]);class_rows.append(dict(code=c,service=apps[c],role='B' if c in bg else 'K' if c in known else 'H',unique_n=int(total[0,c]),rolling_rejection=float(point0[c]),hqr_rejection=float(point1[c]),difference=float(point1[c]-point0[c]),lower=float(lo),upper=float(hi),valid_draws=int(valid.sum()),macro_contribution=float((point1[c]-point0[c])/(3 if c in bg else 30 if c in known else 20))))
write('second-partition-all-class-effects.csv',class_rows)
# Compact FB-background table: top3 prediction-mass groups per partition plus
# every Meta known group, with all remaining groups explicitly aggregated.
compact=[]
for partition in [20261121,20261122]:
 group=[r for r in summary if r['partition']==partition and r['true_code']==52];chosen={r['predicted_code'] for r in sorted(group,key=lambda r:-r['prediction_share'])[:3]};chosen|={r['predicted_code'] for r in group if r['predicted_service'].startswith('facebook-') or r['predicted_service'] in ['instagram','whatsapp']}
 for r in sorted([r for r in group if r['predicted_code'] in chosen],key=lambda r:-r['prediction_share']):compact.append(r)
 other=[r for r in group if r['predicted_code'] not in chosen];n=sum(r['mean_predicted_n'] for r in other);raw=sum(r['mean_predicted_n']*r['raw_within_predicted_rejection'] for r in other if r['mean_predicted_n']);hqr=sum(r['mean_predicted_n']*r['hqr_within_predicted_rejection'] for r in other if r['mean_predicted_n']);compact.append(dict(partition=partition,true_code=52,true_service=apps[52],unique_true_class_n=21407,predicted_code=-1,predicted_service='Other known predictions',mean_predicted_n=n,prediction_share=n/21407,raw_within_predicted_rejection=raw/n if n else None,hqr_within_predicted_rejection=hqr/n if n else None,contribution_to_true_class_change=sum(r['contribution_to_true_class_change'] for r in other),mean_historical_offset=None,fixed_cells=9))
write('facebook-compact-attribution.csv',compact)
report=dict(status='POST_CONFIRMATION_SECOND_PARTITION_ROLE_AND_ATTRIBUTION_COMPLETE',old_original_attribution_reused=True,new_model_training=False,second_partition_independent_nominal_class_count_checks=checks,all_second_partition_classes=53,unknown_classes=23,changed_meta_roles=[r for r in roles if r['meta_family'] and r['changed']],causal_partition_claim=False,denominator='Prediction shares average9fixed model/audit cells on identical true FB flows. Within-prediction rates pool fixed-cell counts and are descriptive; ninefold repeated flows are not unique observations. Contributions sum exactly to each true-class effect.',compact_selection='All known Meta groups plus the top3 prediction-mass groups per partition, regardless of effect; other groups retained and aggregated.',full_group_rows=len(combined))
(O/'complete.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
