"""Development-first fixed OFAT grid; no confirmation setting is optimized."""
from pathlib import Path
import csv,json,os,sys,statistics
os.environ.setdefault('OPENBLAS_NUM_THREADS','2');os.environ.setdefault('OMP_NUM_THREADS','2')
import numpy as np
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'revisions/major-v1/results/sensitivity';OUT.mkdir(exist_ok=False)
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from frozen_protocol import verified_protocol
from normalize_scores import assignment
from reference_quantiles import fit_shift_maps,transform
from array_replay import load_variant,replay,summarize
P,_=verified_protocol(ROOT);known=P['known_codes'];bg={0,51,52}
conditions=[dict(name='hqr-original',offset='quantile',probability=.95,prior=20,window=28,method='rolling_global',pool=100)]
for p in [.90,.975]:conditions.append(dict(conditions[0],name='hqr-p'+str(p),probability=p))
for lam in [0,5,100]:conditions.append(dict(conditions[0],name='hqr-prior'+str(lam),prior=lam))
for w in [7,14,56]:conditions.append(dict(conditions[0],name='hqr-window'+str(w),window=w))
for offset in ['mean','median','raw-full','raw-C','multi-anchor']:conditions.append(dict(conditions[0],name=offset,offset=offset))
for pool in [20,100,500]:conditions.append(dict(conditions[0],name='threshold-shrinkage'+str(pool),offset='raw-full',method='pooled_classwise',pool=pool))
(OUT/'declared-conditions.json').write_text(json.dumps(conditions,indent=2)+'\n')
rows=[]
for phase in ['development','confirmation']:
 for entry in P['models']:
  for score in P['scores']:
   initial,daily=load_variant(ROOT,entry,'raw',score);role=[]
   for path in sorted((ROOT/entry['raw_run']/'scores').glob('cal-*.npz')):
    with np.load(path,allow_pickle=False) as f:indices=f['indices']
    role.extend(assignment(path.stem.split('-',1)[1],indices,20261121).tolist())
   role=np.asarray(role);assert role.sum()==14875 and (~role).sum()==15125
   hs=initial['score'][role];hp=initial['predicted'][role];C={k:v[~role] for k,v in initial.items()}
   hstate=json.loads((ROOT/entry['shift_run']/'reference-maps.json').read_text())['maps'][score];rstate=json.loads((ROOT/entry['reference_run']/'reference-maps.json').read_text())['maps'][score]
   for c in conditions:
    working=daily[:85] if phase=='development' else daily
    if c['offset']=='raw-full':cal=initial;data=working
    elif c['offset']=='raw-C':cal=C;data=working
    else:
     if c['offset']=='multi-anchor':maps=rstate
     elif c['offset']=='quantile' and c['probability']==.95 and c['prior']==20:maps=hstate
     elif c['offset']=='quantile' and c['prior']>0:maps=fit_shift_maps(hs,hp,known,c['probability'],c['prior'])
     else:
      fun=np.mean if c['offset']=='mean' else np.median if c['offset']=='median' else lambda x:np.quantile(x,c['probability']);global_center=float(fun(hs));maps={}
      for k in known:
       local=hs[hp==k];n=len(local);weight=n/(n+c['prior']) if n else 0.;center=weight*float(fun(local))+(1-weight)*global_center if n else global_center
       maps[str(k)]=dict(x=[center-1,center+1],z=[-1.,1.])
     cal=dict(C,score=transform(C['score'],C['predicted'],maps));data=[dict(d,score=transform(d['score'],d['predicted'],maps)) for d in working]
    for seed in P['audit_seeds']:
     counts,ledger=replay(cal,data,known,c['method'],.05,32,7,seed,window=c['window'],pseudocount=c['pool'])
     ids=[i for i,d in enumerate(data) if (d['date']<P['confirmation_start'])==(phase=='development')];counts=counts[ids];assert len(counts)==(85 if phase=='development' else 238)
     rows.append(dict(phase=phase,model_seed=entry['seed'],audit_seed=seed,score=score,condition=c['name'],historical_offset=c['offset'],historical_probability=c['probability'],historical_prior=c['prior'],window_days=c['window'],threshold_shrinkage=c['pool'],**summarize(counts,known,bg)))
   print(json.dumps(dict(sensitivity_phase=phase,model_seed=entry['seed'],score=score,conditions=len(conditions))),flush=True)
 phase_rows=[x for x in rows if x['phase']==phase]
 with (OUT/(phase+'-grid.csv')).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(phase_rows[0]));w.writeheader();w.writerows(phase_rows)
 (OUT/(phase+'-complete.json')).write_text(json.dumps(dict(status='POST_CONFIRMATION_'+phase.upper()+'_FIXED_GRID_COMPLETE',rows=len(phase_rows),confirmation_settings_selected=False),indent=2)+'\n')
means=[];metrics=['known_frr','known_correct_accepted','background_macro_detection','background_micro_detection','heldout_macro_detection','heldout_micro_detection','all_unknown_macro_detection','all_unknown_micro_detection']
for phase in ['development','confirmation']:
 for score in P['scores']:
  for c in conditions:
   group=[v for v in rows if v['phase']==phase and v['score']==score and v['condition']==c['name']];assert len(group)==9
   means.append(dict(phase=phase,score=score,condition=c['name'],**{k:statistics.mean(v[k] for v in group) for k in metrics}))
with (OUT/'grid-means.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(means[0]));w.writeheader();w.writerows(means)
with (ROOT/'outputs/confirmation-analysis-v1/operating-grid.csv').open() as f:original=list(csv.DictReader(f))
checks=[]
for row in rows:
 if row['phase']!='confirmation' or row['condition'] not in ['hqr-original','raw-full','multi-anchor','threshold-shrinkage100']:continue
 variant='shift' if row['condition']=='hqr-original' else 'reference' if row['condition']=='multi-anchor' else 'raw';method='pooled_classwise' if row['condition']=='threshold-shrinkage100' else 'rolling_global'
 old=next(v for v in original if v['variant']==variant and v['method']==method and v['score']==row['score'] and int(v['model_seed'])==row['model_seed'] and int(v['sampling_seed'])==row['audit_seed'] and v['budget']=='32' and v['delay']=='7' and v['view']=='all')
 for newkey,oldkey in [('known_frr','mean_daily_known_frr'),('known_correct_accepted','mean_daily_known_correct_accepted'),('background_macro_detection','period_background_macro_rejection'),('heldout_macro_detection','period_heldout_macro_rejection')]:
  diff=abs(row[newkey]-float(old[oldkey]));assert diff<1e-12,(row['condition'],row['score'],newkey,diff);checks.append(diff)
verified_protocol(ROOT)
report={'status':'POST_CONFIRMATION_DEVELOPMENT_FIRST_SENSITIVITY_COMPLETE','conditions':conditions,'rows':len(rows),'means':len(means),'original_baseline_numeric_checks':len(checks),'maximum_baseline_disagreement':max(checks),'confirmation_selected_hyperparameters':False,'statistical_scope':'Entire fixed grid is descriptive OFAT sensitivity. No grid-wide winner, optimized-confirmation rule or multiplicity-adjusted superiority claim.'}
(OUT/'complete.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='conditions'}),flush=True)
