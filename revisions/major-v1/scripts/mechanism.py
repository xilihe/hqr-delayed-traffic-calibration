"""Post-confirmation historical-tail and mixture diagnostics; no causal claim."""
from pathlib import Path
from collections import defaultdict
import csv,json,os,sys
os.environ.setdefault('OPENBLAS_NUM_THREADS','2');os.environ.setdefault('OMP_NUM_THREADS','2')
import numpy as np
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'revisions/major-v1/results/mechanism';OUT.mkdir(exist_ok=False)
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from normalize_scores import assignment
from frozen_protocol import verified_protocol
from array_replay import load_variant,replay,order_statistic
P,seal=verified_protocol(ROOT);classes=P['known_codes'];background={0,51,52};probabilities=[.05,.25,.5,.75,.9,.95,.99]
quantiles=[];tails=[];mixture=[];decompositions=[];background_effects=[];tail_summary=[]
def csvout(name,rows):
 with (OUT/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
for entry in P['models']:
 seed=entry['seed'];raw_folder=ROOT/entry['raw_run'];hparts=defaultdict(list);cparts=defaultdict(list)
 for path in sorted((raw_folder/'scores').glob('cal-*.npz')):
  with np.load(path,allow_pickle=False) as a:d={k:a[k] for k in a.files}
  selected=assignment(path.stem.split('-',1)[1],d['indices'],20261121);pred=d['codes'][d['prediction']]
  for key,value in [('prediction',pred),('score',d['negative_energy'])]:hparts[key].append(value[selected]);cparts[key].append(value[~selected])
 hs=np.concatenate(hparts['score']).astype(float);hp=np.concatenate(hparts['prediction']);cs=np.concatenate(cparts['score']).astype(float);cp=np.concatenate(cparts['prediction']);assert len(hs)==14875 and len(cs)==15125
 mapped_initial,mapped_days=load_variant(ROOT,entry,'shift','negative_energy');raw_initial,raw_days=load_variant(ROOT,entry,'raw','negative_energy');u=mapped_initial['score'];assert len(u)==len(cs)
 maps=json.loads((ROOT/entry['shift_run']/'reference-maps.json').read_text())['maps']['negative_energy']
 thresholds={'raw':order_statistic(cs,.05),'hqr':order_statistic(u,.05)};ref={}
 for c in classes:
  values=hs[hp==c];center=float(np.mean(maps[str(c)]['x']))
  for p in probabilities:quantiles.append(dict(model_seed=seed,predicted_code=c,H_n=len(values),probability=p,historical_raw_quantile=float(np.quantile(values,p)) if len(values) else None,historical_offset=center))
  mask=cp==c;n=int(mask.sum())
  for name,score in [('raw',cs),('hqr',u)]:
   rate=float(np.mean(score[mask]>thresholds[name])) if n else None;ref[(c,name)]=rate
   tails.append(dict(model_seed=seed,predicted_code=c,C_n=n,coordinate=name,reference_threshold=thresholds[name],C_tail_rejection=rate))
 for coordinate in ['raw','hqr']:
  rows=[v for v in tails if v['model_seed']==seed and v['coordinate']==coordinate and v['C_n']>0];rates=np.array([v['C_tail_rejection'] for v in rows]);weights=np.array([v['C_n'] for v in rows],float);weights/=weights.sum();mu=float(weights@rates)
  tail_summary.append(dict(model_seed=seed,coordinate=coordinate,represented_predicted_groups=len(rows),C_overall_tail=mu,group_tail_min=float(rates.min()),group_tail_max=float(rates.max()),weighted_tail_std=float(np.sqrt(weights@((rates-mu)**2)))))
 future_ids=[i for i,d in enumerate(raw_days) if d['date']>=P['confirmation_start']];months=sorted({raw_days[i]['date'][:6] for i in future_ids})
 for month in months:
  indices=[i for i in future_ids if raw_days[i]['date'].startswith(month)];labels=np.concatenate([raw_days[i]['labels'] for i in indices]);known=np.concatenate([raw_days[i]['known'] for i in indices]);pred=np.concatenate([raw_days[i]['predicted'] for i in indices]);raw=np.concatenate([raw_days[i]['score'] for i in indices]);mapped=np.concatenate([mapped_days[i]['score'] for i in indices]);assert all(np.array_equal(raw_days[i]['labels'],mapped_days[i]['labels']) for i in indices)
  for c in classes:
   mix=known & (pred==c);mixture.append(dict(model_seed=seed,month=month,predicted_code=c,known_n=int(known.sum()),predicted_known_n=int(mix.sum()),known_share=float(mix.sum()/known.sum()),historical_C_share=float((cp==c).mean())))
  for name,score,reference_score in [('raw',raw,cs),('hqr',mapped,u)]:
   q=thresholds[name];mix_term=0.;conditional_term=0.;unsupported_term=0.;unsupported_mass=0.
   for c in classes:
    selected=known & (pred==c);pi=float(selected.sum()/known.sum());pref=float((cp==c).mean());rate=float(np.mean(score[selected]>q)) if selected.any() else 0.;ref_rate=ref[(c,name)]
    if ref_rate is None:unsupported_term+=pi*rate;unsupported_mass+=pi
    else:mix_term+=(pi-pref)*ref_rate;conditional_term+=pi*(rate-ref_rate)
   observed=float(np.mean(score[known]>q));historical=float(np.mean(reference_score>q));error=observed-historical-mix_term-conditional_term-unsupported_term
   assert abs(error)<1e-12
   decompositions.append(dict(model_seed=seed,month=month,coordinate=name,estimand='Flow-weighted known rejection at fixed historical C threshold',current_known_n=int(known.sum()),historical_rate=historical,current_rate=observed,change=observed-historical,mixture_component=mix_term,conditional_component=conditional_term,unsupported_group_component=unsupported_term,unsupported_group_mass=unsupported_mass,identity_error=error))
 # Actual primary decisions, disaggregated by the predicted-service coordinate.
 for audit_seed in P['audit_seeds']:
  rcounts,rledger=replay(raw_initial,raw_days,classes,'rolling_global',.05,32,7,audit_seed)
  hcounts,hledger=replay(mapped_initial,mapped_days,classes,'rolling_global',.05,32,7,audit_seed)
  totals=defaultdict(lambda:np.zeros(3,dtype=np.int64))
  for i in future_ids:
   rd=raw_days[i];hd=mapped_days[i];reject_r=rd['score']>np.array(rledger[i]['cutoffs'],float)[rd['prediction']];reject_h=hd['score']>np.array(hledger[i]['cutoffs'],float)[hd['prediction']]
   assert int(reject_r[rd['background']].sum())==int(rcounts[i,list(background),1].sum())
   assert int(reject_h[hd['background']].sum())==int(hcounts[i,list(background),1].sum())
   for b in sorted(background):
    for c in classes:
     take=(rd['labels']==b)&(rd['predicted']==c);totals[(b,c)]+=np.array([take.sum(),reject_r[take].sum(),reject_h[take].sum()],dtype=np.int64)
  for (b,c),(n,j,k) in totals.items():
   background_effects.append(dict(model_seed=seed,audit_seed=audit_seed,background_code=b,predicted_code=c,historical_offset=float(np.mean(maps[str(c)]['x'])),n=int(n),rolling_rejected=int(j),hqr_rejected=int(k),rejection_difference=(float((k-j)/n) if n else None)))
 print(json.dumps(dict(mechanism_model_complete=seed)),flush=True)
for name,rows in [('historical-predicted-class-quantiles.csv',quantiles),('historical-C-tail-alignment.csv',tails),('monthly-predicted-mixture.csv',mixture),('mixture-conditional-decomposition.csv',decompositions),('background-effect-by-predicted-class.csv',background_effects),('tail-dispersion-summary.csv',tail_summary)]:csvout(name,rows)
verified_protocol(ROOT)
report={'status':'POST_CONFIRMATION_MECHANISM_DIAGNOSTICS_COMPLETE','historical_comparison':'Raw and HQR tail diagnostics use exactly the same disjoint C flow identities, not unequal historical sample sizes.','mixture_decomposition':'At a fixed historical C threshold: change = supported-group mixture term + supported-group conditional-rate term + unsupported-group contribution. Identity independently checked to1e-12.','primary_metric_distinction':'Mechanism decomposition is flow-weighted known rejection at fixed historical thresholds; it is not the primary mean-daily achieved FRR of the rolling policies.','causal_boundary':'All mechanism evidence is descriptive association and arithmetic decomposition; neither identifies the cause of traffic shift nor proves causal mediation.','background_attribution':'Actual primary decisions are disaggregated by true background group and predicted known service; both favorable and unfavorable changes retained.','row_counts':{k:len(v) for k,v in [('quantiles',quantiles),('tails',tails),('mixture',mixture),('decomposition',decompositions),('background_effects',background_effects)]}}
(OUT/'complete.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
