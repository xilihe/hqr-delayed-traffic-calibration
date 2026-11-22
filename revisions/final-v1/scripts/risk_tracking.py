"""Post-confirmation daily risk diagnostics, two partitions and two fixed regimes.

No training or full score/budget/delay grid is run. Original risk-grid counts
are reused; new-partition selection loads development only and records alpha
before reading future caches. Daily denominators are actual known-flow counts.
"""
from pathlib import Path
import csv,json,os,sys,statistics
os.environ.setdefault('OPENBLAS_NUM_THREADS','2');os.environ.setdefault('OMP_NUM_THREADS','2')
import numpy as np
R=Path(__file__).resolve().parents[3];V=R/'revisions/final-v1';O=V/'results/risk-tracking';O.mkdir(parents=True,exist_ok=False)
sys.path[:0]=[str(R/'src'),str(R/'scripts')]
from analyse_confirmation import calendar,block_weights
from frozen_protocol import verified_protocol
from array_replay import load_variant,replay
P,_=verified_protocol(R);N=json.loads((R/'revisions/major-v1/new-partition-plan.json').read_text());plan=json.loads((V/'analysis-plan.json').read_text())
configs={'raw':[('fixed_global','Fixed global'),('fixed_classwise','Fixed classwise'),('rolling_global','Rolling global')],'shift':[('fixed_global','Static HQR'),('rolling_global','Dynamic HQR')],'reference':[('rolling_global','Multi-anchor')]}
methods=[label for pairs in configs.values() for _,label in pairs];audit_seeds=P['audit_seeds'];dates=calendar(P['confirmation_start'],P['confirmation_end']);where={d:i for i,d in enumerate(dates)};obs=[d.rsplit('/D',1)[1] for d in P['confirmation_tables']];indices=[where[d] for d in obs]
def write(name,rows):
 with (O/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
# Development-only choices for the new partition.
dev=[]
for e in N['models']:
 for variant,pairs in configs.items():
  initial,daily=load_variant(R,e,variant,'negative_energy',include_confirmation=False)
  for method,label in pairs:
   for alpha in plan['new_split_development_alpha_grid']:
    for seed in ([0] if method.startswith('fixed_') else audit_seeds):
     counts,_=replay(initial,daily,N['known_codes'],method,alpha,32,7,seed);kn=counts[:,N['known_codes'],0].sum(1);rej=counts[:,N['known_codes'],1].sum(1)
     dev.append(dict(partition=20261122,model_seed=e['seed'],audit_seed=seed,method=label,alpha=alpha,mean_daily_known_frr=float(np.mean(rej/kn))))
 print(json.dumps({'new_partition_development_model_complete':e['seed']}),flush=True)
selected=[]
for label in methods:
 curve=[dict(alpha=alpha,known_frr=statistics.mean(r['mean_daily_known_frr'] for r in dev if r['method']==label and r['alpha']==alpha)) for alpha in plan['new_split_development_alpha_grid']]
 choice=min(curve,key=lambda x:(abs(x['known_frr']-.05),x['alpha']));selected.append(dict(partition=20261122,method=label,alpha=choice['alpha'],development_frr=choice['known_frr']))
old=json.loads((R/'revisions/major-v1/results/development-risk-selection/development-selected-rules.json').read_text())
for x in old['decisions']:
 if x['target_known_frr']==.05:selected.append(dict(partition=20261121,method=x['method'],alpha=x['selected_nominal_alpha'],development_frr=x['achieved_development_frr']))
(O/'development-selected-rules.json').write_text(json.dumps(dict(status='DEVELOPMENT_ONLY_SELECTION_WRITTEN_BEFORE_FUTURE_READ',confirmation_caches_loaded_for_selection=False,choices=selected),indent=2)+'\n');write('new-partition-development-grid.csv',dev)
# Only now load previously computed future counts / selected future caches.
with np.load(R/'revisions/major-v1/results/new-partition-v2/primary-class-counts.npz',allow_pickle=False) as f:new_nominal={k:f[k].astype(np.int64) for k in f.files}
store={};identities={};regressions=0
with np.load(R/'revisions/major-v1/results/risk-curves-v2/risk-grid-class-counts.npz',allow_pickle=False) as original:
 for partition,protocol in [(20261121,P),(20261122,N)]:
  known=protocol['known_codes']
  for regime in ['nominal_0.05','development_selected']:
   for variant,pairs in configs.items():
    for method,label in pairs:
     alpha=.05 if regime=='nominal_0.05' else next(x['alpha'] for x in selected if x['partition']==partition and x['method']==label)
     cells=[];ids=[]
     for mi,e in enumerate(protocol['models']):
      for seed in ([0] if method.startswith('fixed_') else audit_seeds):
       if partition==20261121:
        key=f"m{e['seed']}-{variant}-{method}-a{round(alpha*10000):04}-s{seed}";future=original[key]
       elif alpha==.05:
        ci=mi*3+(0 if seed==0 else audit_seeds.index(seed));future=new_nominal[label.replace(' ','_')][ci,indices];regressions+=1
       else:
        initial,daily=load_variant(R,e,variant,'negative_energy');full,_=replay(initial,daily,known,method,alpha,32,7,seed);future=full[[d['date']>=P['confirmation_start'] for d in daily]]
       kn=future[:,known,0].sum(1);rej=future[:,known,1].sum(1);assert np.all(kn>0) and len(kn)==238
       data=np.zeros((len(dates),2),np.int64);data[indices,0]=kn;data[indices,1]=rej;cells.append(data);ids.append((e['seed'],seed))
     store[(partition,regime,label)]=np.asarray(cells);identities[(partition,regime,label)]=ids
# Shared calendar weights. Cell summaries precede fixed-cell averaging.
weights=block_weights(len(dates),28,2000,20261122)
def metrics(data,w):
 n=data[:,:,0];j=data[:,:,1];valid=n>0;frr=np.divide(j,n,out=np.zeros_like(j,dtype=float),where=valid);den=w@valid.T;err=frr-.05
 def average(v):return np.mean((w@(v*valid).T)/den,axis=1)
 values={'mean_daily_known_frr':average(frr),'mean_absolute_target_deviation':average(abs(err)),'rmse_target_deviation':np.mean(np.sqrt((w@(err**2*valid).T)/den),axis=1)}
 for threshold,name in [(.055,'above_5p5'),(.06,'above_6p0')]:
  above=(frr>threshold)&valid;values[name+'_fraction']=average(above);values[name+'_days']=np.mean(w@above.T,axis=1)
 quantiles=[]
 for cell in range(len(data)):
  take=np.flatnonzero(valid[cell]);order=take[np.argsort(frr[cell,take],kind='stable')];ww=w[:,order];cum=np.cumsum(ww,axis=1);total=cum[:,-1];position=.9*(total-1);lo=np.floor(position).astype(int);hi=np.ceil(position).astype(int)
  il=(cum>lo[:,None]).argmax(1);ih=(cum>hi[:,None]).argmax(1);xx=frr[cell,order];quantiles.append(xx[il]+(position-lo)*(xx[ih]-xx[il]))
 values['daily_frr_p90']=np.mean(quantiles,axis=0);return values
summary=[];cells_out=[];daily_out=[];boots={};points={};npz={}
for key,data in store.items():
 partition,regime,label=key;p=metrics(data,np.ones((1,len(dates))));b=metrics(data,weights);points[key]=p;boots[key]=b;npz[f'{partition}__{regime}__{label.replace(" ","_")}']=data
 alpha=.05 if regime=='nominal_0.05' else next(x['alpha'] for x in selected if x['partition']==partition and x['method']==label)
 for metric,value in p.items():
  lo,hi=np.quantile(b[metric],[.025,.975]);summary.append(dict(partition=partition,regime=regime,method=label,alpha=alpha,metric=metric,value=float(value[0]),lower=float(lo),upper=float(hi),observed_days=238,fixed_cells=len(data),block_days=28,valid_draws=2000))
 for ci,(model,audit) in enumerate(identities[key]):
  take=data[ci,indices];frr=take[:,1]/take[:,0];result=dict(partition=partition,regime=regime,method=label,alpha=alpha,model_seed=model,audit_seed=audit,observed_days=238,mean_daily_known_frr=float(frr.mean()),mean_absolute_target_deviation=float(np.mean(abs(frr-.05))),rmse_target_deviation=float(np.sqrt(np.mean((frr-.05)**2))),above_5p5_days=int((frr>.055).sum()),above_6p0_days=int((frr>.06).sum()),daily_frr_p90=float(np.quantile(frr,.9)))
  cells_out.append(result)
  for date,(known_n,rejected) in zip(obs,take):daily_out.append(dict(partition=partition,regime=regime,method=label,model_seed=model,audit_seed=audit,date=date,known_n=int(known_n),rejected_known_n=int(rejected),known_frr=float(rejected/known_n)))
contrasts=[]
for partition in [20261121,20261122]:
 for regime in ['nominal_0.05','development_selected']:
  a=(partition,regime,'Dynamic HQR')
  for comparator in methods:
   if comparator=='Dynamic HQR':continue
   b=(partition,regime,comparator)
   for metric in points[a]:
    delta=boots[a][metric]-boots[b][metric];lo,hi=np.quantile(delta,[.025,.975]);contrasts.append(dict(partition=partition,regime=regime,candidate='Dynamic HQR',comparator=comparator,metric=metric,difference=float(points[a][metric][0]-points[b][metric][0]),lower=float(lo),upper=float(hi),block_days=28,valid_draws=2000))
write('summary.csv',summary);write('per-cell-summary.csv',cells_out);write('daily-known-counts.csv',daily_out);write('paired-contrasts.csv',contrasts);np.savez_compressed(O/'daily-known-counts.npz',**npz)
# Reuse every requested Fixed-classwise oracle contrast and block sensitivity.
with (R/'revisions/major-v1/results/oracle-risk-bootstrap-v2/paired-oracle-contrasts.csv').open() as f:fixed=[r for r in csv.DictReader(f) if r['candidate']=='Dynamic HQR' and r['comparator']=='Fixed classwise']
write('fixed-classwise-oracle-contrasts.csv',fixed)
report=dict(status='POST_CONFIRMATION_TWO_PARTITION_TWO_REGIME_RISK_DIAGNOSTICS_COMPLETE',model_training=False,full_score_resource_grid_rerun=False,new_development_only_grid_rows=len(dev),summary_rows=len(summary),cell_rows=len(cells_out),daily_rows=len(daily_out),paired_contrast_rows=len(contrasts),new_nominal_saved_count_reuse_checks=regressions,denominator='Actual daily known-flow count in each fixed model/audit cell; days equally weighted within each cell, then cells averaged. Static3cells, dynamic9cells; all share date weights. Day-count table shows mean counts; per-cell CSV retains integer counts.',bootstrap='2000 common28-calendar-day moving-block resamples, seed20261122; conditional temporal intervals, no family-wise significance claim.',p90='Linear empirical90th percentile per cell; bootstrap uses exact expanded-sample rank interpolation with calendar multiplicities.',regimes_separate=True,selection_loaded_confirmation=False)
(O/'complete.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
