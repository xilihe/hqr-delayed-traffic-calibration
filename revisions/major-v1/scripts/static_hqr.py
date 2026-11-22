"""Post-confirmation Static HQR ablation using original H maps and disjoint C.

No deployment label is requested or used in thresholds. Every saved class
count is independently reconstructed from boolean rejection masks.
"""
from pathlib import Path
from collections import defaultdict
import csv,gzip,hashlib,json,math,os,statistics,sys
os.environ.setdefault('OPENBLAS_NUM_THREADS','2');os.environ.setdefault('OMP_NUM_THREADS','2')
import numpy as np
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'revisions/major-v1/results/static-hqr'
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from replay import quantile
from summarize_periods import aggregate
from frozen_protocol import verified_protocol,sha256
from analyse_confirmation import calendar,block_weights,weighted_metrics

def cells(data,score,cutoff,isolated=False):
 scores=data[score];prediction=data['codes'][data['prediction']];labels=data['labels'];known=data['known'];selected=~data['source_overlap'] if isolated else np.ones(len(labels),bool)
 rejected=scores>cutoff;correct=known & (prediction==labels) & ~rejected
 codes,inverse=np.unique(labels[selected],return_inverse=True)
 n=np.bincount(inverse);j=np.bincount(inverse,weights=rejected[selected]);a=np.bincount(inverse,weights=correct[selected]);result=[]
 for c,nc,jc,ac in zip(codes,n,j,a):
  take=selected & (labels==c)
  # Independent slice arithmetic, separate from the grouped bincount path.
  assert int(jc)==int(np.count_nonzero(scores[take]>cutoff))
  assert int(ac)==int(np.count_nonzero(known[take] & (prediction[take]==labels[take]) & ~(scores[take]>cutoff)))
  isknown=bool(data['known'][np.flatnonzero(take)[0]]);isbg=bool(data['background'][np.flatnonzero(take)[0]])
  result.append(dict(label=int(c),known=isknown,kind='known' if isknown else 'background' if isbg else 'heldout',n=int(nc),rejected=int(jc),correct_accepted=int(ac)))
 kn=[v for v in result if v['known']];den=sum(v['n'] for v in kn)
 def macro(kind):
  v=[v for v in result if v['kind']==kind];return statistics.mean(x['rejected']/x['n'] for x in v) if v else None
 return dict(class_counts=result,known_n=den,known_frr=sum(v['rejected'] for v in kn)/den,known_macro_frr=statistics.mean(v['rejected']/v['n'] for v in kn),known_correct_accepted=sum(v['correct_accepted'] for v in kn)/den,heldout_macro_detection=macro('heldout'),background_macro_detection=macro('background'))

def main():
 OUT.mkdir(exist_ok=False)
 p,seal=verified_protocol(ROOT);initials=[];summaries=[];checks=0
 if quantile(list(range(18)),.05)!=math.inf or quantile(list(range(19)),.05)!=18:raise AssertionError('Finite-rank boundary changed')
 assert not np.any(np.asarray([1.,1.])>quantile([1.]*30,.05))
 (OUT/'engineering-fixture.json').write_text(json.dumps({'status':'PASS_TEST_ONLY_QUANTILE_BOUNDARY_AND_TIE_FIXTURE','result_claim_allowed':False})+'\n')
 by_seed={}
 with gzip.open(OUT/'static-daily-counts.jsonl.gz','wt') as stream:
  for entry in p['models']:
   seed=entry['seed'];source=ROOT/entry['shift_run'];state=json.loads((source/'reference-maps.json').read_text());assert state['fit_rows']==14875 and state['guard_rows']==15125
   scores={s:[] for s in p['scores']};inputs=[]
   for path in sorted((source/'scores').glob('cal-*.npz')):
    with np.load(path,allow_pickle=False) as a:data={k:a[k] for k in a.files}
    assert data['known'].all()
    for s in p['scores']:scores[s].append(data['rqp_'+s])
    inputs.append(dict(path=str(path.relative_to(ROOT)),sha256=sha256(path)))
   cutoffs={}
   for s,parts in scores.items():
    values=np.concatenate(parts).astype(float);assert len(values)==15125
    q=quantile(values.tolist(),.05);rank=int(np.ceil((len(values)+1)*.95));other=float(np.partition(values,rank-1)[rank-1]) if rank<=len(values) else math.inf
    assert q==other;cutoffs[s]=q
   initials.append(dict(model_seed=seed,H_mapping=str((source/'reference-maps.json').relative_to(ROOT)),H_mapping_sha256=sha256(source/'reference-maps.json'),C_rows=15125,C_files=inputs,static_cutoffs=cutoffs,labels_requested=0))
   rows={(s,v):[] for s in p['scores'] for v in ['all','input_disjoint']}
   scored=json.loads((ROOT/entry['confirmation_run']/'score-complete.json').read_text());hashes={v['path']:v['sha256'] for v in scored['files']}
   for path in sorted((ROOT/entry['confirmation_run']/'shift/scores').glob('test-*.npz')):
    assert sha256(path)==hashes[str(path.relative_to(ROOT))]
    with np.load(path,allow_pickle=False) as a:data={k:a[k] for k in a.files}
    date=path.stem.split('-',1)[1]
    for s in p['scores']:
     for view in ['all','input_disjoint']:
      measured=cells(data,'rqp_'+s,cutoffs[s],view!='all');row=dict(date=date,**measured);rows[(s,view)].append(row)
      stream.write(json.dumps(dict(model_seed=seed,score=s,view=view,method='static_hqr',deployment_queries=0,**row),allow_nan=False)+'\n');checks+=len(data['labels'])
   for (s,view),values in rows.items():summaries.append(dict(model_seed=seed,score=s,view=view,method='static_hqr',deployment_queries=0,**aggregate(values)))
   by_seed[seed]=rows
   print(json.dumps(dict(static_model_complete=seed,dates=len(rows[('negative_energy','all')]),deployment_queries=0)),flush=True)
 with (OUT/'summary.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(summaries[0]));w.writeheader();w.writerows(summaries)
 with (ROOT/'outputs/confirmation-analysis-v1/operating-grid.csv').open() as f:original=list(csv.DictReader(f))
 original_methods=[('raw','fixed_global','Fixed global'),('raw','fixed_classwise','Fixed classwise'),('raw','rolling_global','Rolling global'),('shift','rolling_global','Dynamic HQR'),('reference','rolling_global','Multi-anchor')]
 fields=['mean_daily_known_frr','period_background_macro_rejection','period_heldout_macro_rejection','mean_daily_known_correct_accepted'];combined=[]
 for s in p['scores']:
  for budget in p['budgets']:
   for delay in p['delays']:
    for view in ['all','input_disjoint']:
     selected=[v for v in summaries if v['score']==s and v['view']==view]
     combined.append(dict(method='Static HQR',score=s,comparison_budget=budget,comparison_delay=delay,view=view,deployment_daily_budget=0,**{k:statistics.mean(v[k] for v in selected) for k in fields}))
     for variant,method,label in original_methods:
      selected=[v for v in original if v['variant']==variant and v['method']==method and v['score']==s and int(v['budget'])==budget and int(v['delay'])==delay and v['view']==view];assert len(selected)==9
      combined.append(dict(method=label,score=s,comparison_budget=budget,comparison_delay=delay,view=view,deployment_daily_budget=0 if method.startswith('fixed_') else budget,**{k:statistics.mean(float(v[k]) for v in selected) for k in fields}))
 with (OUT/'six-method-comparison.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(combined[0]));w.writeheader();w.writerows(combined)
 # Pair the original dynamic primary counts with the deterministic static
 # counts repeated over the same three audit-stream slots for each model.
 dates=calendar(p['confirmation_start'],p['confirmation_end']);di={d:i for i,d in enumerate(dates)};codes=sorted(int(v) for v in json.loads((ROOT/'outputs/inventory.json').read_text())['apps'].values());ci={c:i for i,c in enumerate(codes)};known=set(p['known_codes']);background={0,51,52};masks={k:np.asarray([c in known if k=='known' else c in background if k=='background' else c not in known|background for c in codes]) for k in ['known','background','heldout']}
 shape=(9,len(dates),len(codes),3);cubes={m:np.zeros(shape) for m in ['static_hqr','shift:rolling_global','raw:rolling_global']};cells_index={(e['seed'],a):i for i,(e,a) in enumerate((e,a) for e in p['models'] for a in p['audit_seeds'])}
 for seed,rows in by_seed.items():
  for a in p['audit_seeds']:
   i=cells_index[(seed,a)]
   for row in rows[('negative_energy','all')]:
    for c in row['class_counts']:cubes['static_hqr'][i,di[row['date']],ci[c['label']]]=[c['n'],c['rejected'],c['correct_accepted']]
 with gzip.open(ROOT/'outputs/confirmation-analysis-v1/primary-daily-counts.jsonl.gz','rt') as f:
  for line in f:
   row=json.loads(line)
   if row['view']!='all' or row['method'] not in cubes:continue
   i=cells_index[(row['model_seed'],row['audit_seed'])]
   for c in row['class_counts']:cubes[row['method']][i,di[row['date']],ci[c['label']]]=[c['n'],c['rejected'],c['correct_accepted']]
 assert all(np.array_equal(v[:,:,:,0],cubes['static_hqr'][:,:,:,0]) for v in cubes.values())
 contrasts=[];point={m:weighted_metrics(v,np.ones((1,len(dates))),masks) for m,v in cubes.items()}
 for block in [7,28,56]:
  weights=block_weights(len(dates),block,2000,20261122);boot={m:weighted_metrics(v,weights,masks) for m,v in cubes.items()}
  for candidate,comparator in [('shift:rolling_global','static_hqr'),('static_hqr','raw:rolling_global')]:
   for metric in point[candidate]:
    delta=boot[candidate][metric]-boot[comparator][metric];bounds=np.quantile(delta,[.025,.975]);contrasts.append(dict(candidate=candidate,comparator=comparator,metric=metric,block_days=block,difference=float(point[candidate][metric][0]-point[comparator][metric][0]),interval_95=bounds.tolist()))
 verified_protocol(ROOT)
 report=dict(status='POST_CONFIRMATION_STATIC_HQR_COMPLETE_WITH_INDEPENDENT_COUNTS',original_protocol_sha256=seal['protocol_sha256'],static_deployment_labels=0,static_fit_inputs=initials,independent_decision_count_checks=checks,summary_cells=len(summaries),comparison_rows=len(combined),primary_contrasts=contrasts,original_artifacts_modified=False)
 (OUT/'complete.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ['static_fit_inputs','primary_contrasts']}),flush=True)
if __name__=='__main__':main()
