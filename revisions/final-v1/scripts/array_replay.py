"""Array replay for post-confirmation supplements, with explicit past-only labels.

Nominal 5% results must match every original saved per-class count before any
new-alpha curve is accepted. No scientific implementation is changed in place.
"""
from pathlib import Path
from datetime import datetime
import hashlib,math,random
import numpy as np


def order_statistic(values,alpha):
 values=np.asarray(values,dtype=np.float64)
 if not len(values) or not np.isfinite(values).all() or not 0<alpha<1:raise ValueError('Invalid quantile')
 rank=math.ceil((len(values)+1)*(1-alpha))
 return float(np.partition(values,rank-1)[rank-1]) if rank<=len(values) else math.inf


def load_variant(root,entry,variant,base_score,include_confirmation=True):
 source=root/entry[variant+'_run'];field=base_score if variant=='raw' else 'rqp_'+base_score
 def read(path):
  with np.load(path,allow_pickle=False) as f:
   d={k:f[k] for k in ['labels','known','background','indices','prediction','codes',field]}
  d['score']=np.asarray(d.pop(field),dtype=np.float64);d['predicted']=d['codes'][d['prediction']];d['date']=path.stem.split('-',1)[1];d['ordinal']=datetime.strptime(d['date'],'%Y%m%d').toordinal();return d
 cal=[read(p) for p in sorted((source/'scores').glob('cal-*.npz'))]
 initial={k:np.concatenate([d[k] for d in cal]) for k in ['score','predicted','known']}
 assert initial['known'].all()
 paths=sorted((source/'scores').glob('dev-*.npz'))
 if include_confirmation:paths+=sorted((root/entry['confirmation_run']/variant/'scores').glob('test-*.npz'))
 daily=[read(path) for path in paths]
 expected=323 if include_confirmation else 85
 assert len(daily)==expected and len({d['date'] for d in daily})==expected
 assert all(a['ordinal']<b['ordinal'] for a,b in zip(daily,daily[1:]))
 return initial,daily


def measure(day,cutoffs,class_count):
 threshold=np.asarray(cutoffs)[day['prediction']]
 reject=day['score']>threshold;correct=day['known'] & (day['labels']==day['predicted']) & ~reject
 count=np.bincount(day['labels'],minlength=class_count)
 bad=np.bincount(day['labels'],weights=reject,minlength=class_count)
 good=np.bincount(day['labels'],weights=correct,minlength=class_count)
 return np.stack([count,bad,good],axis=1).astype(np.int64)


def replay(initial,daily,classes,method,alpha,budget,delay,audit_seed,window=28,pseudocount=100):
 if method not in ['fixed_global','fixed_classwise','rolling_global','rolling_classwise','pooled_classwise']:raise ValueError(method)
 global_initial=order_statistic(initial['score'],alpha)
 static=method.startswith('fixed_');output=[];cutoff_ledger=[];requests=[];total=0
 for day in daily:
  recent=[(past,ids) for past,ids in requests if day['ordinal']-window<=past['ordinal']<=day['ordinal']-delay]
  values=[past['score'][ids[past['known'][ids]]] for past,ids in recent]
  recent_values=np.concatenate(values) if values else np.empty(0)
  global_recent=order_statistic(recent_values,alpha) if len(recent_values) else global_initial
  if method=='fixed_global':cutoffs=np.full(len(classes),global_initial)
  elif method=='rolling_global':cutoffs=np.full(len(classes),global_recent)
  else:
   if method=='fixed_classwise':scores=initial['score'];prediction=initial['predicted'];shared=global_initial
   else:
    scores=recent_values;prediction=np.concatenate([past['predicted'][ids[past['known'][ids]]] for past,ids in recent]) if recent else np.empty(0,int);shared=global_recent
   cutoffs=[]
   for c in classes:
    local=scores[prediction==c];q=order_statistic(local,alpha) if len(local) else shared
    if not math.isfinite(q):q=shared
    if method=='pooled_classwise' and math.isfinite(shared):weight=len(local)/(len(local)+pseudocount);q=weight*q+(1-weight)*shared
    cutoffs.append(q)
   cutoffs=np.asarray(cutoffs)
  output.append(measure(day,cutoffs,53));cutoff_ledger.append(dict(date=day['date'],known_available=0 if static else len(recent_values),deployment_queries_before_today=0 if static else total,cutoffs=[float(v) if math.isfinite(v) else '+inf' for v in cutoffs]))
  if not static:
   token=int.from_bytes(hashlib.sha256(f"{audit_seed}:{day['date']}".encode()).digest()[:8],'big')
   ids=np.asarray(sorted(random.Random(token).sample(range(len(day['labels'])),min(budget,len(day['labels'])))),dtype=int)
   requests.append((day,ids));total+=len(ids)
 return np.asarray(output),cutoff_ledger


def summarize(counts,known,background):
 known=np.asarray([c in known for c in range(counts.shape[1])]);bg=np.asarray([c in background for c in range(counts.shape[1])]);held=~(known|bg);unknown=~known
 n=counts[:,:,0].sum(0);j=counts[:,:,1].sum(0)
 kn=counts[:,known,0].sum(1);valid=kn>0
 out={'known_frr':float(np.mean(counts[:,known,1].sum(1)[valid]/kn[valid])),'known_correct_accepted':float(np.mean(counts[:,known,2].sum(1)[valid]/kn[valid]))}
 for kind,mask in [('background',bg),('heldout',held),('all_unknown',unknown)]:
  support=mask & (n>0);out[kind+'_macro_detection']=float(np.mean(j[support]/n[support]));out[kind+'_micro_detection']=float(j[support].sum()/n[support].sum())
 return out
