"""Post-confirmation reconstruction, class robustness and temporal analyses.

Reads immutable saved class counts. No model, threshold or hyperparameter is
selected from confirmation outcomes. Replications share date weights.
"""
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import csv,gzip,hashlib,json,os,statistics,sys
os.environ.setdefault('OPENBLAS_NUM_THREADS','2');os.environ.setdefault('OMP_NUM_THREADS','2')
import numpy as np
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'revisions/major-v1/results'
sys.path.insert(0,str(ROOT/'scripts'))
from analyse_confirmation import block_weights,calendar
P=json.loads((ROOT/'research/frozen/confirmation-v1.json').read_text())
A=json.loads((ROOT/'outputs/confirmation-analysis-v1/analysis.json').read_text())
INPUT=ROOT/'outputs/confirmation-analysis-v1/primary-daily-counts.jsonl.gz'
assert hashlib.sha256(INPUT.read_bytes()).hexdigest()==A['output_hashes'][INPUT.name]
apps={int(v):k for k,v in json.loads((ROOT/'outputs/inventory.json').read_text())['apps'].items()}
classes=sorted(apps);ci={c:i for i,c in enumerate(classes)};dates=calendar(P['confirmation_start'],P['confirmation_end']);di={d:i for i,d in enumerate(dates)}
cells=[(e['seed'],s) for e in P['models'] for s in P['audit_seeds']];ri={c:i for i,c in enumerate(cells)}
methods=[x.removesuffix('|all') for x in A['summaries'] if x.endswith('|all')]
cubes={m:np.zeros((9,len(dates),len(classes),3)) for m in methods}
ledger=defaultdict(lambda:dict(known_daily=[],correct_daily=[],counts=defaultdict(lambda:[0,0])))
with gzip.open(INPUT,'rt') as stream:
 for line in stream:
  row=json.loads(line)
  if row['view']!='all':continue
  m=row['method'];cell=(row['model_seed'],row['audit_seed']);r=ri[cell];t=di[row['date']];k=ledger[(m,cell)]
  known=[v for v in row['class_counts'] if v['known']];n=sum(v['n'] for v in known)
  frr=sum(v['rejected'] for v in known)/n;correct=sum(v['correct_accepted'] for v in known)/n
  assert abs(frr-row['known_frr'])<1e-12 and abs(correct-row['known_correct_accepted'])<1e-12
  k['known_daily'].append(frr);k['correct_daily'].append(correct)
  for v in row['class_counts']:
   cubes[m][r,t,ci[v['label']]]=[v['n'],v['rejected'],v['correct_accepted']]
   k['counts'][v['label']][0]+=v['n'];k['counts'][v['label']][1]+=v['rejected']
known=set(P['known_codes']);bg={c for c,n in apps.items() if n.endswith('-background')};held=set(classes)-known-bg
masks={'known':np.array([c in known for c in classes]),'background':np.array([c in bg for c in classes]),'heldout':np.array([c in held for c in classes]),'all_unknown':np.array([c not in known for c in classes])}
reconstructed={}
for method in methods:
 points=[]
 for cell in cells:
  v=ledger[(method,cell)];counts=v['counts']
  points.append({'known_frr':statistics.mean(v['known_daily']),'known_correct_accepted':statistics.mean(v['correct_daily']),**{kind+'_macro_detection':statistics.mean(counts[c][1]/counts[c][0] for c in sorted(group) if counts[c][0]) for kind,group in [('background',bg),('heldout',held)]}})
 result={metric:statistics.mean(v[metric] for v in points) for metric in points[0]}
 for metric,value in result.items():assert abs(value-A['summaries'][method+'|all'][metric])<1e-12
 reconstructed[method]=result
report={'status':'PASS_INDEPENDENT_SAVED_COUNT_RECONSTRUCTION','new_analysis_type':'POST_CONFIRMATION_SUPPLEMENTARY','input_sha256':hashlib.sha256(INPUT.read_bytes()).hexdigest(),'all_original_methods':reconstructed,'primary_rates_percent':{m:{k:v*100 for k,v in reconstructed[m].items()} for m in ['raw:rolling_global','shift:rolling_global']},'independent_path':'Standard-library summation of class numerators and denominators; separate CSV and NumPy paths retained.'}
(OUT/'original-result-reconstruction.json').write_text(json.dumps(report,indent=2)+'\n')
N=cubes['raw:rolling_global'][:,:,:,0]
assert all(np.array_equal(v[:,:,:,0],N) for v in cubes.values()) and np.all(N==N[0:1])

def ratio(a,b):return np.divide(a,b,out=np.full_like(a,np.nan,dtype=float),where=b>0)
def class_draws(cube,w):
 n=(w@cube[:,:,:,0].transpose(1,0,2).reshape(cube.shape[1],-1)).reshape(len(w),9,len(classes))
 j=(w@cube[:,:,:,1].transpose(1,0,2).reshape(cube.shape[1],-1)).reshape(len(w),9,len(classes))
 return ratio(j,n).mean(axis=1),n

def measures(cube,w):
 rates,n=class_draws(cube,w);rej=(w@cube[:,:,:,1].transpose(1,0,2).reshape(cube.shape[1],-1)).reshape(len(w),9,len(classes))
 answer={}
 for kind in ['background','heldout','all_unknown']:
  mask=masks[kind];answer[kind+'_macro_detection']=np.nanmean(rates[:,mask],axis=1)
  answer[kind+'_micro_detection']=ratio(rej[:,:,mask].sum(2),n[:,:,mask].sum(2)).mean(1)
 dn=cube[:,:,masks['known'],0].sum(2);valid=dn>0
 for label,channel in [('known_frr',1),('known_correct_accepted',2)]:
  vals=ratio(cube[:,:,masks['known'],channel].sum(2),dn);vals=np.nan_to_num(vals)
  answer[label]=ratio(w@vals.T,w@valid.T.astype(float)).mean(1)
 return answer

R=cubes['raw:rolling_global'];H=cubes['shift:rolling_global'];one=np.ones((1,len(dates)))
pointR,_=class_draws(R,one);pointH,_=class_draws(H,one);pointR=pointR[0];pointH=pointH[0]
boot={};core=[]
for block in [7,28,56]:
 w=block_weights(len(dates),block,2000,20261122);r,_=class_draws(R,w);h,_=class_draws(H,w);boot[block]=(r,h)
 mr=measures(R,w);mh=measures(H,w);pr=measures(R,one);ph=measures(H,one)
 for metric in pr:
  delta=mh[metric]-mr[metric];lo,hi=np.nanquantile(delta,[.025,.975])
  core.append(dict(metric=metric,block_days=block,rolling_global=float(pr[metric][0]),hqr=float(ph[metric][0]),difference=float(ph[metric][0]-pr[metric][0]),lower=float(lo),upper=float(hi),valid_draws=int(np.isfinite(delta).sum())))
rows=[];r,h=boot[28]
for i,c in enumerate(classes):
 delta=h[:,i]-r[:,i];valid=np.isfinite(delta);bounds=np.quantile(delta[valid],[.025,.975]) if valid.any() else [np.nan,np.nan]
 rows.append(dict(code=c,service=apps[c],kind='known' if c in known else 'background' if c in bg else 'heldout',unique_n=int(N[0,:,i].sum()),rolling_global=float(pointR[i]),hqr=float(pointH[i]),difference=float(pointH[i]-pointR[i]),lower=float(bounds[0]),upper=float(bounds[1]),valid_draws=int(valid.sum())))
loo=[]
for kind,group in [('background',bg),('heldout',held)]:
 for c in sorted(group):
  indices=[ci[v] for v in sorted(group-{c})];dr=np.nanmean(r[:,indices],axis=1);dh=np.nanmean(h[:,indices],axis=1);delta=dh-dr;bounds=np.nanquantile(delta,[.025,.975])
  pr=float(np.mean(pointR[indices]));ph=float(np.mean(pointH[indices]));loo.append(dict(population=kind,excluded_code=c,excluded_service=apps[c],excluded_unique_n=int(N[0,:,ci[c]].sum()),remaining_classes=len(indices),rolling_global=pr,hqr=ph,difference=ph-pr,lower=float(bounds[0]),upper=float(bounds[1]),valid_draws=int(np.isfinite(delta).sum())))

def write_csv(name,rows):
 with (OUT/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
write_csv('class-specific-uncertainty.csv',rows);write_csv('leave-one-class-out.csv',loo);write_csv('aggregate-detection-and-bootstrap.csv',core)
unknown=[v for v in rows if v['kind']!='known']
def ranks(x):
 order=sorted(range(len(x)),key=lambda i:x[i]);out=np.empty(len(x));start=0
 while start<len(order):
  end=start+1
  while end<len(order) and x[order[end]]==x[order[start]]:end+=1
  for j in order[start:end]:out[j]=(start+end-1)/2+1
  start=end
 return out
sizes=np.asarray([v['unique_n'] for v in unknown],float);changes=np.asarray([v['difference'] for v in unknown])
assoc={'unknown_classes':len(unknown),'pearson_log10_n_delta':float(np.corrcoef(np.log10(sizes),changes)[0,1]),'spearman_n_delta':float(np.corrcoef(ranks(sizes),ranks(changes))[0,1]),'interpretation':'Descriptive class-level association; neither causation nor a selected-significance claim.'}
(OUT/'class-size-association.json').write_text(json.dumps(assoc,indent=2)+'\n')
# Month-level uncertainty uses seven-calendar-day paired blocks; the two
# prespecified multi-month focus periods also retain 28/56-day sensitivity.
periods=[(f'{y}-{m:02}',f'{y}{m:02}01',f'{y}{m:02}31',[7]) for y,m in [(2024,10),(2024,11),(2024,12),(2025,1),(2025,2),(2025,3),(2025,4),(2025,5)]]
periods += [('2024-Oct-Nov','20241001','20241130',[7,28,56]),('2025-Mar-May','20250301','20250531',[7,28,56])]
time_rows=[]
for label,start,end,blocks in periods:
 ids=[i for i,d in enumerate(dates) if start<=d<=end];rc=R[:,ids];hc=H[:,ids];point=np.ones((1,len(ids)));pr=measures(rc,point);ph=measures(hc,point);sample=N[0,ids].sum(0)
 for block in blocks:
  weights=block_weights(len(ids),block,2000,20261122);br=measures(rc,weights);bh=measures(hc,weights)
  for metric in pr:
   delta=bh[metric]-br[metric];lo,hi=np.nanquantile(delta,[.025,.975]);time_rows.append(dict(period=label,calendar_days=len(ids),observed_dates=int((N[0,ids].sum(1)>0).sum()),known_n=int(sample[masks['known']].sum()),background_n=int(sample[masks['background']].sum()),heldout_n=int(sample[masks['heldout']].sum()),unknown_classes_present=int((sample[masks['all_unknown']]>0).sum()),metric=metric,block_days=block,rolling_global=float(pr[metric][0]),hqr=float(ph[metric][0]),difference=float(ph[metric][0]-pr[metric][0]),lower=float(lo),upper=float(hi),valid_draws=int(np.isfinite(delta).sum())))
write_csv('temporal-periods.csv',time_rows)
summary={'status':'POST_CONFIRMATION_CLASS_AND_TEMPORAL_ANALYSES_COMPLETE','primary_original_reconstruction':report['primary_rates_percent'],'class_rows':len(rows),'unknown_classes':len(unknown),'leave_one_out_rows':len(loo),'aggregate_rows':len(core),'temporal_rows':len(time_rows),'association':assoc,'bootstrap':'Paired calendar blocks; 2000 draws; same date weights for nine fixed cells; no nine-independent-datasets interpretation.','rare_class_uncertainty':'Intervals condition on represented classes; valid draw counts are reported. Low-support class estimates can be unstable.','original_inputs_unchanged':True}
(OUT/'saved-count-analysis-complete.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)
