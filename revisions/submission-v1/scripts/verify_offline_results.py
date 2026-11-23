"""Recompute headline outcomes and core paired/oracle intervals from saved counts.

No model training, score inference, flow download or confirmation tuning occurs.
"""
from pathlib import Path
import csv,json,hashlib,random,os
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import numpy as np
R=Path(__file__).resolve().parents[3];V=R/'revisions/submission-v1';O=V/'results/offline-reproduction';O.mkdir(parents=True,exist_ok=False);I=V/'results/offline-inputs';meta=json.loads((I/'metadata.json').read_text());assert hashlib.sha256((I/'class-date-mean-counts.npz').read_bytes()).hexdigest()==meta['output_sha256'];dates=meta['calendar_dates'];valid=np.asarray([d in set(meta['observed_dates']) for d in dates]);alphas=[.01,.02,.03,.04,.05,.06,.07,.08,.10];methods=['Fixed global','Fixed classwise','Static HQR','Rolling global','Dynamic HQR','Multi-anchor'];cubes={}
with np.load(I/'class-date-mean-counts.npz',allow_pickle=False) as f:cubes={k:f[k] for k in f.files}
def cube(part,method,alpha=.05):return cubes[f'{part}__{method.replace(" ","_")}__a{round(alpha*10000):04}']
def weights(block):
 rng=np.random.default_rng(20261122);out=[]
 for b in range(2000):
  starts=rng.integers(0,len(dates)-block+1,size=(len(dates)+block-1)//block);idx=[d for start in starts for d in range(int(start),int(start)+block)][:len(dates)];out.append(np.bincount(idx,minlength=len(dates)))
 return np.asarray(out)
def masks(part):
 k=np.isin(np.arange(53),meta[part+'_known_codes']);b=np.isin(np.arange(53),[0,51,52]);return {'known':k,'background':b,'heldout':~(k|b),'all_unknown':~k}
def measure(a,w,part):
 masks_=masks(part);n=a[:,:,0];j=a[:,:,1];nn=w@n;jj=w@j;r=np.divide(jj,nn,out=np.full_like(jj,np.nan),where=nn>0);out={}
 for name in ['background','heldout','all_unknown']:
  group=masks_[name];out[name+'_macro_detection']=np.nanmean(r[:,group],axis=1);out[name+'_micro_detection']=jj[:,group].sum(1)/nn[:,group].sum(1)
 known=masks_['known'];dn=n[:,known].sum(1);good=dn>0
 for name,ch in [('known_frr',1),('known_correct_accepted',2)]:
  daily=np.divide(a[:,known,ch].sum(1),dn,out=np.zeros_like(dn),where=good);out[name]=(w@daily)/(w@good)
 return out
one=np.ones((1,len(dates)));points={};primary=[];sumchecks=0
source_old=list(csv.DictReader((R/'revisions/major-v1/results/static-hqr/six-method-comparison.csv').open()));source_new=list(csv.DictReader((R/'revisions/major-v1/results/new-partition-v2/grid-means.csv').open()))
for part in ['original','second']:
 for m in methods:
  a=cube(part,m);assert int(a[:,:,0].sum())==1417414 and np.count_nonzero(a[:,:,0].sum(1))==238;z=measure(a,one,part);points[(part,m)]=z
  if part=='original':
   s=next(x for x in source_old if x['method']==m and x['score']=='negative_energy' and x['comparison_budget']=='32' and x['comparison_delay']=='7' and x['view']=='all');fields={'known_frr':'mean_daily_known_frr','known_correct_accepted':'mean_daily_known_correct_accepted','background_macro_detection':'period_background_macro_rejection','heldout_macro_detection':'period_heldout_macro_rejection'}
  else:s=next(x for x in source_new if x['method']==m and x['score']=='negative_energy' and x['budget']=='32' and x['delay']=='7');fields={k:k for k in z}
  for k,f in fields.items():assert abs(z[k][0]-float(s[f]))<2e-12;(None);sumchecks+=1
  primary.append({'partition':part,'method':m,**{k:float(x[0]) for k,x in z.items()}})
A=json.loads((R/'outputs/confirmation-analysis-v1/analysis.json').read_text());N=json.loads((R/'revisions/major-v1/results/new-partition-v2/complete.json').read_text());intervals=[];maxerr=0.
for block in [7,28,56]:
 w=weights(block)
 for part in ['original','second']:
  rr=measure(cube(part,'Rolling global'),w,part);hh=measure(cube(part,'Dynamic HQR'),w,part)
  for k in ['known_frr','known_correct_accepted','background_macro_detection','heldout_macro_detection']:
   delta=hh[k]-rr[k];bounds=np.quantile(delta,[.025,.975]);d=points[(part,'Dynamic HQR')][k][0]-points[(part,'Rolling global')][k][0]
   if part=='original':s=next(x for x in A['contrasts'] if x['view']=='all' and x['comparator']=='raw:rolling_global' and x['block_days']==block and x['metric']==k)
   else:s=next(x for x in N['primary_contrasts'] if x['candidate']=='Dynamic HQR' and x['comparator']=='Rolling global' and x['block_days']==block and x['metric']==k)
   e=max(abs(float(d)-s['difference']),float(np.max(abs(bounds-s['interval_95']))));assert e<2e-12;maxerr=max(maxerr,e);intervals.append({'partition':part,'metric':k,'block_days':block,'difference':float(d),'lower':float(bounds[0]),'upper':float(bounds[1]),'saved_reference_max_error':e})
 print(json.dumps({'primary_block_reproduced':block}),flush=True)
# Oracle matching: same declared adjacent-segment preference, never extrapolate.
w=weights(28);curves={};frrname='known_frr'
for m in methods:
 draws=[measure(cube('original',m,a),w,'original') for a in alphas];curves[m]={k:np.stack([z[k] for z in draws],1) for k in draws[0]}
def match(curve,target):
 x=curve[frrname];answer={k:np.full(len(x),np.nan) for k in ['background_macro_detection','heldout_macro_detection','all_unknown_macro_detection']};assigned=np.zeros(len(x),bool)
 for i in sorted(range(8),key=lambda i:abs((alphas[i]+alphas[i+1])/2-target)):
  left=x[:,i];right=x[:,i+1];take=(~assigned)&(np.minimum(left,right)<=target)&(np.maximum(left,right)>=target)&(left!=right);fraction=np.divide(target-left,right-left,out=np.zeros_like(left),where=left!=right)
  for k in answer:answer[k][take]=(curve[k][:,i]+fraction*(curve[k][:,i+1]-curve[k][:,i]))[take]
  assigned|=take
 return answer
oracle=list(csv.DictReader((R/'revisions/major-v1/results/oracle-risk-bootstrap-v2/paired-oracle-contrasts.csv').open()));oc=[]
for x in oracle:
 if x['block_days']!='28':continue
 a=match(curves[x['candidate']],float(x['target_actual_frr']));b=match(curves[x['comparator']],float(x['target_actual_frr']));delta=a[x['metric']]-b[x['metric']];good=np.isfinite(delta);lo,hi=np.quantile(delta[good],[.025,.975]);e=max(abs(lo-float(x['lower'])),abs(hi-float(x['upper'])));assert e<2e-12;oc.append({**x,'recomputed_lower':float(lo),'recomputed_upper':float(hi),'max_error':e})
# Recreate service selection from source-support counts with Python's actual rule.
roles=[];support=json.loads((R/'outputs/pilot-transformer-v1/prepare-complete.json').read_text())['source_counts'];apps=json.loads((R/'outputs/inventory.json').read_text())['apps'];bg={0,51,52};eligible=sorted(int(k) for k,n in support.items() if int(k) not in bg and int(n)>=100);assert len(eligible)==44
for part,seed in [('original',20261121),('second',20261122)]:
 known=sorted(random.Random(seed).sample(eligible,30));assert known==meta[part+'_known_codes']
 for name,c in sorted(apps.items(),key=lambda x:int(x[1])):roles.append({'partition':part,'code':c,'service':name,'role':'K' if c in known else 'B' if c in bg else 'H'})
def write(n,rr):
 with (O/n).open('w',newline='') as f:d=csv.DictWriter(f,fieldnames=list(rr[0]));d.writeheader();d.writerows(rr)
write('primary-metrics.csv',primary);write('primary-intervals.csv',intervals);write('oracle-interval-checks.csv',oc);write('reconstructed-service-roles.csv',roles)
report={'status':'PASS_ACTUAL_SAVED_DECISION_RECOMPUTATION','unique_flow_count':1417414,'observed_dates':238,'calendar_positions':243,'method_partition_points':len(primary),'headline_point_reference_checks':sumchecks,'primary_paired_intervals_7_28_56':len(intervals),'max_primary_difference':maxerr,'oracle_28day_interval_checks':len(oc),'both_service_selections_reconstructed_from_source_support':True,'known_per_partition':30,'class_list_rows':len(roles),'scope':'Numerators/denominators and temporal bootstrap regenerated; no training, scoreinference or labelpolicy rerun.'};(O/'complete.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
