"""Exploratory paired calendar-block intervals for descriptive oracle matching.

All methods use the same date weights. Cell-mean numerators can be averaged
before class ratios only because the class denominators are exactly identical.
"""
from pathlib import Path
import csv,json,os,sys
os.environ.setdefault('OPENBLAS_NUM_THREADS','2');os.environ.setdefault('OMP_NUM_THREADS','2')
import numpy as np
ROOT=Path(__file__).resolve().parents[3];BASE=ROOT/'revisions/major-v1/results/risk-curves-v2';OUT=ROOT/'revisions/major-v1/results/oracle-risk-bootstrap-v2';OUT.mkdir(exist_ok=False)
sys.path.insert(0,str(ROOT/'scripts'))
from analyse_confirmation import calendar,block_weights
P=json.loads((ROOT/'research/frozen/confirmation-v1.json').read_text());OLD=json.loads((ROOT/'outputs/confirmation-analysis-v1/analysis.json').read_text());dates=calendar(P['confirmation_start'],P['confirmation_end']);obs=[p.rsplit('/D',1)[1] for p in P['confirmation_tables']];positions=[dates.index(d) for d in obs];alphas=[.01,.02,.03,.04,.05,.06,.07,.08,.10]
labels={'Fixed global':('raw','fixed_global'),'Fixed classwise':('raw','fixed_classwise'),'Rolling global':('raw','rolling_global'),'Static HQR':('shift','fixed_global'),'Dynamic HQR':('shift','rolling_global'),'Multi-anchor':('reference','rolling_global')}
known=np.array([c in P['known_codes'] for c in range(53)]);bg=np.array([c in {0,51,52} for c in range(53)]);held=~(known|bg);masks={'background_macro_detection':bg,'heldout_macro_detection':held,'all_unknown_macro_detection':~known}
cubes={};nref=None
with np.load(BASE/'risk-grid-class-counts.npz',allow_pickle=False) as f:
 for label,(variant,method) in labels.items():
  for alpha in alphas:
   prefix=f'-{variant}-{method}-a{round(alpha*10000):04}-';arrays=[f[k] for k in f.files if prefix in k];assert len(arrays)==(3 if method.startswith('fixed_') else 9)
   n=arrays[0][:,:,0];assert all(np.array_equal(a[:,:,0],n) for a in arrays)
   if nref is None:nref=n
   else:assert np.array_equal(n,nref)
   cube=np.zeros((len(dates),53,3));cube[positions]=np.mean(arrays,axis=0);cubes[(label,alpha)]=cube
N=cubes[('Rolling global',.05)][:,:,0];dn=N[:,known].sum(1);valid_days=dn>0

def draws(weights,cube):
 total_n=weights@N;rejected=weights@cube[:,:,1];rate=np.divide(rejected,total_n,out=np.full_like(rejected,np.nan),where=total_n>0);out={}
 for name,mask in masks.items():out[name]=np.nanmean(rate[:,mask],axis=1)
 for name,ch in [('known_frr',1),('known_correct_accepted',2)]:
  daily=np.divide(cube[:,known,ch].sum(1),dn,out=np.zeros_like(dn),where=valid_days);out[name]=(weights@daily)/(weights@valid_days.astype(float))
 return out

def match(curve,target):
 count=curve['known_frr'].shape[0];out={k:np.full(count,np.nan) for k in curve if k!='known_frr'}
 # Same outcome-independent segment preference as the point estimator.
 order=sorted(range(len(alphas)-1),key=lambda i:abs((alphas[i]+alphas[i+1])/2-target));assigned=np.zeros(count,bool)
 for i in order:
  a=curve['known_frr'][:,i];b=curve['known_frr'][:,i+1];take=(~assigned)&(np.minimum(a,b)<=target)&(np.maximum(a,b)>=target)&(a!=b);w=np.divide(target-a,b-a,out=np.zeros_like(a),where=b!=a)
  for metric in out:out[metric][take]=(curve[metric][:,i]+w*(curve[metric][:,i+1]-curve[metric][:,i]))[take]
  assigned|=take
 return out

points={key:draws(np.ones((1,len(dates))),cube) for key,cube in cubes.items()};matched_points={}
for label in labels:
 curve={k:np.stack([points[(label,a)][k] for a in alphas],axis=1) for k in next(iter(points.values()))}
 for risk in [.03,.05,.07]:matched_points[(label,risk)]=match(curve,risk)
rows=[];numeric_checks=[]
for block in [7,28,56]:
 weights=block_weights(len(dates),block,2000,20261122);boot={key:draws(weights,cube) for key,cube in cubes.items()}
 # Full-risk original contrasts are an independent reference for this new
 # algebraically reduced implementation, before doing any oracle matching.
 for metric in ['known_frr','known_correct_accepted','background_macro_detection','heldout_macro_detection']:
  delta=boot[('Dynamic HQR',.05)][metric]-boot[('Rolling global',.05)][metric];interval=np.quantile(delta,[.025,.975]);old=next(v for v in OLD['contrasts'] if v['view']=='all' and v['comparator']=='raw:rolling_global' and v['block_days']==block and v['metric']==metric);error=float(np.max(np.abs(interval-old['interval_95'])));assert error<1e-12;numeric_checks.append(dict(block=block,metric=metric,max_interval_difference=error))
 matched={}
 for label in labels:
  curve={k:np.stack([boot[(label,a)][k] for a in alphas],axis=1) for k in next(iter(boot.values()))}
  for risk in [.03,.05,.07]:matched[(label,risk)]=match(curve,risk)
 for candidate,comparator in [('Dynamic HQR','Rolling global'),('Static HQR','Rolling global'),('Dynamic HQR','Static HQR'),('Dynamic HQR','Fixed classwise'),('Static HQR','Fixed classwise')]:
  for risk in [.03,.05,.07]:
   for metric in masks:
    delta=matched[(candidate,risk)][metric]-matched[(comparator,risk)][metric];valid=np.isfinite(delta);bounds=np.quantile(delta[valid],[.025,.975]) if valid.any() else [np.nan,np.nan];point=float(matched_points[(candidate,risk)][metric][0]-matched_points[(comparator,risk)][metric][0]);rows.append(dict(candidate=candidate,comparator=comparator,target_actual_frr=risk,metric=metric,block_days=block,difference=point,lower=float(bounds[0]),upper=float(bounds[1]),valid_draws=int(valid.sum()),analysis_type='POST_CONFIRMATION_DESCRIPTIVE_ORACLE'))
 print(json.dumps(dict(oracle_bootstrap_block_complete=block)),flush=True)
with (OUT/'paired-oracle-contrasts.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(OUT/'complete.json').write_text(json.dumps({'status':'POST_CONFIRMATION_PAIRED_ORACLE_RISK_INTERVALS_COMPLETE','rows':len(rows),'original_interval_regression_checks':numeric_checks,'date_weights_shared':True,'replications_are_fixed_not_independent_networks':True,'interpretation':'Confirmation actual FRR defines the interpolation; all matched values and intervals remain exploratory descriptive oracle quantities, not deployment guarantees or evidence of equivalence.','no_extrapolation':True,'insufficient_grid_support':'Invalid draws are excluded pairwise and valid draw counts are reported.'},indent=2)+'\n')
