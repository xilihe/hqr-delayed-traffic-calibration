"""Independent CSV/cell arithmetic and directly expanded calendar-bootstrap draws."""
from pathlib import Path
from collections import defaultdict
import csv,json,os,gzip
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import numpy as np
R=Path(__file__).resolve().parents[3];V=R/'revisions/submission-v1';S=V/'results/risk-tracking'
P=json.loads((R/'research/frozen/confirmation-v1.json').read_text());from datetime import datetime,timedelta
start=datetime.strptime(P['confirmation_start'],'%Y%m%d');end=datetime.strptime(P['confirmation_end'],'%Y%m%d');dates=[(start+timedelta(days=i)).strftime('%Y%m%d') for i in range((end-start).days+1)];di={d:i for i,d in enumerate(dates)}
rows=list(csv.DictReader((S/'daily-known-counts.csv').open()));groups=defaultdict(list);known_den={}
for r in rows:
 n=int(r['known_n']);j=int(r['rejected_known_n']);assert n>0 and 0<=j<=n and abs(j/n-float(r['known_frr']))<1e-15
 k=tuple(r[t] for t in ['partition','regime','method','model_seed','audit_seed']);groups[k].append(r);nk=(r['partition'],r['date']);assert nk not in known_den or known_den[nk]==n;known_den[nk]=n
# Independent known denominators from saved per-class decisions, not rate CSV.
original={}
with gzip.open(R/'outputs/confirmation-analysis-v1/primary-daily-counts.jsonl.gz','rt') as f:
 for line in f:
  x=json.loads(line)
  if x['view']=='all' and x['method']=='raw:rolling_global':
   n=sum(c['n'] for c in x['class_counts'] if c['known']);key=x['date'];assert key not in original or original[key]==n;original[key]=n
for d,n in original.items():assert known_den[('20261121',d)]==n
N=json.loads((R/'revisions/major-v1/new-partition-plan.json').read_text());
with np.load(V/'results/offline-inputs/class-date-mean-counts.npz',allow_pickle=False) as f:
 a=f['second__Rolling_global__a0500'];assert a.shape[0]==len(dates)
 for day in range(len(dates)):
  n=int(a[day,N['known_codes'],0].sum())
  if n:assert known_den[('20261122',dates[day])]==n
# Construct block sequences separately; do not import the production helper.
rng=np.random.default_rng(20261122);expanded=[]
for draw in range(2000):
 starts=rng.integers(0,len(dates)-28+1,size=(len(dates)+27)//28);indices=[]
 for x in starts:indices.extend(range(int(x),int(x)+28))
 expanded.append(np.asarray(indices[:len(dates)]))
with np.load(S/'bootstrap-draws.npz',allow_pickle=False) as f:archive={k:f[k] for k in f.files}
w=np.stack([np.bincount(x,minlength=len(dates)) for x in expanded]);assert np.array_equal(w,archive['calendar_weights'])
keylist=archive['keys'].tolist();names=archive['metrics'].tolist();cellboots=defaultdict(list);cellpoints=defaultdict(list);observed_sets=[]
for ci,(key,rr) in enumerate(sorted(groups.items())):
 rr=sorted(rr,key=lambda x:x['date']);observed=np.array([di[x['date']] for x in rr]);assert len(observed)==238 and len(set(observed))==238;valid=np.zeros(len(dates),bool);valid[observed]=True;observed_sets.append(valid);rates=np.array([int(x['rejected_known_n'])/int(x['known_n']) for x in rr]);e=rates-.05;ww=w[:,observed];den=ww.sum(1);out=np.empty((2000,len(names)));pt={}
 values={'mean_daily_known_frr':rates,'mean_absolute_target_deviation':abs(e),'rmse_target_deviation':e**2,'above_5p5_fraction':(rates>.055).astype(float),'above_6p0_fraction':(rates>.06).astype(float)}
 for metric,val in values.items():
  point=float(val.mean());boot=(ww@val)/den
  if metric=='rmse_target_deviation':point=point**.5;boot=np.sqrt(boot)
  pt[metric]=point;out[:,names.index(metric)]=boot
 for pref in ['above_5p5','above_6p0']:pt[pref+'_days']=238*pt[pref+'_fraction'];out[:,names.index(pref+'_days')]=238*out[:,names.index(pref+'_fraction')]
 pt['daily_frr_p90']=float(np.quantile(rates,.9));lookup=np.full(len(dates),np.nan);lookup[observed]=rates
 # Direct expansion, deleting missing calendar dates before quantile evaluation.
 for b,idx in enumerate(expanded):
  vals=lookup[idx];vals=vals[np.isfinite(vals)];out[b,names.index('daily_frr_p90')]=np.quantile(vals,.9)
 k='__'.join([key[0],key[1],key[2].replace(' ','_')]);cellboots[k].append(out);cellpoints[k].append([pt[n] for n in names])
 if (ci+1)%24==0:print(json.dumps({'independent_cells_complete':ci+1,'total':len(groups)}),flush=True)
boots={k:np.mean(a,axis=0) for k,a in cellboots.items()};points={k:np.mean(a,axis=0) for k,a in cellpoints.items()};maxerror=0.
for k,a in boots.items():
 i=keylist.index(k);error=float(np.max(abs(a-archive['corrected_metrics'][i])));assert error<2e-12,(k,error);maxerror=max(maxerror,error)
for x in csv.DictReader((S/'summary.csv').open()):
 k='__'.join([x['partition'],x['regime'],x['method'].replace(' ','_')]);mi=names.index(x['metric']);bounds=np.quantile(boots[k][:,mi],[.025,.975]);assert np.allclose([points[k][mi],*bounds],[float(x[t]) for t in ['value','lower','upper']],rtol=0,atol=2e-12)
for x in csv.DictReader((S/'paired-contrasts.csv').open()):
 a='__'.join([x['partition'],x['regime'],'Dynamic_HQR']);b='__'.join([x['partition'],x['regime'],x['comparator'].replace(' ','_')]);mi=names.index(x['metric']);bounds=np.quantile(boots[a][:,mi]-boots[b][:,mi],[.025,.975]);assert np.allclose([points[a][mi]-points[b][mi],*bounds],[float(x[t]) for t in ['difference','lower','upper']],rtol=0,atol=2e-12)
report={'status':'PASS_INDEPENDENT_FULL_DAILY_RISK_BOOTSTRAP','daily_rows_checked':len(rows),'fixed_cells':len(groups),'known_denominator_sources':['OriginalsavedperclassJSONL','Secondpartitionsavedclass/date-counts'],'calendar_sequences_independently_constructed':2000,'all192summary_points_and_intervals_checked':True,'all160paired_points_and_intervals_checked':True,'p90_method':'Directexpandedresamples, missingdatesremoved; linear np.quantile percell beforeaveraging','max_draw_absolute_difference':maxerror,'corrected_count_draws_within_0_238':bool(np.all(archive['corrected_metrics'][:,:,[names.index('above_5p5_days'),names.index('above_6p0_days')]]>=0) and np.all(archive['corrected_metrics'][:,:,[names.index('above_5p5_days'),names.index('above_6p0_days')]]<=238))};(V/'results/independent-risk-verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
