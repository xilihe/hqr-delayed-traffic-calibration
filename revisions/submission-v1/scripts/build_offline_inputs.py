"""Build compact sufficient statistics from real saved class decisions.

Averaging numerators over fixed cells preserves the study's period ratios and
mean-daily known rates because every compared cell has identical denominators.
These are derived class/date counts, never raw packet or user records.
"""
from pathlib import Path
import json,csv,hashlib,os
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import numpy as np
R=Path(__file__).resolve().parents[3];V=R/'revisions/submission-v1';O=V/'results/offline-inputs';O.mkdir(parents=True,exist_ok=False)
P=json.loads((R/'research/frozen/confirmation-v1.json').read_text());N=json.loads((R/'revisions/major-v1/new-partition-plan.json').read_text());from datetime import datetime,timedelta
start=datetime.strptime(P['confirmation_start'],'%Y%m%d');dates=[(start+timedelta(days=i)).strftime('%Y%m%d') for i in range((datetime.strptime(P['confirmation_end'],'%Y%m%d')-start).days+1)];obs=[x.rsplit('/D',1)[1] for x in P['confirmation_tables']];pos=[dates.index(x) for x in obs];alphas=[.01,.02,.03,.04,.05,.06,.07,.08,.10];labels={'Fixed global':('raw','fixed_global'),'Fixed classwise':('raw','fixed_classwise'),'Rolling global':('raw','rolling_global'),'Static HQR':('shift','fixed_global'),'Dynamic HQR':('shift','rolling_global'),'Multi-anchor':('reference','rolling_global')};out={};references=[];nref=None
p=R/'revisions/major-v1/results/risk-curves-v2/risk-grid-class-counts.npz'
with np.load(p,allow_pickle=False) as f:
 for label,(variant,method) in labels.items():
  for alpha in alphas:
   prefix=f'-{variant}-{method}-a{round(alpha*10000):04}-';keys=[k for k in f.files if prefix in k];a=[f[k] for k in keys];assert len(a)==(3 if method.startswith('fixed_') else 9);nn=a[0][:,:,0];assert all(np.array_equal(x[:,:,0],nn) for x in a)
   if nref is None:nref=nn
   else:assert np.array_equal(nref,nn)
   cube=np.zeros((len(dates),53,3));cube[pos]=np.mean(a,axis=0);out[f'original__{label.replace(" ","_")}__a{round(alpha*10000):04}']=cube;references.append({'partition':'original','method':label,'alpha':alpha,'fixed_cells':len(a),'all_cell_denominators_identical':True})
p2=R/'revisions/major-v1/results/new-partition-v2/primary-class-counts.npz'
with np.load(p2,allow_pickle=False) as f:
 for label in labels:
  a=f[label.replace(' ','_')];assert len(a)==9 and all(np.array_equal(x[:,:,0],a[0,:,:,0]) for x in a);cube=a.mean(0);assert np.array_equal(cube[pos,:,0],nref);out['second__'+label.replace(' ','_')+'__a0500']=cube;references.append({'partition':'second','method':label,'alpha':.05,'fixed_cells':3 if label in ['Fixed global','Fixed classwise','Static HQR'] else 9,'all_cell_denominators_identical':True,'fixed_repeated_audit_values_collapsed_without_new_samples':True})
for a in out.values():assert int(a[:,:,0].sum())==1417414 and np.count_nonzero(a[:,:,0].sum(1))==238
np.savez_compressed(O/'class-date-mean-counts.npz',**out)
meta={'status':'DERIVED_SUFFICIENT_STATISTICS_FROM_SAVED_DECISIONS','calendar_dates':dates,'observed_dates':obs,'calendar_days':len(dates),'observed_days':len(obs),'unique_confirmation_flows':1417414,'original_known_codes':P['known_codes'],'second_known_codes':N['known_codes'],'class_codes':list(range(53)),'channels':['n','rejected','correct_accepted'],'reference_cubes':references,'pooling_justification':'Identical actual class/date denominators across all fixedcells; mean numerators preserve all used fixedcell-average ratios and mean daily known rates exactly. Not independent-network replication.','source_hashes':{str(q.relative_to(R)):hashlib.sha256(q.read_bytes()).hexdigest() for q in [p,p2,R/'research/frozen/confirmation-v1.json',R/'revisions/major-v1/new-partition-plan.json']},'output_sha256':hashlib.sha256((O/'class-date-mean-counts.npz').read_bytes()).hexdigest()};(O/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps({'cubes':len(out),'bytes':(O/'class-date-mean-counts.npz').stat().st_size,'unique_flows':1417414,'observed_days':238}))
