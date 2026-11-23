"""Repair fixed238-day exceedance intervals from real saved daily decisions.

Original calendar draws, probabilities and frozen experiments remain unchanged.
No clipping is used. Retain raw calendar-weighted counts for audit only.
"""
from pathlib import Path
import os,sys,ast,csv,json,hashlib
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import numpy as np
R=Path(__file__).resolve().parents[3];V=R/'revisions/submission-v1';S=R/'revisions/final-v1/results/risk-tracking';O=V/'results/risk-tracking';O.mkdir(parents=True,exist_ok=False)
sys.path.insert(0,str(R/'scripts'));from analyse_confirmation import block_weights,calendar
# Load only the old metric function, not the experiment/training/selection code.
source=R/'revisions/final-v1/scripts/risk_tracking.py';tree=ast.parse(source.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='metrics');module=ast.Module(body=[node],type_ignores=[]);namespace={'np':np};exec(compile(ast.fix_missing_locations(module),str(source),'exec'),namespace);old_metrics=namespace['metrics']
P=json.loads((R/'research/frozen/confirmation-v1.json').read_text());dates=calendar(P['confirmation_start'],P['confirmation_end']);weights=block_weights(len(dates),28,2000,20261122);oldsummary=list(csv.DictReader((S/'summary.csv').open()));oldcontrasts=list(csv.DictReader((S/'paired-contrasts.csv').open()));index={(x['partition'],x['regime'],x['method'],x['metric']):x for x in oldsummary};metrics=list(dict.fromkeys(x['metric'] for x in oldsummary));keys=[];point=[];oldboot=[];newboot=[];observed_den=[];summary=[];changes=[];den_diagnostics=[];maxerror=0.
source_npz=S/'daily-known-counts.npz'
if source_npz.exists():
 with np.load(source_npz,allow_pickle=False) as f:saved_counts={k:f[k] for k in f.files}
else:
 from collections import OrderedDict
 grouped=OrderedDict()
 for row in csv.DictReader((S/'daily-known-counts.csv').open()):
  key=row['partition']+'__'+row['regime']+'__'+row['method'].replace(' ','_');cell=(row['model_seed'],row['audit_seed']);grouped.setdefault(key,OrderedDict());grouped[key].setdefault(cell,np.zeros((len(dates),2),np.int64));grouped[key][cell][dates.index(row['date'])]=[int(row['known_n']),int(row['rejected_known_n'])]
 saved_counts={k:np.asarray(list(x.values())) for k,x in grouped.items()}
for key,data in saved_counts.items():
 partition,regime,name=key.split('__');method=name.replace('_',' ');valid=data[:,:,0]>0;assert np.all(valid.sum(1)==238) and np.all(valid==valid[0]);assert np.all(data[:,:,1]<=data[:,:,0]);p=old_metrics(data,np.ones((1,len(dates))));b=old_metrics(data,weights);fixed={k:v.copy() for k,v in b.items()}
 for prefix in ['above_5p5','above_6p0']:fixed[prefix+'_days']=238*fixed[prefix+'_fraction']
 den=weights@valid.T;observed_den.append(den[:,0]);keys.append(key);point.append([p[k][0] for k in metrics]);oldboot.append(np.stack([b[k] for k in metrics],axis=1));newboot.append(np.stack([fixed[k] for k in metrics],axis=1));missing=[dates[i] for i in np.flatnonzero(~valid[0])]
 den_diagnostics.append({'partition':partition,'regime':regime,'method':method,'fixed_cells':len(data),'observed_dates':238,'calendar_dates':len(dates),'missing_dates':missing,'bootstrap_observed_weight_min':float(den.min()),'bootstrap_observed_weight_max':float(den.max()),'bootstrap_calendar_weight':int(weights.sum(1)[0])})
 for k in metrics:
  old=index[(partition,regime,method,k)];lo,hi=np.quantile(b[k],[.025,.975]);expected=np.array([float(old[x]) for x in ['value','lower','upper']]);actual=np.array([p[k][0],lo,hi]);error=np.max(abs(expected-actual));maxerror=max(maxerror,float(error));assert np.allclose(expected,actual,rtol=0,atol=3e-14),(key,k,error)
  nlo,nhi=np.quantile(fixed[k],[.025,.975]);row=old.copy();row.update(value=float(p[k][0]),lower=float(nlo),upper=float(nhi));summary.append(row)
  if k.endswith('_days'):
   assert 0<=nlo<=nhi<=238
   changes.append(dict(partition=partition,regime=regime,method=method,metric=k,point=float(p[k][0]),old_lower=float(lo),old_upper=float(hi),corrected_lower=float(nlo),corrected_upper=float(nhi),old_draws_above238=int(np.sum(b[k]>238)),corrected_draws_above238=int(np.sum(fixed[k]>238))))
points=np.asarray(point);oldboot=np.asarray(oldboot);newboot=np.asarray(newboot);contrasts=[];old_pair_error=0.
for x in oldcontrasts:
 a=keys.index(x['partition']+'__'+x['regime']+'__Dynamic_HQR');c=keys.index(x['partition']+'__'+x['regime']+'__'+x['comparator'].replace(' ','_'));mi=metrics.index(x['metric']);delta=oldboot[a,:,mi]-oldboot[c,:,mi];lo,hi=np.quantile(delta,[.025,.975]);error=np.max(abs(np.array([points[a,mi]-points[c,mi],lo,hi])-np.array([float(x[k]) for k in ['difference','lower','upper']])));old_pair_error=max(old_pair_error,float(error));assert error<3e-14
 newdelta=newboot[a,:,mi]-newboot[c,:,mi];lo,hi=np.quantile(newdelta,[.025,.975]);row=x.copy();row.update(difference=float(points[a,mi]-points[c,mi]),lower=float(lo),upper=float(hi));contrasts.append(row)
def write(name,rows):
 with (O/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
write('summary.csv',summary);write('paired-contrasts.csv',contrasts);write('count-interval-changes.csv',changes);write('bootstrap-denominator-diagnostics.csv',den_diagnostics)
np.savez_compressed(O/'bootstrap-draws.npz',keys=np.asarray(keys),metrics=np.asarray(metrics),points=points,calendar_weights=weights,observed_day_weight=np.asarray(observed_den),old_metrics=oldboot,corrected_metrics=newboot)
# Source counts, original selection and integer per-cell counts are never altered.
import shutil
for n in ['daily-known-counts.csv','per-cell-summary.csv','development-selected-rules.json','new-partition-development-grid.csv','fixed-classwise-oracle-contrasts.csv']:shutil.copy2(S/n,O/n)
np.savez_compressed(O/'daily-known-counts.npz',**saved_counts)
noncount=[k for k in metrics if not k.endswith('_days')];same=all(np.array_equal(oldboot[:,:,metrics.index(k)],newboot[:,:,metrics.index(k)]) for k in noncount);assert same
report={'status':'CORRECTED_FIXED238_DAY_SCALE_NO_CLIPPING','classification':'Exceedance count interval estimand/presentation mismatch; calendar block algorithm and probability summaries were valid.','original_summary_rows_reproduced':len(oldsummary),'original_paired_rows_reproduced':len(oldcontrasts),'old_summary_max_absolute_error':maxerror,'old_pair_max_absolute_error':old_pair_error,'calendar_days':len(dates),'observed_days':238,'missing_dates':missing,'observed_weight_min':float(min(x['bootstrap_observed_weight_min'] for x in den_diagnostics)),'observed_weight_max':float(max(x['bootstrap_observed_weight_max'] for x in den_diagnostics)),'fixed_scale_day_count_rows':len(changes),'all_point_estimates_unchanged':True,'all_noncount_draws_and_intervals_bitwise_unchanged':same,'changed_summary_intervals':sum(float(x['lower'])!=float(index[(x['partition'],x['regime'],x['method'],x['metric'])]['lower']) or float(x['upper'])!=float(index[(x['partition'],x['regime'],x['method'],x['metric'])]['upper']) for x in summary),'bootstrap_seed':20261122,'replicates':2000,'block_calendar_days':28,'source_daily_counts_csv_sha256':hashlib.sha256((S/'daily-known-counts.csv').read_bytes()).hexdigest(), 'saved_npz_used':source_npz.exists(),'raw_weighted_counts_retained_for_audit':True,'no_training_or_policy_replay':True};(O/'complete.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
