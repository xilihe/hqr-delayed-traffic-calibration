"""Independent standard-library aggregation of saved new-split class counts."""
from pathlib import Path
import csv, hashlib, json, math, statistics
import numpy as np
R=Path(__file__).resolve().parents[3];D=R/'revisions/major-v1/results/new-partition-v2'
plan=json.loads((R/'revisions/major-v1/new-partition-plan.json').read_text());known=set(plan['known_codes']);background={0,51,52};heldout=set(range(53))-known-background
with (D/'grid-means.csv').open() as f:expected={r['method']:r for r in csv.DictReader(f) if r['score']=='negative_energy' and r['budget']=='32' and r['delay']=='7'}
with np.load(D/'primary-class-counts.npz',allow_pickle=False) as f:arrays={name:f[name].tolist() for name in f.files}
calculated={};errors=[];checks=0
for key,cube in arrays.items():
    method=key.replace('_',' ');cells=[]
    for cell in cube:
        dates=[day for day in cell if sum(row[0] for row in day)>0];assert len(dates)==238
        points={}
        for metric,column in [('known_frr',1),('known_correct_accepted',2)]:
            points[metric]=statistics.mean(sum(day[c][column] for c in known)/sum(day[c][0] for c in known) for day in dates)
        for population,classes in [('background',background),('heldout',heldout),('all_unknown',background|heldout)]:
            n={c:sum(day[c][0] for day in dates) for c in classes};j={c:sum(day[c][1] for day in dates) for c in classes};represented=[c for c in classes if n[c]>0]
            points[population+'_macro_detection']=statistics.mean(j[c]/n[c] for c in represented)
            points[population+'_micro_detection']=sum(j.values())/sum(n.values())
        cells.append(points)
    calculated[method]={metric:statistics.mean(c[metric] for c in cells) for metric in cells[0]}
    for metric,value in calculated[method].items():errors.append(abs(value-float(expected[method][metric])));checks+=1
report=json.loads((D/'complete.json').read_text())
for contrast in report['primary_contrasts']:
    value=calculated[contrast['candidate']][contrast['metric']]-calculated[contrast['comparator']][contrast['metric']]
    errors.append(abs(value-contrast['difference']));checks+=1
assert max(errors)<1e-12
out=dict(status='PASS',scope='Every six-method primary mean for all eight outcomes and every saved primary contrast point; independent Python sums and ratios, no replay.summarize or weighted_metrics call. Intervals use the separately regression-verified calendar bootstrap implementation.',checks=checks,maximum_absolute_error=max(errors),verified_points=calculated,count_file_sha256=hashlib.sha256((D/'primary-class-counts.npz').read_bytes()).hexdigest(),unique_observed_dates=238)
(D/'independent-aggregate-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='verified_points'}))
