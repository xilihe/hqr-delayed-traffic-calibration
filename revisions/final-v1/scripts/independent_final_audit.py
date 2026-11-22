"""Independent arithmetic checks of final diagnostics; no policy tuning."""
from pathlib import Path
import csv,json,math,statistics,hashlib,collections,sys
R=Path(__file__).resolve().parents[3];V=R/'revisions/final-v1'
def rows(p):return list(csv.DictReader(p.open()))
def close(a,b):
 assert math.isclose(float(a),float(b),abs_tol=2e-12,rel_tol=1e-10),(a,b)
def q90(v):
 s=sorted(v);p=.9*(len(s)-1);i=int(p);return s[i]+(p-i)*(s[math.ceil(p)]-s[i])
groups=collections.defaultdict(list)
for r in rows(V/'results/risk-tracking/daily-known-counts.csv'):
 n=int(r['known_n']);j=int(r['rejected_known_n']);assert 0<=j<=n;close(j/n,r['known_frr']);groups[tuple(r[k] for k in ['partition','regime','method','model_seed','audit_seed'])].append(j/n)
recomputed=collections.defaultdict(list)
for key,v in groups.items():
 assert len(v)==238
 d={'mean_daily_known_frr':statistics.mean(v),'mean_absolute_target_deviation':statistics.mean(abs(x-.05) for x in v),'rmse_target_deviation':math.sqrt(statistics.mean((x-.05)**2 for x in v)),'daily_frr_p90':q90(v)}
 for t,s in [(.055,'above_5p5'),(.06,'above_6p0')]:d[s+'_days']=sum(x>t for x in v);d[s+'_fraction']=d[s+'_days']/238
 for k,x in d.items():recomputed[key[:3]+(k,)].append(x)
for r in rows(V/'results/risk-tracking/summary.csv'):close(statistics.mean(recomputed[tuple(r[k] for k in ['partition','regime','method','metric'])]),r['value'])
for r in rows(V/'results/risk-tracking/paired-contrasts.csv'):
 a=tuple(r[k] for k in ['partition','regime'])
 close(statistics.mean(recomputed[a+('Dynamic HQR',r['metric'])])-statistics.mean(recomputed[a+(r['comparator'],r['metric'])]),r['difference'])
fb=rows(V/'results/partition-attribution/facebook-compact-attribution.csv')
for partition in ['20261121','20261122']:
 g=[r for r in fb if r['partition']==partition];close(sum(float(x['prediction_share']) for x in g),1)
 for r in g:close(float(r['prediction_share'])*(float(r['hqr_within_predicted_rejection'])-float(r['raw_within_predicted_rejection'])),r['contribution_to_true_class_change'])
 if partition=='20261122':
  source=next(r for r in rows(V/'results/partition-attribution/second-partition-all-class-effects.csv') if r['service']=='facebook-background');close(sum(float(r['contribution_to_true_class_change']) for r in g),source['difference'])
cl=rows(V/'results/partition-attribution/second-partition-all-class-effects.csv')
prior=json.loads((R/'revisions/major-v1/results/new-partition-v2/complete.json').read_text())['primary_contrasts']
for role,metric in [('B','background_macro_detection'),('H','heldout_macro_detection')]:
 expected=next(x['difference'] for x in prior if x['candidate']=='Dynamic HQR' and x['comparator']=='Rolling global' and x['metric']==metric and x['block_days']==28)
 close(statistics.mean(float(r['difference']) for r in cl if r['role']==role),expected)
a=R/'research/verification/aci-selection.json';b=V/'results/aci-selection-reconstructed.json';assert a.read_bytes()==b.read_bytes()
frozen=R/'research/frozen/confirmation-v1.json';assert hashlib.sha256(frozen.read_bytes()).hexdigest()=='34f6f6a981cfd11bd202a58cd95705c86d05e566b5f51082dd45a63e1d335ac0'
report={'status':'PASS_INDEPENDENT_ARITHMETIC','daily_rows':sum(len(x) for x in groups.values()),'fixed_cells':len(groups),'risk_summary_values':len(recomputed),'paired_contrasts':160,'facebook_partition_share_and_contribution_identities':True,'aci_development_selection_byte_identical':True,'original_freeze_unchanged':True,'scope':'Independent CSV arithmetic, not an independent dataset or a second classifier fit.'}
(V/'results/independent-final-audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
