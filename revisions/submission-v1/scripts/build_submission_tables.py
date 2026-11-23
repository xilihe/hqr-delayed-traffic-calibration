"""Reproduce S23-S31; only fixed-scale count intervals/captions change."""
from pathlib import Path
import csv,json,hashlib,shutil
R=Path(__file__).resolve().parents[3];V=R/'revisions/submission-v1';T=V/'manuscript/tables';T.mkdir(parents=True,exist_ok=True)
base=R/'revisions/final-v1/manuscript/tables'
for p in base.glob('final-risk-*.tex'):shutil.copy2(p,T/p.name)
rows=list(csv.DictReader((V/'results/risk-tracking/summary.csv').open()));methods=['Fixed global','Fixed classwise','Static HQR','Rolling global','Dynamic HQR','Multi-anchor'];changed=[]
for pi,partition in enumerate(['20261121','20261122']):
 for ri,regime in enumerate(['nominal_0.05','development_selected']):
  tag=f'final-risk-{pi}-{ri}-exceed';reg=r'nominal $\alpha=0.05$' if ri==0 else r'development-selected $\alpha$ for target5\%';reg=reg.replace('target5','target 5');index={(x['method'],x['metric']):x for x in rows if x['partition']==partition and x['regime']==regime};headers=['Method',r'\% days$>5.5\%$',r'Equiv. days',r'\% days$>6\%$',r'Equiv. days'];caption=f'Daily exceedance diagnostics: partition {partition}, {reg}. Proportions are reported directly. Point counts average fixed-cell counts over238 observed dates. Day intervals use238 times each bootstrap exceedance proportion, a fixed238-day equivalent scale; no clipping. Strict exceedance; paired28-calendar-day resampling.'
  for a,b in [('over238','over 238'),('use238','use 238'),('fixed238','fixed 238'),('paired28','paired 28')]:caption=caption.replace(a,b)
  text=r'\begingroup\small\setlength{\tabcolsep}{3pt}'+'\n'+r'\begin{longtable}{lrrrr}'+'\n'+r'\caption{'+caption+r'}\label{tab:'+tag+r'}\\'+'\n\\toprule\n'+' & '.join(headers)+r'\\'+'\n\\midrule\n\\endfirsthead\n\\toprule\n'+' & '.join(headers)+r'\\'+'\n\\midrule\n\\endhead\n'
  for method in methods:
   cells=[method]
   for metric in ['above_5p5_fraction','above_5p5_days','above_6p0_fraction','above_6p0_days']:
    x=index[(method,metric)];scale=100 if metric.endswith('fraction') else 1;cells.append('%.2f [%.2f, %.2f]'%tuple(float(x[k])*scale for k in ['value','lower','upper']))
   text+=' & '.join(cells)+r'\\'+'\n'
  text+='\\bottomrule\n\\end{longtable}\n\\endgroup\n';p=T/(tag+'.tex');p.write_text(text);changed.append(p.name)
unchanged=[]
for p in T.glob('final-risk-*.tex'):
 if p.name not in changed:assert p.read_bytes()==(base/p.name).read_bytes();unchanged.append(p.name)
report={'status':'S23_S31_GENERATED_FROM_CORRECTED_COUNTS','changed_tables':['S24','S26','S28','S30'],'unchanged_tables':['S23','S25','S27','S29','S31'],'point_estimates_unchanged':True,'changed_tex_files':changed,'unchanged_tex_files':unchanged,'source_sha256':hashlib.sha256((V/'results/risk-tracking/summary.csv').read_bytes()).hexdigest()};(V/'results/table-generation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
