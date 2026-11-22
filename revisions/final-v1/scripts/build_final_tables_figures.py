"""Final tables and vector figures, read solely from recorded result CSV files."""
from pathlib import Path
import csv,json,hashlib,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[3];V=R/'revisions/final-v1';T=V/'manuscript/tables';F=V/'figures';F.mkdir(parents=True,exist_ok=True);T.mkdir(parents=True,exist_ok=True);(V/'manuscript/figures').mkdir(parents=True,exist_ok=True)
def read(p):return list(csv.DictReader(p.open()))
def escape(s):return str(s).replace('_',r'\_').replace('%',r'\%')
def f(x):return f'{float(x)*100:.2f}'
def ci(r):return f"{f(r['value'])} [{f(r['lower'])}, {f(r['upper'])}]"
def table(name,caption,headers,rows,label,long=False):
 env='longtable' if long else 'tabular';spec='l'+'r'*(len(headers)-1)
 a=(r'\begingroup\small\setlength{\tabcolsep}{3pt}'+'\n')
 if long:a+=f'\\begin{{longtable}}{{{spec}}}\n\\caption{{{caption}}}\\label{{{label}}}\\\\\n'
 else:a+='\\begin{table}[tbp]\n\\centering\n'+f'\\caption{{{caption}}}\\label{{{label}}}\n\\begin{{tabular}}{{{spec}}}\n'
 a+='\\toprule\n'+' & '.join(headers)+r'\\'+'\n\\midrule\n'
 if long:a+='\\endfirsthead\n\\toprule\n'+' & '.join(headers)+r'\\'+'\n\\midrule\n\\endhead\n'
 a+='\n'.join(' & '.join(map(str,row))+r'\\' for row in rows)+'\n\\bottomrule\n'+f'\\end{{{env}}}\n'
 if not long:a+='\\end{table}\n'
 a+='\\endgroup\n';(T/(name+'.tex')).write_text(a)
oracle=read(V/'results/risk-tracking/fixed-classwise-oracle-contrasts.csv');rs=[]
for target in ['0.03','0.05','0.07']:
 row=[f(target)+r'\%']
 for metric in ['background_macro_detection','heldout_macro_detection','all_unknown_macro_detection']:
  x=next(x for x in oracle if x['target_actual_frr']==target and x['metric']==metric and x['block_days']=='28');row.append(f"{f(x['difference'])} [{f(x['lower'])}, {f(x['upper'])}]")
 rs.append(row)
table('final-fixedclass-oracle','Dynamic HQR minus Fixed classwise at descriptive confirmation-oracle FRR targets. Differences and paired 28-calendar-day 95\\% intervals are pp; no deployable rule or formal equivalence is implied.',['Target FRR','Background','Held-out','All unknown'],rs,'tab:fixedclass-oracle')
risk=read(V/'results/risk-tracking/summary.csv');methods=['Fixed global','Fixed classwise','Static HQR','Rolling global','Dynamic HQR','Multi-anchor'];inputs=[V/'results/risk-tracking/summary.csv']
for pi,p in enumerate(['20261121','20261122']):
 for ri,g in enumerate(['nominal_0.05','development_selected']):
  selected=[x for x in risk if x['partition']==p and x['regime']==g];index={(x['method'],x['metric']):x for x in selected};tag=f'final-risk-{pi}-{ri}';reg='nominal $\\alpha=0.05$' if ri==0 else 'development-selected $\\alpha$ for target 5\\%'
  rs=[]
  for m in methods:
   rs.append([m,next(x['alpha'] for x in selected if x['method']==m)]+[ci(index[(m,k)]) for k in ['mean_daily_known_frr','mean_absolute_target_deviation','rmse_target_deviation','daily_frr_p90']])
  table(tag,f'Daily risk diagnostics: partition {p}, {reg}. Mean FRR and p90 are percentages; MAE and RMSE are pp from 5\\%. Brackets are conditional 28-calendar-day 95\\% intervals.',['Method',r'$\alpha$','Mean FRR','MAE','RMSE','p90'],rs,'tab:'+tag,True)
  rs=[]
  for m in methods:
   row=[m]
   for k in ['above_5p5_days','above_5p5_fraction','above_6p0_days','above_6p0_fraction']:
    x=index[(m,k)];scale=1 if k.endswith('days') else 100
    row.append('%.2f [%.2f, %.2f]'%tuple(float(x[t])*scale for t in ['value','lower','upper']))
   rs.append(row)
  table(tag+'-exceed',f'Daily exceedance diagnostics: partition {p}, {reg}. Mean counts across fixed cells out of 238 observed dates; each cell retains integer counts in source CSV. Strict exceedance; conditional 28-day intervals.',['Method',r'Days$>5.5\%$',r'\% days$>5.5\%$',r'Days$>6\%$',r'\% days$>6\%$'],rs,'tab:'+tag+'-exceed',True)
pairs=read(V/'results/risk-tracking/paired-contrasts.csv');rs=[]
for x in pairs:
 if x['metric']=='mean_absolute_target_deviation':rs.append([x['partition'][-2:],'Nominal' if x['regime']=='nominal_0.05' else 'Dev.-selected',x['comparator'],f(x['difference']),f"[{f(x['lower'])}, {f(x['upper'])}]"])
table('final-risk-paired','Dynamic HQR minus each comparator in daily MAE (pp). Negative values favor Dynamic HQR. All intervals use paired 28-calendar-day resampling; these exploratory comparisons are not family-wise adjusted.',['Partition','Regime','Comparator','Difference','95\\% interval'],rs,'tab:final-risk-paired',True)
fb=read(V/'results/partition-attribution/facebook-compact-attribution.csv');rs=[]
for x in fb:rs.append([x['partition'][-2:],escape(x['predicted_service']),f(x['prediction_share']),f(x['raw_within_predicted_rejection']),f(x['hqr_within_predicted_rejection']),f(x['contribution_to_true_class_change'])])
table('final-facebook-attribution','Facebook-background attribution in both partitions (21,407 unique flows). Prediction share and within-group rejection are percentages; contribution to this true class\'s total change is pp. All known Meta groups and the top three prediction-mass groups are retained, with other groups combined. Group rates pool fixed-cell counts; these are repeated evaluations of the same flows.',['Partition','Predicted service','Share','Raw rej.','HQR rej.','Contribution'],rs,'tab:final-facebook',True)
roles=read(V/'results/partition-attribution/all-service-roles.csv');rs=[[escape(x['service']),x['original_role'],x['second_role']] for x in roles if x['meta_family']=='True'];table('final-meta-roles','All Meta-family service roles verified from the two actual class lists. K: known; H: held-out; B: background.',['Service','Original','Second'],rs,'tab:final-meta-roles',True)
cl=read(V/'results/partition-attribution/second-partition-all-class-effects.csv');rs=[]
for x in cl:
 if x['role']=='K':continue
 rs.append([escape(x['service']),x['role'],x['unique_n'],f(x['rolling_rejection']),f(x['hqr_rejection']),f(x['difference']),f"[{f(x['lower'])}, {f(x['upper'])}]"])
table('final-second-classes','All 23 second-partition unknown classes: Dynamic HQR minus Rolling global. Rates in percent; differences and conditional paired 28-day 95\\% intervals in pp. Exploratory, unadjusted across classes; source CSV also retains all 30 known classes and macro contributions.',['Service','Role','N','Raw','HQR','Change','95\\% interval'],rs,'tab:final-second-classes',True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,'xtick.labelsize':7.5,'ytick.labelsize':7.5,'axes.linewidth':.6,'pdf.fonttype':42,'svg.fonttype':'none'})
manifest=[]
def save(fig,name,src):
 files=[]
 for ext in ['pdf','svg','png']:
  p=F/(name+'.'+ext);fig.savefig(p,dpi=300);files.append({'path':str(p.relative_to(R)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
 shutil.copy2(F/(name+'.pdf'),V/'manuscript/figures'/(name+'.pdf'));plt.close(fig);manifest.append({'name':name,'inputs':[str(x.relative_to(R)) for x in src],'outputs':files})
fig,axes=plt.subplots(2,2,figsize=(130/25.4,112/25.4),sharex=True,layout='constrained')
for ai,(ax,(p,g)) in enumerate(zip(axes.flat,[(p,g) for p in ['20261121','20261122'] for g in ['nominal_0.05','development_selected']])):
 for j,m in enumerate(methods):
  x=next(x for x in risk if x['partition']==p and x['regime']==g and x['method']==m and x['metric']=='mean_absolute_target_deviation');color='#D55E00' if m=='Dynamic HQR' else '#0072B2';ax.hlines(j,float(x['lower'])*100,float(x['upper'])*100,color=color,lw=1);ax.plot(float(x['value'])*100,j,'o',color=color,ms=3)
 ax.set_yticks(range(6),methods if ai%2==0 else []);ax.invert_yaxis();ax.set_title(('Original' if p=='20261121' else 'Second')+'; '+('nominal' if g=='nominal_0.05' else 'dev.-selected'),fontsize=8);ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',lw=.4,color='#DDDDDD');ax.set_xlim(0,5);ax.text(-.03,1.18,'abcd'[ai],transform=ax.transAxes,fontweight='bold');
 if ai>=2:ax.set_xlabel('Daily MAE from 5% (pp)')
save(fig,'final-risk-tracking',inputs)
cp=R/'revisions/major-v1/results/class-specific-uncertainty.csv';lp=R/'revisions/major-v1/results/leave-one-class-out.csv';rr=sorted([x for x in read(cp) if x['kind']!='known'],key=lambda x:(x['kind']!='background',int(x['code'])));look={x['excluded_code']:x for x in read(lp)}
fig,axes=plt.subplots(1,3,figsize=(130/25.4,145/25.4),sharey=True,layout='constrained',gridspec_kw={'width_ratios':[1,1.1,1]})
for i,r in enumerate(rr):
 c='#D55E00' if r['kind']=='background' else '#0072B2'
 for k,ax in enumerate(axes):
  x=r if k<2 else look[r['code']];point=float(x['difference'])*100;lo=float(x['lower'])*100;hi=float(x['upper'])*100
  ax.hlines(i,lo,hi,color=c,lw=.8);ax.plot(point,i,'o',color=c,ms=2.5)
  if k==1 and point < -8:ax.plot(-7.8,i,'<',color=c,ms=4)
axes[0].set_yticks(range(len(rr)),[x['service']+' ('+x['unique_n']+')' for x in rr]);axes[0].invert_yaxis();axes[1].set_xlim(-8,12);axes[0].set_xlabel('Class change\nfull range (pp)');axes[1].set_xlabel('Class change\ncentral zoom (pp)');axes[2].set_xlabel('Leave-one-out\nmacro change (pp)')
for ax,l in zip(axes,'abc'):
 ax.axvline(0,color='#777777',lw=.6);ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',color='#E6E6E6',lw=.4);ax.set_axisbelow(True);ax.text(0,1.015,l,transform=ax.transAxes,fontweight='bold',fontsize=10)
save(fig,'final-class-robustness',[cp,lp]);(F/'source-mapping.json').write_text(json.dumps({'status':'EXECUTED_REAL_RESULTS_PENDING_VISUAL_QA','figures':manifest},indent=2)+'\n');print('Built 14 final tables and two vector figures from CSV.')
