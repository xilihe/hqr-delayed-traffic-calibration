"""Plot all requested unknown classes and monthly paired effects from real counts."""
from pathlib import Path
import csv,json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[3];D=ROOT/'revisions/major-v1/results';OUT=ROOT/'revisions/major-v1/figures'
def read(path):
 with path.open() as f:return list(csv.DictReader(f))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,'legend.fontsize':8,'xtick.labelsize':8,'ytick.labelsize':7.5,'axes.linewidth':.6,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
outputs=[]
def save(fig,name,inputs,caption):
 for extension in ['pdf','svg','png']:
  path=OUT/(name+'.'+extension);fig.savefig(path,dpi=300);outputs.append(dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
 plt.close(fig)
 return dict(name=name,caption=caption,inputs=[dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in inputs])
def clean(ax):ax.spines[['top','right']].set_visible(False);ax.axvline(0,color='#777777',linewidth=.6);ax.grid(axis='x',color='#E6E6E6',linewidth=.4);ax.set_axisbelow(True)
classpath=D/'class-specific-uncertainty.csv';loopath=D/'leave-one-class-out.csv';rows=[r for r in read(classpath) if r['kind']!='known'];rows=sorted(rows,key=lambda r:(r['kind']!='background',int(r['code'])));lookup={int(r['excluded_code']):r for r in read(loopath)}
fig,axes=plt.subplots(1,2,figsize=(130/25.4,135/25.4),sharey=True,layout='constrained');y=np.arange(len(rows));labels=[r['service']+' ('+r['unique_n']+')' for r in rows]
for i,row in enumerate(rows):
 color='#D55E00' if row['kind']=='background' else '#0072B2'
 for ax,v in zip(axes,[row,lookup[int(row['code'])]]):
  point=float(v['difference'])*100;lo=float(v['lower'])*100;hi=float(v['upper'])*100;ax.hlines(i,lo,hi,color=color,linewidth=.8);ax.plot(point,i,'o',color=color,markersize=2.4)
axes[0].set_yticks(y,labels);axes[0].invert_yaxis();axes[0].set_xlabel('Class-specific change (pp)');axes[1].set_xlabel('Within-population\nleave-one-out change (pp)')
for ax,letter in zip(axes,'ab'):clean(ax);ax.text(0,1.015,letter,transform=ax.transAxes,fontweight='bold',fontsize=10)
class_caption='All23 unknown classes, with unique confirmation-flow counts in parentheses. (a) Dynamic HQR minus rolling-global class-specific rejection. (b) Corresponding difference in the macro after removing that class from its own population: two remaining background groups or19 remaining held-out services. Orange/blue distinguish background/held-out classes. Intervals are paired28-calendar-day bootstrap95% intervals, conditional on the same traffic and fixed model/audit cells. Negative and positive effects are all retained. Class rarity and valid bootstrap draws are reported in the full source table.'
reports=[save(fig,'revision-class-robustness',[classpath,loopath],class_caption)]
timepath=D/'temporal-periods.csv';data=read(timepath);months=['2024-10','2024-11','2024-12','2025-01','2025-02','2025-03','2025-04','2025-05'];x=np.arange(8)
fig,axes=plt.subplots(3,1,figsize=(130/25.4,120/25.4),sharex=True,layout='constrained')
for ax,metric,letter in zip(axes[:2],['background_macro_detection','heldout_macro_detection'],'ab'):
 selected=[next(r for r in data if r['period']==m and r['metric']==metric and r['block_days']=='7') for m in months];point=np.array([float(r['difference']) for r in selected])*100;lo=np.array([float(r['lower']) for r in selected])*100;hi=np.array([float(r['upper']) for r in selected])*100
 ax.vlines(x,lo,hi,color='#D55E00',linewidth=.9);ax.plot(x,point,'o-',color='#D55E00',markersize=3,linewidth=1);ax.axhline(0,color='#777777',linewidth=.6);ax.set_ylabel(('Background' if metric.startswith('background') else 'Held-out')+' Δ (pp)');ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',color='#E6E6E6',linewidth=.4);ax.text(-.12,1.02,letter,transform=ax.transAxes,fontweight='bold',fontsize=10)
selected=[next(r for r in data if r['period']==m and r['metric']=='background_macro_detection' and r['block_days']=='7') for m in months];n=np.array([[int(r[k]) for r in selected] for k in ['known_n','heldout_n','background_n']],float);n/=n.sum(0);bottom=np.zeros(8)
for values,color,label in zip(n,['#80B1D3','#B3DE69','#FDB462'],['Known','Held-out','Background']):axes[2].bar(x,values,bottom=bottom,color=color,label=label,width=.8);bottom+=values
axes[2].set_ylabel('Flow proportion');axes[2].set_ylim(0,1);axes[2].spines[['top','right']].set_visible(False);axes[2].set_xticks(x,['Oct','Nov','Dec','Jan','Feb','Mar','Apr','May']);axes[2].set_xlabel('2024                          2025');axes[2].legend(loc='upper center',bbox_to_anchor=(.5,-.48),ncols=3,frameon=False);axes[2].text(-.12,1.02,'c',transform=axes[2].transAxes,fontweight='bold',fontsize=10)
reports.append(save(fig,'revision-monthly-effects',[timepath],'Post-confirmation monthly composition and paired Dynamic HQR minus rolling-global effects. Unknown detection is a period macro within each month. Intervals use2000 paired7-calendar-day block resamples within the month, not independent model seeds. Missing dates remain absent observations. Differences between month estimates are descriptive and are not a tested time-interaction effect.'))
(OUT/'class-time-source-mapping.json').write_text(json.dumps(dict(status='REAL_RESULT_FIGURES_RENDERED_PENDING_VISUAL_QA',figures=reports,outputs=outputs),indent=2)+'\n');print('Rendered all-class robustness and monthly effects from actual supplementary tables.')
