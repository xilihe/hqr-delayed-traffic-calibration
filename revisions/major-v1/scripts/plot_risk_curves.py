"""Render real post-confirmation risk-grid estimates with the existing PAA style."""
from pathlib import Path
import csv,hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[3];source=ROOT/'revisions/major-v1/results/risk-curves-v2/risk-grid-means.csv';out=ROOT/'revisions/major-v1/figures'
with source.open() as f:rows=list(csv.DictReader(f))
styles={'Fixed global':('#999999',':','+'),'Fixed classwise':('#555555','-.','d'),'Static HQR':('#6A5ACD','--','s'),'Rolling global':('#0072B2','--','o'),'Dynamic HQR':('#D55E00','-','o'),'Multi-anchor':('#009E73','-.','v')}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,'legend.fontsize':8,'xtick.labelsize':8,'ytick.labelsize':8,'axes.linewidth':.6,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
fig,axes=plt.subplots(3,1,figsize=(130/25.4,135/25.4),sharex=True,layout='constrained')
for ax,metric,label,panel in zip(axes,['background_macro_detection','heldout_macro_detection','all_unknown_macro_detection'],['Background macro (%)','Held-out macro (%)','All-unknown macro (%)'],'abc'):
 for method,(color,line,marker) in styles.items():
  selected=sorted([r for r in rows if r['method']==method],key=lambda r:float(r['nominal_alpha']));x=np.array([float(r['known_frr'])*100 for r in selected]);y=np.array([float(r[metric])*100 for r in selected]);alpha=np.array([float(r['nominal_alpha']) for r in selected])
  ax.plot(x,y,color=color,linestyle=line,marker=marker,markersize=2.5,linewidth=1,label=method);i=int(np.flatnonzero(alpha==.05)[0]);ax.plot(x[i],y[i],marker=marker,color=color,markersize=5)
 for risk in [3,5,7]:ax.axvline(risk,color='#BBBBBB',linestyle=':',linewidth=.6,zorder=0)
 ax.set_ylabel(label);ax.set_ylim(bottom=0);ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',color='#E6E6E6',linewidth=.4);ax.set_axisbelow(True);ax.text(-.12,1.02,panel,transform=ax.transAxes,fontweight='bold',fontsize=10)
axes[-1].set_xlabel('Actual mean daily known false rejection (%)');axes[-1].set_xlim(left=0)
fig.legend(*axes[0].get_legend_handles_labels(),loc='outside upper center',ncols=2,frameon=False)
files=[]
for extension in ['pdf','svg','png']:
 p=out/('revision-risk-grid.'+extension);fig.savefig(p,dpi=300);files.append(dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
report={'status':'REAL_RISK_GRID_RENDERED_PENDING_VISUAL_QA','input':str(source.relative_to(ROOT)),'input_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'reference':'revisions/major-v1/figures/figure-plan.md','outputs':files,'caption':'Post-confirmation energy operating-point grid at32 daily requests and7-day delay for dynamic methods; static methods request0 labels. Each point averages3 source-model cells for static rules or9 fixed model/audit cells for dynamic rules on the same traffic. All nine declared nominal risk levels are retained; larger markers denote nominal5%. Lines connect separately calibrated policies and are not a single frozen-score ROC. Vertical guides mark actual3%,5%,7% FRR. No interval or superiority claim is implied by line interpolation. Actual-FRR matching is separately labelled descriptive oracle analysis.'}
(out/'risk-grid-source-mapping.json').write_text(json.dumps(report,indent=2)+'\n');print(report['status'])
