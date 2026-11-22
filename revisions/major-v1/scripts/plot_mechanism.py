"""Plot actual C-tail alignment and fixed-threshold monthly decomposition."""
from pathlib import Path
import csv, hashlib, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R=Path(__file__).resolve().parents[3];D=R/'revisions/major-v1/results/mechanism';O=R/'revisions/major-v1/figures'
def read(name):
    with (D/name).open() as f:return list(csv.DictReader(f))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':7.5,'axes.linewidth':.6,'pdf.fonttype':42,'svg.fonttype':'none'})
tails=read('historical-C-tail-alignment.csv');decomp=read('mixture-conditional-decomposition.csv');groups=sorted({int(r['predicted_code']) for r in tails});months=sorted({r['month'] for r in decomp})
fig,axes=plt.subplots(3,1,figsize=(130/25.4,135/25.4),layout='constrained')
raw=[];hqr=[];support=[]
for code in groups:
    rr=[r for r in tails if int(r['predicted_code'])==code]
    raw.append(np.mean([float(r['C_tail_rejection']) for r in rr if r['coordinate']=='raw'])*100)
    hqr.append(np.mean([float(r['C_tail_rejection']) for r in rr if r['coordinate']=='hqr'])*100)
    support.append(sum(int(r['C_n']) for r in rr if r['coordinate']=='raw')/3)
tail_limit=float(5*np.ceil(max(raw+hqr)/5))
axes[0].plot([0,tail_limit],[0,tail_limit],color='#888888',linewidth=.6,linestyle='--')
axes[0].scatter(raw,hqr,s=8+np.sqrt(support),facecolor='#0072B2',edgecolor='white',linewidth=.3)
axes[0].axhline(5,color='#D55E00',linewidth=.7,linestyle=':');axes[0].set_xlim(-.5,tail_limit);axes[0].set_ylim(-.5,tail_limit)
axes[0].set_xlabel('Raw historical C group tail (%)');axes[0].set_ylabel('HQR group tail (%)')

x=np.arange(len(months))
for ax,coordinate in zip(axes[1:],['raw','hqr']):
    rows=[[r for r in decomp if r['month']==month and r['coordinate']==coordinate] for month in months]
    mix=np.array([np.mean([float(r['mixture_component']) for r in rr])*100 for rr in rows]);cond=np.array([np.mean([float(r['conditional_component']) for r in rr])*100 for rr in rows]);change=np.array([np.mean([float(r['change']) for r in rr])*100 for rr in rows]);unsupported=np.array([float(r['unsupported_group_component']) for rr in rows for r in rr]);assert np.allclose(unsupported,0)
    ax.bar(x-.16,mix,width=.30,color='#0072B2',label='Mixture term');ax.bar(x+.16,cond,width=.30,color='#D55E00',label='Conditional-rate term');ax.plot(x,change,'ko-',markersize=2.5,linewidth=.7,label='Total change');ax.axhline(0,color='#888888',linewidth=.6)
    ax.set_ylabel(('Raw' if coordinate=='raw' else 'HQR')+' risk change (pp)');ax.set_xticks(x,['Oct','Nov','Dec','Jan','Feb','Mar','Apr','May']);ax.set_ylim(-2,5)
axes[1].legend(frameon=False,ncols=3,loc='upper center',bbox_to_anchor=(.5,1.18),columnspacing=.9,handlelength=1.4);axes[1].tick_params(labelbottom=False);axes[2].set_xlabel('2024                         2025')
for ax,letter in zip(axes,'abc'):
    ax.spines[['top','right']].set_visible(False);ax.text(-.13,1.02,letter,transform=ax.transAxes,fontweight='bold',fontsize=10)
outputs=[]
for extension in ['pdf','svg','png']:
    p=O/('revision-tail-mechanism.'+extension);fig.savefig(p,dpi=300);outputs.append(dict(path=str(p.relative_to(R)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
plt.close(fig)
record=dict(status='RENDERED_PENDING_VISUAL_QA',classification='RESULT',profile='JOURNAL_RESULT',reference_mode='PROJECT_REUSE',inputs=[dict(path=str((D/name).relative_to(R)),sha256=hashlib.sha256((D/name).read_bytes()).hexdigest()) for name in ['historical-C-tail-alignment.csv','mixture-conditional-decomposition.csv']],outputs=outputs,transformations='Panel a averages each predicted-group C rejection over three fitted models, using identical raw/HQR C identities. Panels b/c average monthly exact decomposition terms over those models, at separate fixed historical thresholds. No uncertainty or causal effect is implied. Unsupported-group term is identically zero for these data.',caption='Historical tail alignment and fixed-threshold drift diagnostics. (a) Thirty predicted-known groups on the same disjoint C flows; each point averages three models. Marker size denotes historical support; the dotted line is 5%. (b,c) Monthly flow-weighted known-rejection change from the historical C rate, decomposed into mixture and conditional-rate terms for raw and HQR coordinates. Total change is their sum. These fixed-threshold diagnostics differ from mean-daily FRR of rolling policies and do not identify causal mechanisms.')
(O/'mechanism-source-mapping.json').write_text(json.dumps(record,indent=2)+'\n')
print('Rendered actual historical-tail and monthly decomposition figure.')
