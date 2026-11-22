"""Render manuscript figures from verified confirmation outputs only.

Display smoothing uses trailing 28 calendar days with >=7 observed dates.
Point estimates and uncertainty come from the frozen analysis, not the display
smoothing. Error bars across model fits are ranges, not confidence intervals.
"""
import argparse
from collections import defaultdict
import csv
from datetime import datetime,timedelta
import gzip
import hashlib
import json
from pathlib import Path
import platform
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D

COLORS={'raw:fixed_global':'#999999','raw:fixed_classwise':'#777777','raw:rolling_global':'#0072B2','raw:rolling_classwise':'#CC79A7','raw:pooled_classwise':'#6A5ACD','raw:delayed_aci_global':'#56B4E9','shift:rolling_global':'#D55E00','reference:rolling_global':'#009E73'}
LABELS={'raw:fixed_global':'Fixed global','raw:fixed_classwise':'Fixed classwise','raw:rolling_global':'Rolling global','raw:rolling_classwise':'Rolling classwise','raw:pooled_classwise':'Threshold shrinkage','raw:delayed_aci_global':'Delayed ACI','shift:rolling_global':'HQR','reference:rolling_global':'Multi-anchor'}
STYLES={'raw:fixed_classwise':('--','s'),'raw:rolling_global':('--','s'),'raw:rolling_classwise':('-.','^'),'raw:delayed_aci_global':(':','D'),'shift:rolling_global':('-','o'),'reference:rolling_global':('-.','v')}
SCORES={'negative_energy':'Energy','one_minus_msp':'1 − MSP','mahalanobis':'Mahalanobis'}


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);args=parser.parse_args();root=args.root
    base=root/'outputs/confirmation-analysis-v1';analysis=json.loads((base/'analysis.json').read_text());protocol_path=root/'research/frozen/confirmation-v1.json';protocol=json.loads(protocol_path.read_text())
    if analysis['protocol_sha256']!=digest(protocol_path):raise ValueError('Analysis and protocol differ')
    inputs=[protocol_path,base/'analysis.json',root/'outputs/inventory.json']
    for name,expected in analysis['output_hashes'].items():
        if digest(base/name)!=expected:raise ValueError('Verified figure source changed: '+name)
        inputs.append(base/name)
    with (base/'operating-grid.csv').open() as stream:grid=list(csv.DictReader(stream))
    out=root/'manuscript/figures';out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,'axes.titlesize':9,'legend.fontsize':8,'xtick.labelsize':8,'ytick.labelsize':8,'axes.linewidth':.6,'lines.linewidth':1.2,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
    width=130/25.4;outputs=[]
    def clean(ax):
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',color='#E7E7E7',linewidth=.45);ax.set_axisbelow(True);ax.tick_params(width=.6,length=2.5)
    def panel(ax,label):ax.text(-.12,1.04,label,transform=ax.transAxes,fontweight='bold',fontsize=10,va='bottom')
    def save(fig,name,sources,description):
        for extension in ['pdf','svg','png']:
            path=out/(name+'.'+extension);fig.savefig(path,dpi=300);outputs.append(dict(path=str(path.relative_to(root)),sha256=digest(path),sources=sources,description=description))
        plt.close(fig)
    start=datetime.strptime(protocol['confirmation_start'],'%Y%m%d');end=datetime.strptime(protocol['confirmation_end'],'%Y%m%d');calendar=[start+timedelta(days=i) for i in range((end-start).days+1)];index={d.strftime('%Y%m%d'):i for i,d in enumerate(calendar)};length=len(calendar)
    def temporal(ax):
        clean(ax);ax.set_xlim(start,end);ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2));ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    def trailing(values,denominator=None):
        result=np.full(length,np.nan)
        for t in range(length):
            if not np.isfinite(values[t]):continue
            a=values[max(0,t-27):t+1];valid=np.isfinite(a)
            if valid.sum()<7:continue
            if denominator is None:result[t]=np.nanmean(a)
            else:
                b=denominator[max(0,t-27):t+1];result[t]=np.nansum(a)/np.nansum(b)
        return result
    ledgers=[]
    for entry in protocol['models']:
        path=root/entry['confirmation_run']/'score-complete.json';report=json.loads(path.read_text());inputs.append(path)
        if report['protocol_sha256']!=analysis['protocol_sha256']:raise ValueError('Classifier-context source differs')
        ledgers.append(report['confirmation_tables_read'])
    population=np.full((3,length),np.nan);accuracy=[]
    for model,ledger in enumerate(ledgers):
        n=np.full(length,np.nan);correct=n.copy()
        for row in ledger:
            i=index[row['table'].rsplit('/D',1)[1]];n[i]=row['known_n'];correct[i]=row['known_correct']
            if model==0:population[:,i]=np.asarray([row['known_n'],row['heldout_n'],row['background_n']])/row['rows']
        accuracy.append(trailing(correct,n)*100)
    accuracy=np.asarray(accuracy)
    fig,axes=plt.subplots(2,1,figsize=(width,95/25.4),sharex=True,layout='constrained')
    axes[0].stackplot(calendar,*population,labels=['Known services','Held-out services','Background groups'],colors=['#80B1D3','#B3DE69','#FDB462'],linewidth=0)
    axes[0].set_ylim(0,1);axes[0].set_ylabel('Sample proportion');fig.legend(*axes[0].get_legend_handles_labels(),loc='outside upper center',ncols=3,frameon=False,fontsize=8)
    with np.errstate(invalid='ignore'):
        mean=np.nanmean(accuracy,axis=0);lower=np.nanmin(accuracy,axis=0);upper=np.nanmax(accuracy,axis=0)
    axes[1].plot(calendar,mean,color='#222222');axes[1].fill_between(calendar,lower,upper,color='#AAAAAA',alpha=.3,linewidth=0)
    axes[1].set_ylabel('Known-flow accuracy (%)');axes[1].set_ylim(0,100)
    for label,ax in zip('ab',axes):temporal(ax);panel(ax,label)
    save(fig,'figure1-cohort',['confirmation score ledgers','frozen date inventory'],'Observed-date composition; trailing28-calendar-day classification accuracy before rejection, mean and range across three source-model seeds.')
    # Primary traced decisions, retaining true-class counts for period macros.
    methods=['raw:rolling_global','raw:delayed_aci_global','shift:rolling_global','reference:rolling_global']
    cells=[(entry['seed'],seed) for entry in protocol['models'] for seed in protocol['audit_seeds']];cell_index={x:i for i,x in enumerate(cells)}
    inventory=json.loads((root/'outputs/inventory.json').read_text());codes=sorted(int(x) for x in inventory['apps'].values());code_index={c:i for i,c in enumerate(codes)}
    known=set(protocol['known_codes']);background={int(v) for k,v in inventory['apps'].items() if k.casefold().endswith('background')}
    kinds={'background':np.asarray([c in background for c in codes]),'heldout':np.asarray([c not in known|background for c in codes])}
    daily={method:np.full((9,length),np.nan) for method in methods};counts={method:np.zeros((9,length,len(codes),2)) for method in methods}
    with gzip.open(base/'primary-daily-counts.jsonl.gz','rt') as stream:
        for line in stream:
            row=json.loads(line);method=row['method']
            if method not in methods or row['view']!='all':continue
            c=cell_index[(row['model_seed'],row['audit_seed'])];d=index[row['date']];daily[method][c,d]=row['known_frr']
            for cell in row['class_counts']:counts[method][c,d,code_index[cell['label']]]=[cell['n'],cell['rejected']]
    def rolling_macro(data,mask):
        result=np.full((9,length),np.nan)
        for t in range(length):
            if data[0,t,:,0].sum()==0:continue
            window=data[:,max(0,t-27):t+1,mask,:]
            observed=np.count_nonzero(window[0,:,:,0].sum(axis=1))
            if observed<7:continue
            total=window.sum(axis=1);rates=np.divide(total[:,:,1],total[:,:,0],out=np.full(total.shape[:2],np.nan),where=total[:,:,0]>0)
            result[:,t]=np.nanmean(rates,axis=1)
        return result
    fig,axes=plt.subplots(3,1,figsize=(width,135/25.4),sharex=True,layout='constrained')
    for method in methods:
        values=[np.asarray([trailing(x) for x in daily[method]]),rolling_macro(counts[method],kinds['background']),rolling_macro(counts[method],kinds['heldout'])]
        for ax,data in zip(axes,values):
            with np.errstate(invalid='ignore'):
                model=data.reshape(3,3,length).mean(axis=1)*100;mean=np.nanmean(model,axis=0);low=np.nanmin(model,axis=0);high=np.nanmax(model,axis=0)
            ax.plot(calendar,mean,color=COLORS[method],label=LABELS[method],linestyle=STYLES[method][0],linewidth=1.4 if method.startswith('shift') else 1)
            ax.fill_between(calendar,low,high,color=COLORS[method],alpha=.10,linewidth=0)
    axes[0].axhline(5,color='#555555',linestyle='--',linewidth=.7)
    for ax,label,y in zip(axes,'abc',['Known false rejection (%)','Background detection (%)','Held-out detection (%)']):
        temporal(ax);panel(ax,label);ax.set_ylabel(y);ax.set_ylim(bottom=0)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='outside upper center',ncols=2,frameon=False,fontsize=8)
    save(fig,'figure2-primary',['primary-daily-counts.jsonl.gz'],'Trailing28-calendar-day display with >=7 observed dates; class counts pooled before unknown-class macros; mean and range of three model means, each averaged over three audit streams.')
    displayed=['raw:fixed_classwise','raw:rolling_global','raw:rolling_classwise','raw:delayed_aci_global','shift:rolling_global']
    def grouped(metric,score,budget,delay,key,view='all'):
        v,m=key.split(':',1);selected=[r for r in grid if r['variant']==v and r['method']==m and r['score']==score and int(r['budget'])==budget and int(r['delay'])==delay and r['view']==view]
        if len(selected)!=9:raise ValueError('Operating-grid cell lacks all nine fixed replications')
        by_model=defaultdict(list)
        for r in selected:by_model[int(r['model_seed'])].append(float(r[metric])*100)
        means=np.asarray([np.mean(by_model[s]) for s in sorted(by_model)])
        return means
    fig,axes=plt.subplots(2,2,figsize=(width,100/25.4),sharex=True,layout='constrained');x=np.arange(3);budgets=protocol['budgets']
    for col,delay in enumerate(protocol['delays']):
        for row,metric in enumerate(['period_background_macro_rejection','mean_daily_known_frr']):
            ax=axes[row,col]
            for key in displayed:
                model=np.asarray([grouped(metric,'negative_energy',b,delay,key) for b in budgets]);mean=model.mean(axis=1)
                ax.errorbar(x,mean,yerr=np.stack([mean-model.min(axis=1),model.max(axis=1)-mean]),color=COLORS[key],linestyle=STYLES[key][0],marker=STYLES[key][1],markersize=3,capsize=2,label=LABELS[key])
            if row==0:ax.set_title(f'{delay}-day label delay')
            if row==1:ax.axhline(5,color='#555555',linestyle=':',linewidth=.6);ax.set_xlabel('Requested labels per day')
            clean(ax);panel(ax,chr(97+row*2+col));ax.set_xticks(x,[str(b) for b in budgets]);ax.set_ylim(bottom=0)
    axes[0,0].set_ylabel('Background detection (%)');axes[1,0].set_ylabel('Known false rejection (%)')
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncols=3,fontsize=8,frameon=False)
    save(fig,'figure3-budget-delay',['operating-grid.csv'],'Energy operating points for every frozen budget/delay setting. Whiskers show the range of three model means over common audit streams. Static classwise reference has zero deployment queries.')
    # Population-specific mechanism and declared sensitivity views.
    fig,axes=plt.subplots(2,2,figsize=(width,105/25.4),layout='constrained')
    group_codes=sorted(background);xx=np.arange(len(group_codes));lookup={int(v):k for k,v in inventory['apps'].items()}
    for offset,key in [(-.16,'raw:rolling_global'),(0,'shift:rolling_global'),(.16,'reference:rolling_global')]:
        values=[next(r['mean_rejection']*100 for r in analysis['class_detection'] if r['method']==key and r['view']=='all' and r['label']==c) for c in group_codes]
        axes[0,0].plot(xx+offset,values,STYLES[key][1],color=COLORS[key],markersize=4,label=LABELS[key])
    axes[0,0].set_xticks(xx,[lookup[c].removesuffix('-background').capitalize() for c in group_codes],rotation=15);axes[0,0].set_ylabel('Background detection (%)');axes[0,0].set_ylim(bottom=0);axes[0,0].legend(frameon=False,fontsize=8)
    metrics=[('period_background_macro_rejection','Background detection','#D55E00','o'),('mean_daily_known_frr','Known false rejection','#555555','x')]
    for offset,(metric,label,color,marker) in zip([-.10,.10],metrics):
        deltas=np.asarray([grouped(metric,s,32,7,'shift:rolling_global')-grouped(metric,s,32,7,'raw:rolling_global') for s in protocol['scores']]);mean=deltas.mean(axis=1)
        axes[0,1].errorbar(mean,np.arange(3)+offset,xerr=np.stack([mean-deltas.min(axis=1),deltas.max(axis=1)-mean]),fmt=marker,color=color,capsize=2,markersize=4,label=label)
    axes[0,1].set_yticks(np.arange(3),[SCORES[s] for s in protocol['scores']]);axes[0,1].invert_yaxis();axes[0,1].set_xlabel('HQR − rolling global (pp)');axes[0,1].axvline(0,color='#777777',linewidth=.6);axes[0,1].legend(frameon=False,fontsize=8,loc='lower center',bbox_to_anchor=(.5,1.02))
    contrast_metrics=['background_macro_detection','heldout_macro_detection'];short=['Background','Held-out']
    for col,comparator in enumerate(['raw:rolling_global','reference:rolling_global']):
        ax=axes[1,col]
        for offset,view,filled in [(-.10,'all',True),(.10,'input_disjoint',False)]:
            selected=[next(r for r in analysis['contrasts'] if r['view']==view and r['comparator']==comparator and r['metric']==m and r['block_days']==28) for m in contrast_metrics]
            means=np.asarray([r['difference']*100 for r in selected]);bounds=np.asarray([r['interval_95'] for r in selected])*100
            y=np.arange(2)+offset;color=COLORS['shift:rolling_global']
            ax.hlines(y,bounds[:,0],bounds[:,1],color=color,linewidth=1)
            for edge in [bounds[:,0],bounds[:,1]]:ax.vlines(edge,y-.025,y+.025,color=color,linewidth=1)
            ax.plot(means,y,'o',color=color,mfc=color if filled else 'white',markersize=4,label='All inputs' if filled else 'Input disjoint')
        ax.set_yticks(np.arange(2),short);ax.invert_yaxis();ax.axvline(0,color='#777777',linewidth=.6);ax.set_xlabel('HQR − '+('rolling global' if col==0 else 'multi-anchor')+' (pp)');ax.legend(frameon=False,fontsize=8)
    for ax,label in zip(axes.flat,'abcd'):clean(ax);panel(ax,label)
    save(fig,'figure4-mechanism-sensitivity',['analysis.json','operating-grid.csv'],'Primary population-specific rates, per-score paired model-mean ranges, and all/input-disjoint paired28-day bootstrap95% intervals. pp means percentage points.')
    report=dict(status='FIGURES_RENDERED_PENDING_VISUAL_QA',python=platform.python_version(),numpy=np.__version__,matplotlib=matplotlib.__version__,backend=matplotlib.get_backend(),input_files=[dict(path=str(p.relative_to(root)),sha256=digest(p)) for p in inputs],outputs=outputs,style_reference='research/figures/reference-selection.md',profile='JOURNAL_RESULT',result_pixels_from_real_verified_inputs=True,source_smoothing='Trailing28-calendar-day display with at least7 observed dates; no smoothed values enter frozen statistical estimates.',qa_status='PENDING_ACTUAL_RENDER_INSPECTION')
    (out/'source-mapping.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(status=report['status'],figures=len(outputs)//3,files=len(outputs))),flush=True)


if __name__=='__main__':main()
