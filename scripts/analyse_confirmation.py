"""Paired calendar-block uncertainty for the frozen primary comparison.

Model and audit replications share the same traffic. Blocks are shared across
all methods/replications; nine cells are averaged, never treated as nine
independent traffic collections. No p-value or grid-wide winner is selected.
"""
import argparse
import csv
from datetime import datetime,timedelta
import gzip
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from frozen_protocol import verified_protocol,sha256


def calendar(start,end):
    first=datetime.strptime(start,'%Y%m%d');last=datetime.strptime(end,'%Y%m%d')
    return [(first+timedelta(days=i)).strftime('%Y%m%d') for i in range((last-first).days+1)]


def block_weights(length,block,replicates,seed):
    if not 1<=block<=length:raise ValueError('Invalid calendar block length')
    rng=np.random.default_rng(seed);out=np.zeros((replicates,length),dtype=np.float64)
    for i in range(replicates):
        starts=rng.integers(0,length-block+1,size=int(np.ceil(length/block)))
        indices=np.concatenate([np.arange(s,s+block) for s in starts])[:length]
        out[i]=np.bincount(indices,minlength=length)
    if not np.all(out.sum(axis=1)==length):raise ValueError('Bootstrap calendar length differs')
    return out


def weighted_metrics(counts,weights,kind_masks):
    # counts: [model/audit cell, calendar day, true class, n/rejected/correct].
    cells,days,classes,_=counts.shape
    n=counts[:,:,:,0];rejected=counts[:,:,:,1];correct=counts[:,:,:,2]
    known=kind_masks['known'];daily_n=n[:,:,known].sum(axis=2)
    valid=daily_n>0
    daily_frr=np.divide(rejected[:,:,known].sum(axis=2),daily_n,out=np.zeros_like(daily_n),where=valid)
    daily_correct=np.divide(correct[:,:,known].sum(axis=2),daily_n,out=np.zeros_like(daily_n),where=valid)
    denominator=weights@valid.T.astype(float)
    result={}
    for name,values in [('known_frr',daily_frr),('known_correct_accepted',daily_correct)]:
        total=weights@values.T
        result[name]=np.divide(total,denominator,out=np.full_like(total,np.nan),where=denominator>0).mean(axis=1)
    weighted_n=(weights@n.transpose(1,0,2).reshape(days,-1)).reshape(len(weights),cells,classes)
    weighted_rejected=(weights@rejected.transpose(1,0,2).reshape(days,-1)).reshape(len(weights),cells,classes)
    proportions=np.divide(weighted_rejected,weighted_n,out=np.full_like(weighted_n,np.nan),where=weighted_n>0)
    for kind in ['heldout','background']:
        subset=proportions[:,:,kind_masks[kind]]
        result[kind+'_macro_detection']=np.nanmean(subset,axis=2).mean(axis=1)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);args=parser.parse_args();root=args.root
    protocol,seal=verified_protocol(root);primary=protocol['primary'];dates=calendar(protocol['confirmation_start'],protocol['confirmation_end']);date_index={d:i for i,d in enumerate(dates)}
    inventory=json.loads((root/'outputs/inventory.json').read_text());classes=sorted(int(v) for v in inventory['apps'].values());class_index={c:i for i,c in enumerate(classes)}
    known=set(protocol['known_codes']);background={int(v) for k,v in inventory['apps'].items() if k.casefold().endswith('background')}
    masks={kind:np.asarray([c in known if kind=='known' else c in background if kind=='background' else c not in known|background for c in classes]) for kind in ['known','heldout','background']}
    methods=[('raw',m) for m in protocol['variant_methods']['raw']]+[('raw','delayed_aci_global'),('shift','rolling_global'),('reference','rolling_global')]
    cells=[(e['seed'],s) for e in protocol['models'] for s in protocol['audit_seeds']];cell_index={c:i for i,c in enumerate(cells)}
    arrays={(variant+':'+method,view):np.zeros((len(cells),len(dates),len(classes),3),dtype=np.float64) for variant,method in methods for view in ['all','input_disjoint']}
    out=root/'outputs/confirmation-analysis-v1';out.mkdir(exist_ok=False);inputs=[];grid=[]
    with gzip.open(out/'primary-daily-counts.jsonl.gz','wt') as source_output:
        for entry in protocol['models']:
            folder=root/entry['confirmation_run'];complete=json.loads((folder/'evaluation-complete.json').read_text())
            if complete['protocol_sha256']!=seal['protocol_sha256'] or sha256(folder/'summary.csv')!=complete['summary_sha256']:raise ValueError('Unverified confirmation result')
            trace_hashes={f['path']:f['sha256'] for f in complete['files']}
            with (folder/'summary.csv').open() as stream:grid.extend(list(csv.DictReader(stream)))
            for audit_seed in protocol['audit_seeds']:
                cell=cell_index[(entry['seed'],audit_seed)]
                for variant in ['raw','shift','reference']:
                    score=primary['score'] if variant=='raw' else 'rqp_'+primary['score']
                    stem=f'{score}-budget{primary["budget"]}-delay{primary["delay"]}-seed{audit_seed}'
                    paths=[folder/'replay'/variant/(stem+'.jsonl.gz')]
                    if variant=='raw':paths.append(folder/'replay'/variant/(stem+'-aci.jsonl.gz'))
                    for path in paths:
                        relative=str(path.relative_to(root));actual=sha256(path)
                        if trace_hashes.get(relative)!=actual:raise ValueError('Primary trace checksum differs')
                        inputs.append(dict(path=relative,sha256=actual))
                        with gzip.open(path,'rt') as stream:
                            for line in stream:
                                record=json.loads(line);d=record['date']
                                if d not in date_index:continue
                                key=variant+':'+record['method']
                                for view in ['all','input_disjoint']:
                                    measured=record if view=='all' else record['source_disjoint_metrics'];target=arrays[(key,view)]
                                    for count in measured['class_counts']:
                                        target[cell,date_index[d],class_index[count['label']]]=[count['n'],count['rejected'],count['correct_accepted']]
                                    source_output.write(json.dumps(dict(model_seed=entry['seed'],audit_seed=audit_seed,method=key,view=view,date=d,class_counts=measured['class_counts'],known_frr=measured['known_frr'],known_correct_accepted=measured['known_correct_accepted']))+'\n')
    with (out/'operating-grid.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(grid[0]));writer.writeheader();writer.writerows(grid)
    # Every method must have exactly the same sampled true-class denominators.
    reference=arrays[('raw:rolling_global','all')][:,:,:,0]
    for (key,view),values in arrays.items():
        target=reference if view=='all' else arrays[('raw:rolling_global','input_disjoint')][:,:,:,0]
        if not np.array_equal(values[:,:,:,0],target):raise ValueError('Primary comparisons do not share the same flow population')
    if not np.all(reference==reference[0:1]):raise ValueError('Model/audit replications do not share the fixed traffic cohort')
    class_detection=[]
    for (key,view),counts in arrays.items():
        total=counts[:,:,:,0].sum(axis=1);rejected=counts[:,:,:,1].sum(axis=1)
        rates=np.divide(rejected,total,out=np.full_like(total,np.nan),where=total>0)
        for i,code in enumerate(classes):
            if total[0,i]:class_detection.append(dict(method=key,view=view,label=code,kind='known' if code in known else 'background' if code in background else 'heldout',unique_flow_n=int(total[0,i]),mean_rejection=float(np.nanmean(rates[:,i]))))
    point_weights=np.ones((1,len(dates)));summaries={};intervals={};contrasts=[]
    resampling=protocol['resampling']
    for block in resampling['block_lengths']:
        weights=block_weights(len(dates),block,resampling['replicates'],resampling['seed'])
        boot={}
        for (key,view),counts in arrays.items():
            label=key+'|'+view;point=weighted_metrics(counts,point_weights,masks);draws=weighted_metrics(counts,weights,masks);boot[label]=draws
            if block==resampling['primary_block']:
                summaries[label]={metric:float(value[0]) for metric,value in point.items()}
                intervals[label]={metric:np.quantile(value,[.025,.975]).tolist() for metric,value in draws.items()}
        for view in ['all','input_disjoint']:
            candidate='shift:rolling_global|'+view
            for comparator in ['raw:rolling_global','reference:rolling_global']:
                reference_label=comparator+'|'+view
                for metric in boot[candidate]:
                    difference=boot[candidate][metric]-boot[reference_label][metric]
                    point=float(weighted_metrics(arrays[('shift:rolling_global',view)],point_weights,masks)[metric][0]-weighted_metrics(arrays[(comparator,view)],point_weights,masks)[metric][0])
                    contrasts.append(dict(candidate='shift:rolling_global',comparator=comparator,view=view,metric=metric,block_days=block,difference=point,interval_95=np.quantile(difference,[.025,.975]).tolist()))
    # Individual model means expose seed sensitivity without pretending the
    # model seeds are independent networks or additional traffic populations.
    model_sensitivity=[]
    for (key,view),counts in arrays.items():
        for entry in protocol['models']:
            selected=[i for i,c in enumerate(cells) if c[0]==entry['seed']]
            result=weighted_metrics(counts[selected],point_weights,masks)
            model_sensitivity.append(dict(method=key,view=view,model_seed=entry['seed'],**{k:float(v[0]) for k,v in result.items()}))
    verified_protocol(root)
    report=dict(status='FROZEN_CONFIRMATION_PAIRED_BLOCK_ANALYSIS_COMPLETE',protocol_sha256=seal['protocol_sha256'],primary=primary,calendar_days=len(dates),observed_dates=len(protocol['confirmation_tables']),replication_cells=cells,estimand='Average across the fixed three model seeds and three audit streams; dates resampled jointly for every paired comparison.',summaries=summaries,intervals_95_primary_block=intervals,contrasts=contrasts,class_detection=class_detection,model_sensitivity=model_sensitivity,resampling=resampling,inputs=inputs,output_hashes={name:sha256(out/name) for name in ['primary-daily-counts.jsonl.gz','operating-grid.csv']})
    (out/'analysis.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(status=report['status'],calendar_days=len(dates),observed_dates=len(protocol['confirmation_tables']),paired_contrasts=len(contrasts))),flush=True)


if __name__=='__main__':main()
