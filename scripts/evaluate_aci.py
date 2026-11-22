"""Compare delayed ACI on existing development scores; never retrain or read flows."""
import argparse
import csv
from datetime import timedelta
import gzip
import hashlib
import json
from pathlib import Path
import random
import sys
import numpy as np
from configio import study_config, local_path
from evaluate import records
from audit import recount, date
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from online_aci import run_delayed_aci
from fast_metrics import PackedDay


def audit_trace(results, daily_arrays, initial_scores, score, config, budget, delay, seed, gamma):
    """Verify saved levels using cumulative errors and independent NumPy quantiles."""
    queries_by_date={}
    total_queries=0
    for record in results:
        current_date=record['date'];today=date(current_date)
        latest=today-timedelta(days=delay);oldest=today-timedelta(days=config['window_days'])
        arrived_dates=[d for d in sorted(queries_by_date) if date(d)<=latest]
        known_count=sum(queries_by_date[d]['known_count'] for d in arrived_dates)
        rejected_count=sum(queries_by_date[d]['known_rejected_count'] for d in arrived_dates)
        expected_level=config['alpha']+gamma*(config['alpha']*known_count-rejected_count)
        if not np.isclose(record['adaptive_alpha'],expected_level,rtol=1e-10,atol=1e-10):
            raise ValueError('ACI level differs from cumulative historical-error identity')
        recent_dates=[d for d in arrived_dates if date(d)>=oldest]
        parts=[daily_arrays[d][score][queries_by_date[d]['known_indices']] for d in recent_dates]
        values=np.concatenate(parts).astype(np.float64) if parts and sum(map(len,parts)) else initial_scores
        if expected_level<=0:expected=np.inf
        elif expected_level>=1:expected=-np.inf
        else:
            rank=int(np.ceil((len(values)+1)*(1-expected_level)))
            expected=np.inf if rank>len(values) else np.partition(values,rank-1)[rank-1]
        observed={'+inf':np.inf,'-inf':-np.inf}.get(record['cutoff'],record['cutoff'])
        if not np.isclose(observed,expected,rtol=1e-10,atol=1e-10):
            raise ValueError('ACI cutoff differs from independent order statistic')
        data=daily_arrays[current_date]
        cutoffs={str(c):float(observed) for c in config['known_codes']}
        if recount(data,score,cutoffs)!=record['class_counts']:
            raise ValueError('ACI decision counts differ')
        if recount(data,score,cutoffs,True)!=record['source_disjoint_metrics']['class_counts']:
            raise ValueError('ACI source-disjoint decision counts differ')
        if record['deployment_queries_before_today']!=total_queries:
            raise ValueError('ACI label cost differs from common audit stream')
        token=int.from_bytes(hashlib.sha256(f'{seed}:{current_date}'.encode()).digest()[:8],'big')
        indices=np.asarray(sorted(random.Random(token).sample(range(len(data['labels'])),min(budget,len(data['labels'])))),dtype=np.int64)
        known_indices=indices[data['known'][indices]]
        queries_by_date[current_date]=dict(known_indices=known_indices,known_count=len(known_indices),
                                          known_rejected_count=int(np.count_nonzero(data[score][known_indices]>observed)))
        total_queries+=len(indices)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--config',default='configs/pilot-v1.json');args=parser.parse_args();root=args.root
    base_path,config,folder=study_config(root,args.config)
    comparator_path=local_path(root,config.get('aci_comparator_config','configs/aci-development.json'));comparator=json.loads(comparator_path.read_text())
    if comparator['source_run']!=folder.name:raise ValueError('ACI comparator points to another source run')
    prepared=json.loads((folder/'prepare-complete.json').read_text())
    if prepared['protected_flow_tables_read'] or prepared['config_sha256']!=hashlib.sha256(base_path.read_bytes()).hexdigest():
        raise ValueError('Prepared run identity or protected-data boundary differs')
    config['known_codes']=prepared['known_codes']
    out=folder/'aci-comparison';out.mkdir(exist_ok=False)
    daily_arrays={}
    for p in sorted((folder/'scores').glob('dev-*.npz')):
        d=p.stem.split('-',1)[1]
        if d>=config['protected_start']:raise ValueError('Protected score cache')
        with np.load(p,allow_pickle=False) as data:daily_arrays[d]={k:data[k].copy() for k in data.files}
    summary=[];trace_files=[]
    for score in config['rejection_scores']:
        initial=[r for p in sorted((folder/'scores').glob('cal-*.npz')) for r in records(p,score)]
        daily={p.stem.split('-',1)[1]:records(p,score) for p in sorted((folder/'scores').glob('dev-*.npz'))}
        initial_scores=np.asarray([r['score'] for r in initial],dtype=np.float64)
        packed={date:PackedDay(rows,config['known_codes'],date) for date,rows in daily.items()}
        for budget in config['label_budgets_per_day']:
            for delay in config['label_delays_days']:
                for seed in config['label_sampling_seeds']:
                    for gamma in comparator['gamma_grid']:
                        result=run_delayed_aci(initial,daily,config,budget,delay,seed,gamma,metric_engines=packed)
                        name=f'{score}-budget{budget}-delay{delay}-seed{seed}-gamma{gamma}'
                        path=out/(name+'.jsonl.gz')
                        with gzip.open(path,'wt') as stream:
                            for row in result:stream.write(json.dumps(row,allow_nan=False)+'\n')
                        with gzip.open(path,'rt') as stream:retained=[json.loads(line) for line in stream]
                        audit_trace(retained,daily_arrays,initial_scores,score,config,budget,delay,seed,gamma)
                        row=dict(score=score,method='delayed_aci_global',gamma=gamma,budget=budget,delay=delay,sampling_seed=seed,evaluated_days=len(result),days_exceeding_alpha=sum(r['known_frr_exceeds_alpha'] for r in result))
                        for field in ['known_frr','known_frr_excess','known_macro_frr','known_correct_accepted','heldout_macro_detection','background_macro_detection']:
                            values=[r[field] for r in result if r[field] is not None];row['mean_daily_'+field]=sum(values)/len(values) if values else None
                        summary.append(row);trace_files.append({'file':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
                        print(json.dumps({'completed_and_recounted':name}),flush=True)
    with (out/'summary.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    report=dict(status='DEVELOPMENT_ACI_COMPLETE_WITH_INDEPENDENT_NUMPY_AUDIT',summary_cells=len(summary),comparator_config_sha256=hashlib.sha256(comparator_path.read_bytes()).hexdigest(),source_config_sha256=prepared['config_sha256'],files=trace_files,formal_confirmation_performed=False,checks=['Cumulative delayed-error update identity','Independent finite-sample order selection','Common query stream and cost','All per-class decision counts','Source-disjoint counts'])
    (out/'complete.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
