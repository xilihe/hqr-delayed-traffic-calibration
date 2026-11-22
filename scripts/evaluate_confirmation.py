"""Replay a sealed confirmation once, with development history as warmup.

All saved decisions, including warmup, are independently reconstructed. Only
confirmation-date counts enter the reported confirmation estimands.
"""
import argparse
from collections import defaultdict
from datetime import timedelta
import csv
import gzip
import hashlib
import json
from pathlib import Path
import random
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from frozen_protocol import verified_protocol,sha256
from replay import run_replay
from online_aci import run_delayed_aci
from fast_metrics import PackedDay
from evaluate import records
from audit import date,recount,independent_cutoffs
from evaluate_aci import audit_trace
from summarize_periods import aggregate


def audit_methods(retained,cached,initial_scores,initial_prediction,score,config,budget,delay,seed):
    """Rebuild history/order statistics separately from replay's queue state."""
    requests={};total=0;by_date=defaultdict(list);decisions=0
    for record in retained:by_date[record['date']].append(record)
    if sorted(by_date)!=sorted(cached):raise ValueError('Saved date coverage differs')
    for current_date in sorted(cached):
        today=date(current_date);oldest=today-timedelta(days=config['window_days']);latest=today-timedelta(days=delay)
        available=[(d,i) for d in sorted(requests) if oldest<=date(d)<=latest for i in requests[d]]
        ids=[d+':'+str(int(cached[d]['indices'][i])) for d,i in available]
        history=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()
        known=[(d,i) for d,i in available if cached[d]['known'][i]]
        recent_scores=np.asarray([cached[d][score][i] for d,i in known],dtype=np.float64)
        recent_prediction=np.asarray([cached[d]['codes'][cached[d]['prediction'][i]] for d,i in known])
        daily=by_date[current_date]
        if sorted(r['method'] for r in daily)!=sorted(config['methods']):raise ValueError('Saved method coverage differs')
        for record in daily:
            expected=independent_cutoffs(initial_scores,initial_prediction,recent_scores,recent_prediction,config['known_codes'],record['method'],config['alpha'],config['pooling_pseudocount'])
            observed={k:np.inf if v is None else v for k,v in record['thresholds'].items()}
            keys=sorted(expected)
            if set(keys)!=set(observed) or not np.allclose([expected[k] for k in keys],[observed[k] for k in keys],rtol=1e-12,atol=1e-12):raise ValueError('Numeric threshold audit failed')
            if record['history_sha256']!=history or record['labels_requested_before_today']!=total or record['known_labels_available']!=len(known):raise ValueError('Delayed-label history audit failed')
            if record['labels_available_in_window']!=len(available):raise ValueError('Audit window support differs')
            expected_support={str(c):int(np.count_nonzero(recent_prediction==c)) for c in config['known_codes']}
            if record['recent_known_support_by_predicted_class']!=expected_support:raise ValueError('Predicted-class audit support differs')
            cost=0 if record['method'].startswith('fixed_') else total
            if record['deployment_queries_before_today']!=cost:raise ValueError('Deployment query cost differs')
            for isolated in [False,True]:
                view=record['source_disjoint_metrics'] if isolated else record
                if recount(cached[current_date],score,record['thresholds'],isolated)!=view['class_counts']:raise ValueError('Independent class-count audit failed')
            decisions+=len(cached[current_date]['labels'])
        data=cached[current_date];token=int.from_bytes(hashlib.sha256(f'{seed}:{current_date}'.encode()).digest()[:8],'big')
        requests[current_date]=sorted(random.Random(token).sample(range(len(data['labels'])),min(budget,len(data['labels']))))
        total+=len(requests[current_date])
    return decisions


def retained_trace(path,result):
    with gzip.open(path,'wt') as stream:
        for row in result:stream.write(json.dumps(row,allow_nan=False)+'\n')
    with gzip.open(path,'rt') as stream:return [json.loads(line) for line in stream]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--seed',type=int,required=True)
    args=parser.parse_args();root=args.root;protocol,seal=verified_protocol(root)
    entry=next(e for e in protocol['models'] if e['seed']==args.seed)
    output=root/entry['confirmation_run'];scored=json.loads((output/'score-complete.json').read_text())
    if scored['protocol_sha256']!=seal['protocol_sha256']:raise ValueError('Score cache uses another protocol')
    for file in scored['files']:
        if sha256(root/file['path'])!=file['sha256']:raise ValueError('Confirmation cache changed after scoring')
    replay_dir=output/'replay';replay_dir.mkdir(exist_ok=False)
    summaries=[];provenance=[];decision_count=0
    for variant in ['raw','reference','shift']:
        source=root/entry[variant+'_run'];config=json.loads((root/entry[variant+'_config']).read_text())
        config['known_codes']=protocol['known_codes'];config['methods']=protocol['variant_methods'][variant]
        scores=protocol['scores']
        out=replay_dir/variant;out.mkdir()
        paths=sorted((source/'scores').glob('dev-*.npz'))+sorted((output/variant/'scores').glob('test-*.npz'))
        if len(paths)!=protocol['warmup_dates']+len(protocol['confirmation_tables']):raise ValueError('Warmup or confirmation date count differs')
        cached={}
        for path in paths:
            d=path.stem.split('-',1)[1]
            if d in cached:raise ValueError('Repeated chronological date')
            with np.load(path,allow_pickle=False) as archive:cached[d]={k:archive[k] for k in archive.files}
            data=cached[d]
            if not np.array_equal(data['known'],np.isin(data['labels'],protocol['known_codes'])):raise ValueError('Known/unknown class coding differs')
        for base_score in scores:
            score=base_score if variant=='raw' else 'rqp_'+base_score
            initial=[r for path in sorted((source/'scores').glob('cal-*.npz')) for r in records(path,score)]
            daily={p.stem.split('-',1)[1]:records(p,score) for p in paths}
            initial_scores=np.asarray([r['score'] for r in initial],dtype=np.float64)
            initial_prediction=np.asarray([r['predicted'] for r in initial])
            packed={d:PackedDay(rows,config['known_codes'],d) for d,rows in daily.items()}
            for budget in protocol['budgets']:
                for delay in protocol['delays']:
                    for audit_seed in protocol['audit_seeds']:
                        name=f'{score}-budget{budget}-delay{delay}-seed{audit_seed}'
                        result=run_replay(initial,daily,config,budget,delay,audit_seed,metric_engines=packed)
                        path=out/(name+'.jsonl.gz');retained=retained_trace(path,result)
                        decision_count+=audit_methods(retained,cached,initial_scores,initial_prediction,score,config,budget,delay,audit_seed)
                        provenance.append(dict(path=str(path.relative_to(root)),sha256=sha256(path)))
                        batches=[(m,'',[r for r in retained if r['method']==m]) for m in config['methods']]
                        if variant=='raw':
                            gamma=protocol['aci_gamma'][variant][base_score][str(budget)][str(delay)]
                            aci=run_delayed_aci(initial,daily,config,budget,delay,audit_seed,gamma,metric_engines=packed)
                            aci_path=out/(name+'-aci.jsonl.gz');aci=retained_trace(aci_path,aci)
                            audit_trace(aci,cached,initial_scores,score,config,budget,delay,audit_seed,gamma)
                            decision_count+=sum(len(cached[r['date']]['labels']) for r in aci)
                            provenance.append(dict(path=str(aci_path.relative_to(root)),sha256=sha256(aci_path)))
                            batches.append(('delayed_aci_global',gamma,aci))
                        for method,gamma,rows in batches:
                            selected=[r for r in rows if protocol['confirmation_start']<=r['date']<=protocol['confirmation_end']]
                            if len(selected)!=len(protocol['confirmation_tables']):raise ValueError('Confirmation summary dates differ')
                            for isolated in [False,True]:
                                identity=dict(model_seed=args.seed,variant=variant,score=base_score,method=method,gamma=gamma,budget=budget,delay=delay,sampling_seed=audit_seed,view='input_disjoint' if isolated else 'all')
                                identity['confirmation_queries']=0 if method.startswith('fixed_') else sum(min(budget,len(cached[r['date']]['labels'])) for r in selected)
                                identity['warmup_queries']=0 if method.startswith('fixed_') else sum(min(budget,len(cached[r['date']]['labels'])) for r in rows if r['date']<protocol['confirmation_start'])
                                summaries.append(dict(**identity,**aggregate(selected,isolated)))
                        print(json.dumps(dict(completed_and_independently_audited=variant+':'+name)),flush=True)
    path=output/'summary.csv'
    with path.open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(summaries[0]));writer.writeheader();writer.writerows(summaries)
    verified_protocol(root)
    report=dict(status='FROZEN_CONFIRMATION_REPLAY_AND_INDEPENDENT_AUDIT_COMPLETE',model_seed=args.seed,protocol_sha256=seal['protocol_sha256'],summary_cells=len(summaries),decision_evaluations=decision_count,files=provenance,summary_sha256=sha256(path),checks=['Calendar-day warmup continuation','Shared uniformly sampled requests including unknown labels','Independent order statistics and feedback histories','All true-class decision counts and input-disjoint sensitivity','Prediction-time cumulative ACI errors','Confirmation-only period and daily estimands'])
    (output/'evaluation-complete.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='files'}),flush=True)


if __name__=='__main__':main()
