"""TEST_ONLY_SIMULATION: engineering audit, never research-result evidence."""
import argparse
from datetime import date,timedelta
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from replay import run_replay
from online_aci import run_delayed_aci
from fast_metrics import PackedDay
from evaluate_confirmation import audit_methods
from evaluate_aci import audit_trace
from reference_quantiles import fit_shift_maps,transform
from analyse_confirmation import weighted_metrics,block_weights
from summarize_periods import aggregate


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);args=parser.parse_args()
    initial=[dict(id='cal:'+str(i),date='20240101',predicted=i%2,label=i%2,known=True,kind='known',source_disjoint=True,score=float(i)/80) for i in range(80)]
    config=dict(alpha=.05,window_days=7,known_codes=[0,1],pooling_pseudocount=100,methods=['fixed_global','fixed_classwise','rolling_global','rolling_classwise','pooled_classwise'])
    cached={};daily={}
    for day_index in [0,1,2,4,5,6,7,8,10,11]:
        d=(date(2024,1,2)+timedelta(days=day_index)).strftime('%Y%m%d');i=np.arange(200);labels=(i+day_index)%4;known=labels<2;prediction=(i//3+day_index)%2;score=((i*7+day_index*3)%201)/201
        data=dict(labels=labels,known=known,background=labels==3,indices=i,codes=np.asarray([0,1]),prediction=prediction,negative_energy=score,source_overlap=i%13==0);cached[d]=data
        daily[d]=[dict(id=d+':'+str(int(j)),date=d,predicted=int(prediction[j]),label=int(labels[j]),known=bool(known[j]),kind='known' if known[j] else 'background' if labels[j]==3 else 'heldout',source_disjoint=not bool(data['source_overlap'][j]),score=float(score[j])) for j in i]
    packed={d:PackedDay(rows,config['known_codes'],d) for d,rows in daily.items()}
    result=run_replay(initial,daily,config,8,3,101,metric_engines=packed)
    decisions=audit_methods(json.loads(json.dumps(result)),cached,np.asarray([r['score'] for r in initial]),np.asarray([r['predicted'] for r in initial]),'negative_energy',config,8,3,101)
    aci=run_delayed_aci(initial,daily,config,8,3,101,.001,metric_engines=packed)
    audit_trace(json.loads(json.dumps(aci)),cached,np.asarray([r['score'] for r in initial]),'negative_energy',config,8,3,101,.001)
    values=np.linspace(-7,9,80);prediction=np.arange(80)%2
    maps=fit_shift_maps(values,prediction,[0,1,2],.95,20)
    for c in [0,1,2]:
        selected=values[prediction==c];n=len(selected);q=np.quantile(selected,.95) if n else np.quantile(values,.95);center=(n*q+20*np.quantile(values,.95))/(n+20)
        probe=np.asarray([-100.,0.,100.]);actual=transform(probe,np.full(3,c),maps)
        if not np.allclose(actual,probe-center,rtol=1e-12,atol=1e-12):raise ValueError('Historical additive control differs from analytic subtraction')
    single=[r for r in result if r['method']=='rolling_global']
    cube=np.zeros((1,len(single)+2,4,3),dtype=float)
    for d,row in enumerate(single):
        for count in row['class_counts']:cube[0,d,count['label']]=[count['n'],count['rejected'],count['correct_accepted']]
    masks={'known':np.asarray([True,True,False,False]),'heldout':np.asarray([False,False,True,False]),'background':np.asarray([False,False,False,True])}
    weights=np.ones((1,len(single)+2));observed=weighted_metrics(cube,weights,masks);reference=aggregate(single)
    for field,expected in [('known_frr',reference['mean_daily_known_frr']),('known_correct_accepted',reference['mean_daily_known_correct_accepted']),('heldout_macro_detection',reference['period_heldout_macro_rejection']),('background_macro_detection',reference['period_background_macro_rejection'])]:
        if not np.isclose(observed[field][0],expected,rtol=1e-12,atol=1e-12):raise ValueError('Weighted bootstrap estimand differs from count-based aggregation')
    weights[0,0]=3;observed=weighted_metrics(cube,weights,masks);reference=aggregate(single+[single[0],single[0]])
    if not np.isclose(observed['known_frr'][0],reference['mean_daily_known_frr']):raise ValueError('Repeated-date weighting differs')
    first=block_weights(12,7,20,20261122)
    if not np.array_equal(first,block_weights(12,7,20,20261122)) or not np.all(first.sum(axis=1)==12):raise ValueError('Paired calendar resampling is not reproducible')
    report=dict(status='PASS_CONFIRMATION_ENGINE_TEST_ONLY_SIMULATION',result_claim_allowed=False,decision_evaluations=decisions,checks=['Independent five-method reconstruction with delayed arrival and calendar gaps','Matched prediction-time ACI cumulative-error reconstruction','Historical additive quantile control including absent predicted class','Strict separation of engineering fixtures from result evidence','Weighted bootstrap estimands equal independent period aggregation including missing calendar dates','Reproducible paired calendar block weights'])
    (args.root/'outputs/confirmation-engine-preflight-v2.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
