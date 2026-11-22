"""Audit new array replay using the original separate NumPy audit path."""
from datetime import datetime, timedelta
import hashlib, random, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from audit import independent_cutoffs,recount

def audit(initial,daily,classes,method,counts,ledger,alpha,budget,delay,seed,window=28):
    requests={};decisions=0
    for t,day in enumerate(daily):
        current=datetime.strptime(day['date'],'%Y%m%d').date()
        past=[(d,ids) for d,ids in requests.values() if current-timedelta(days=window)<=datetime.strptime(d['date'],'%Y%m%d').date()<=current-timedelta(days=delay)]
        scores=np.concatenate([d['score'][ids[d['known'][ids]]] for d,ids in past]) if past else np.empty(0)
        prediction=np.concatenate([d['predicted'][ids[d['known'][ids]]] for d,ids in past]) if past else np.empty(0,int)
        expected=independent_cutoffs(initial['score'],initial['predicted'],scores,prediction,classes,method,alpha,100)
        observed={str(c):float(q) if q!='+inf' else np.inf for c,q in zip(classes,ledger[t]['cutoffs'])}
        assert np.allclose([expected[str(c)] for c in classes],[observed[str(c)] for c in classes],rtol=1e-12,atol=1e-12)
        checked=np.zeros((53,3),dtype=np.int64)
        for row in recount(day,'score',observed):checked[row['label']]=[row['n'],row['rejected'],row['correct_accepted']]
        assert np.array_equal(checked,counts[t]),(day['date'],method,'Independent class counts differ')
        decisions+=len(day['labels'])
        if not method.startswith('fixed_'):
            token=int.from_bytes(hashlib.sha256(f"{seed}:{day['date']}".encode()).digest()[:8],'big')
            ids=np.array(sorted(random.Random(token).sample(range(len(day['labels'])),min(budget,len(day['labels'])))),dtype=int)
            requests[day['date']]=(day,ids)
    return decisions
