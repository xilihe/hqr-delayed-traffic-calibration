"""Reproduce development-only ACI selection from the retained audit grid.

Added for portable reproduction after the original internal freeze. The
original selection JSON, already frozen before confirmation, is never edited.
"""
import argparse
import csv
import json
from pathlib import Path
import statistics


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',default='research/verification/aci-selection-new-reproduction.json');args=parser.parse_args()
    root=args.root.resolve();relative=Path(args.output)
    if relative.is_absolute() or '..' in relative.parts:raise ValueError('Output must remain inside the study root')
    output=root/relative
    if output.exists():raise FileExistsError('Retain the original selection; choose a new output path')
    selection={};ledger=[]
    for variant,run in [('raw','pilot-transformer-v1'),('reference','pilot-transformer-rqp-v1')]:
        path=root/'outputs'/run/'aci-comparison/summary.csv'
        with path.open() as stream:rows=list(csv.DictReader(stream))
        selection[variant]={}
        for score in dict.fromkeys(r['score'] for r in rows):
            base=score.removeprefix('rqp_');selection[variant][base]={}
            for budget in [32,128,512]:
                selection[variant][base][str(budget)]={}
                for delay in [1,7]:
                    candidates=[]
                    for gamma in [.0001,.001,.01]:
                        group=[r for r in rows if r['score']==score and int(r['budget'])==budget and int(r['delay'])==delay and float(r['gamma'])==gamma]
                        if len(group)!=3 or {int(r['sampling_seed']) for r in group}!={101,102,103}:raise ValueError('Development grid lacks the three declared audit streams')
                        frr=statistics.mean(float(r['mean_daily_known_frr']) for r in group);excess=statistics.mean(float(r['mean_daily_known_frr_excess']) for r in group)
                        criterion=2*excess-(frr-.05);candidates.append((criterion,gamma))
                        ledger.append(dict(variant=variant,score=base,budget=budget,delay=delay,gamma=gamma,mean_daily_frr=frr,mean_daily_positive_excess=excess,mean_absolute_target_deviation=criterion))
                    selection[variant][base][str(budget)][str(delay)]=min(candidates)[1]
    report=dict(status='DEVELOPMENT_ONLY_ACI_SELECTION_COMPLETE',criterion='Minimum mean daily absolute known-FRR deviation from alpha=.05, averaged over all three audit streams; ties favor smaller gamma.',selected=selection,development_grid=ledger,confirmation_scores_read=False)
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],grid_entries=len(ledger),output=str(relative),confirmation_scores_read=False)),flush=True)


if __name__=='__main__':main()
