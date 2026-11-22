"""Run post-confirmation supplements in a completed fresh original rerun.

No scheduler, credentials or private transport is needed. Use --plan first;
only the new-partition-fit stage trains models. Existing result stages are
never overwritten. Presentation regeneration is a separate lightweight route.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, json, shutil, subprocess, sys

SOURCE=Path(__file__).resolve().parents[3]
STAGES=['initialize','original-supplements','new-partition-fit','new-partition-evaluation','presentation']
SCRIPTS=['array_replay','independent_array_audit','analyse_saved_counts','static_hqr','risk_curves','bootstrap_risk_matching','development_risk_selection','mechanism','sensitivity','class_attribution','new_partition_pipeline','score_new_partition','evaluate_new_partition','audit_new_partition_aggregates','plot_risk_curves','plot_class_time','plot_mechanism','build_revision_tables','run_revision_reproduction']
def steps(root):
    def call(name,*args):return [sys.executable,str(root/'revisions/major-v1/scripts'/(name+'.py')),*args]
    return {'initialize':[], 'original-supplements':[call(s) for s in ['analyse_saved_counts','static_hqr','risk_curves','bootstrap_risk_matching','development_risk_selection','mechanism','sensitivity','class_attribution']], 'new-partition-fit':[call('new_partition_pipeline','--root',str(root))], 'new-partition-evaluation':[call('evaluate_new_partition'),call('audit_new_partition_aggregates')], 'presentation':[call(s) for s in ['plot_risk_curves','plot_class_time','plot_mechanism','build_revision_tables']]}
def initialize(root):
    receipt=root/'rerun-analysis.json'
    if not receipt.is_file() or json.loads(receipt.read_text()).get('status')!='COMPLETE':raise ValueError('First complete the original scripts/run_reproduction.py analysis stage in this separate work directory.')
    target=root/'revisions/major-v1'
    if target.exists():raise FileExistsError('Revision workspace already exists; inspect it instead of overwriting.')
    target.mkdir(parents=True)
    for directory in ['scripts','configs','results','figures','receipts','manuscript/tables']:(target/directory).mkdir(parents=True,exist_ok=True)
    for name in SCRIPTS:shutil.copy2(SOURCE/'revisions/major-v1/scripts'/(name+'.py'),target/'scripts'/(name+'.py'))
    for p in (SOURCE/'revisions/major-v1/configs').glob('split*.json'):shutil.copy2(p,target/'configs'/p.name)
    for name in ['analysis-plan.json','new-partition-plan.json']:shutil.copy2(SOURCE/'revisions/major-v1'/name,target/name)
    (target/'rerun-provenance.json').write_text(json.dumps({'status':'FRESH_SUPPLEMENTARY_REPRODUCTION','original_workdir':str(root),'created_at_utc':datetime.now(timezone.utc).isoformat(),'scientific_status':'Post-confirmation exploratory analyses; original reported freeze is not extended retroactively.','partition_seed':20261122,'partition_screening':False},indent=2)+'\n')
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--work-dir',type=Path,required=True);p.add_argument('--stage',choices=['all']+STAGES,default='all');p.add_argument('--plan',action='store_true');a=p.parse_args();root=a.work_dir.resolve();selected=STAGES if a.stage=='all' else [a.stage];commands=steps(root)
    if a.plan:
        print(json.dumps({'status':'PLAN_ONLY_NO_EXPERIMENT_EXECUTED','prerequisite':'A completed fresh original rerun in --work-dir, with historical/future score caches and its own valid seal.','stages':{s:commands[s] for s in selected},'training':'Three new-partition source fits only,50 epochs each. Existing original source models are reused. Allocate a CUDA environment for fitting/scoring; CPU-only for evaluation and statistics.'},indent=2));return
    if root==SOURCE:raise ValueError('Use a separate completed original rerun; published scientific results are never overwritten.')
    if 'initialize' in selected:initialize(root)
    elif not (root/'revisions/major-v1/rerun-provenance.json').is_file():raise ValueError('Initialize the revision workspace first.')
    for stage in selected:
        receipt=root/'revisions/major-v1/receipts'/('rerun-'+stage+'.json')
        if receipt.exists():raise FileExistsError('Stage already has a receipt; inspect its outputs and status before retrying.')
        record={'stage':stage,'status':'RUNNING','commands':commands[stage],'started_at_utc':datetime.now(timezone.utc).isoformat()};receipt.write_text(json.dumps(record,indent=2)+'\n')
        try:
            for command in commands[stage]:subprocess.run(command,cwd=root,check=True)
        except BaseException:
            record['status']='FAILED_OR_INTERRUPTED_INSPECT_RETAINED_OUTPUTS';receipt.write_text(json.dumps(record,indent=2)+'\n');raise
        record.update(status='COMPLETE',finished_at_utc=datetime.now(timezone.utc).isoformat());receipt.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'status':'SELECTED_SUPPLEMENTARY_STAGES_COMPLETE','stages':selected}))
if __name__=='__main__':main()
