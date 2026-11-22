"""Execute only the predeclared new service partition; preserve every run."""
import argparse,json,subprocess,sys,time
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root;plan=json.loads((root/'revisions/major-v1/new-partition-plan.json').read_text());log=root/'revisions/major-v1/receipts/new-partition-pipeline.json'
 log.parent.mkdir(parents=True,exist_ok=True)
 if log.exists():raise FileExistsError('Inspect the existing pipeline; no automatic overwrite or restart.')
 state={'status':'RUNNING','class_selection_seed':plan['class_selection_seed'],'started_at_unix':time.time(),'completed':[]};log.write_text(json.dumps(state,indent=2)+'\n')
 for entry in plan['models']:
  stages=[('scripts/prepare.py',['--config',entry['raw_config']])]
  for variant in ['shift','reference']:stages += [('scripts/normalize_scores.py',['--config',entry[variant+'_config']]),('scripts/audit_reference_maps.py',['--config',entry[variant+'_config']])]
  stages += [('revisions/major-v1/scripts/score_new_partition.py',['--seed',str(entry['seed'])])]
  for script,args in stages:
   command=[sys.executable,str(root/script),'--root',str(root),*args];print(json.dumps({'stage_started':script,'seed':entry['seed']}),flush=True)
   subprocess.run(command,cwd=root,check=True);state['completed'].append({'seed':entry['seed'],'script':script,'finished_at_unix':time.time()});log.write_text(json.dumps(state,indent=2)+'\n')
 state.update(status='COMPLETE_ALL_THREE_PREDECLARED_MODEL_SEEDS',finished_at_unix=time.time());log.write_text(json.dumps(state,indent=2)+'\n');print(json.dumps(state),flush=True)
if __name__=='__main__':main()
