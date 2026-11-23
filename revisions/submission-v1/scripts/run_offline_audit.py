"""Run the actual low-cost audit in a new isolated public-checkout copy."""
from pathlib import Path
import argparse,json,shutil,subprocess,sys,time,hashlib,platform
R=Path(__file__).resolve().parents[3]
p=argparse.ArgumentParser();p.add_argument('--work-dir',required=True,type=Path);p.add_argument('--plan',action='store_true');args=p.parse_args();work=args.work_dir.resolve()
commands=[['revisions/submission-v1/scripts/repair_risk_bootstrap.py'],['revisions/submission-v1/scripts/verify_risk_repair.py'],['revisions/submission-v1/scripts/verify_offline_results.py'],['revisions/major-v1/scripts/analyse_saved_counts.py'],['scripts/build_result_tables.py','--root',str(work)],['revisions/major-v1/scripts/build_revision_tables.py'],['revisions/final-v1/scripts/build_final_tables_figures.py'],['revisions/submission-v1/scripts/build_submission_tables.py']]
if args.plan:
 print(json.dumps({'mode':'LOW_COST_SAVED_DECISION_REPRODUCTION_NOT_TRAINING','new_work_dir':str(work),'commands':commands},indent=2));sys.exit(0)
manifest=R/'submission-release-manifest.json'
if not manifest.is_file():raise RuntimeError('Run this driver from the curated public release containing submission-release-manifest.json; never copy an arbitrary private project tree.')
if work.exists():raise FileExistsError('Choose a new work directory; do not overwrite a completed or partial audit.')
skip=['revisions/submission-v1/results/risk-tracking/','revisions/submission-v1/results/offline-reproduction/','revisions/submission-v1/manuscript/','revisions/submission-v1/results/independent-risk-verification.json','revisions/submission-v1/results/table-generation.json']
files=json.loads(manifest.read_text())['files'];work.mkdir(parents=True)
for x in files:
 name=x['path']
 if any(name.startswith(z) for z in skip):continue
 src=R/name
 if hashlib.sha256(src.read_bytes()).hexdigest()!=x['sha256']:raise RuntimeError('Curated input hash changed: '+name)
 dest=work/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
log=work/'offline-audit-execution';log.mkdir();started=time.perf_counter();records=[]
for i,command in enumerate(commands):
 before=time.perf_counter();full=[sys.executable,*command]
 with (log/f'{i+1:02d}.stdout').open('w') as out,(log/f'{i+1:02d}.stderr').open('w') as err:r=subprocess.run(full,cwd=work,stdout=out,stderr=err)
 event={'command':command,'exit_code':r.returncode,'elapsed_seconds':time.perf_counter()-before};records.append(event);(log/'execution.json').write_text(json.dumps({'status':'RUNNING' if r.returncode==0 else 'FAILED','steps':records},indent=2)+'\n');print(json.dumps(event),flush=True)
 if r.returncode:raise SystemExit('Failed command retained in '+str(log))
summary={'status':'PASS_ALL_EIGHT_OFFLINE_STEPS','elapsed_seconds':time.perf_counter()-started,'python':platform.python_version(),'scope':'Actualsaved-countstatistics, pairedbootstrap, tables andfigures; no training, rawtraffic download, score inference or policy replay.','steps':records};(log/'execution.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)
