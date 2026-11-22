"""Run the published pipeline in a fresh directory, without lab transport.

The published seal is retained in the release as a reference. A fresh rerun
creates its own models, numerical evidence and seal. No completed or partial
scientific output is silently replaced. Use --plan before allocating compute.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

SOURCE = Path(__file__).resolve().parents[1]
STAGES = ['initialize', 'fit', 'maps', 'development', 'freeze', 'confirmation', 'analysis']
DATA_SHA256 = '6b323c4c53b7c9d0ec9011f895b3c5ba6b35247576e41ef19fe84ab8632810ab'

def pipeline(root):
    def cmd(script, *extra):
        return [sys.executable, str(root / 'scripts' / (script + '.py')), '--root', str(root), *extra]
    def cfg(script, name):
        return cmd(script, '--config', 'configs/' + name + '.json')
    steps = {'initialize': [cmd('acquire'), cmd('verify_confirmation_engine')]}
    steps['fit'] = [cfg('prepare', n) for n in ['transformer-v1', 'transformer-seed43-v1', 'transformer-seed44-v1']]
    map_configs = ['transformer-rqp-v1', 'transformer-shift-v1', 'transformer-shift-full-v1', 'transformer-rqp-seed43-v1', 'transformer-shift-seed43-v1', 'transformer-rqp-seed44-v1', 'transformer-shift-seed44-v1']
    steps['maps'] = [cfg(script, n) for n in map_configs for script in ['normalize_scores', 'audit_reference_maps']]
    steps['development'] = []
    for n in ['transformer-v1', 'transformer-rqp-v1', 'transformer-shift-v1']:
        steps['development'] += [cfg('evaluate', n), cfg('audit', n)]
        if n != 'transformer-shift-v1':
            steps['development'].append(cfg('evaluate_aci', n))
        steps['development'].append(cfg('summarize_periods', n))
    steps['development'].append(cmd('select_aci', '--output', 'research/verification/aci-selection.json'))
    steps['freeze'] = [cmd('freeze_confirmation')]
    steps['confirmation'] = [cmd(script, '--seed', str(seed)) for seed in [42, 43, 44] for script in ['score_confirmation', 'evaluate_confirmation']]
    steps['analysis'] = [cmd('analyse_confirmation')]
    return steps

def initialize(root, data):
    if root.exists():
        raise FileExistsError('Use a new work directory. Existing results are never replaced.')
    if not data.is_file():
        raise FileNotFoundError('Provide the official XS HDF5 file with --data.')
    if data.stat().st_size != 3544625537:
        raise ValueError('The official XS file has 3544625537 bytes.')
    h = hashlib.sha256()
    with data.open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    if h.hexdigest() != DATA_SHA256:
        raise ValueError('The official XS dataset SHA256 differs.')
    root.mkdir(parents=True)
    for folder in ['src', 'scripts', 'configs']:
        shutil.copytree(SOURCE / folder, root / folder, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for relative in ['research/confirmation-design-working.md', 'research/verification/development-review.json']:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SOURCE / relative, target)
    (root / 'data').mkdir()
    (root / 'data/CESNET-QUICEXT-25-XS.h5').symlink_to(data)
    (root / 'outputs').mkdir()
    (root / 'rerun-provenance.json').write_text(json.dumps({'status':'FRESH_REPRODUCTION', 'created_at_utc': datetime.now(timezone.utc).isoformat(), 'source_release':str(SOURCE), 'published_historical_decisions_retained':True, 'original_reported_results_are_not_new_observations':True}, indent=2)+'\n')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work-dir', type=Path, required=True)
    p.add_argument('--data', type=Path)
    p.add_argument('--stage', choices=['all']+STAGES, default='all')
    p.add_argument('--plan', action='store_true')
    a=p.parse_args();root=a.work_dir.resolve()
    if root == SOURCE or SOURCE.is_relative_to(root):
        raise ValueError('Choose a separate work directory, outside the published repository.')
    selected = STAGES if a.stage == 'all' else [a.stage]
    steps=pipeline(root)
    if a.plan:
        print(json.dumps({'status':'PLAN_ONLY_NO_EXPERIMENT_EXECUTED', 'stages':{s:steps[s] for s in selected}, 'training':'Three source-model fits, 50 epochs each; allocate suitable compute before execution.'},indent=2))
        return
    if 'initialize' in selected:
        if a.data is None:
            p.error('--data is required for initialization')
        initialize(root,a.data.resolve())
    elif not (root/'rerun-provenance.json').is_file():
        raise ValueError('Initialize a fresh rerun first.')
    for stage in selected:
        receipt=root/('rerun-'+stage+'.json')
        if receipt.exists():
            raise FileExistsError('Stage already has a receipt; inspect it before any further action.')
        record={'stage':stage,'status':'RUNNING','commands':steps[stage],'started_at_utc':datetime.now(timezone.utc).isoformat()}
        receipt.write_text(json.dumps(record,indent=2)+'\n')
        try:
            for command in steps[stage]:
                subprocess.run(command,cwd=root,check=True)
        except BaseException:
            record['status']='INTERRUPTED_OR_FAILED_INSPECT_RETAINED_OUTPUTS'
            receipt.write_text(json.dumps(record,indent=2)+'\n')
            raise
        record.update(status='COMPLETE',finished_at_utc=datetime.now(timezone.utc).isoformat())
        receipt.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'status':'SELECTED_REPRODUCTION_STAGES_COMPLETE','stages':selected}))

if __name__=='__main__':
    main()
