"""Fit reference maps on a fixed half of initial labels; reserve the other half.

Reads only the already-produced development score cache, never raw flow tables.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from configio import study_config
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from reference_quantiles import fit_maps, fit_shift_maps, transform


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assignment(date, indices, seed):
    return np.asarray([hashlib.sha256(f'reference-role:{seed}:{date}:{int(i)}'.encode()).digest()[0]%2==0 for i in indices],dtype=bool)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--config',required=True);args=parser.parse_args();root=args.root
    target_path,config,output=study_config(root,args.config)
    upstream_path,upstream,folder=study_config(root,config['upstream_config'])
    prepared=json.loads((folder/'prepare-complete.json').read_text())
    if prepared['protected_flow_tables_read'] or prepared['config_sha256']!=digest(upstream_path):raise ValueError('Upstream identity or protected-date boundary differs')
    known=np.asarray(prepared['known_codes']);settings=config['reference_quantiles'];fit=[];fit_ids=[];guard_ids=[];calibration=[]
    for path in sorted((folder/'scores').glob('cal-*.npz')):
        date=path.stem.split('-',1)[1]
        if date>=config['development_start']:raise ValueError('Reference fitting overlaps development')
        with np.load(path,allow_pickle=False) as archive:data={k:archive[k] for k in archive.files}
        if not data['known'].all() or not np.isin(data['labels'],known).all() or not np.array_equal(data['codes'],known):raise ValueError('Reference maps require the original known calibration population')
        selected=assignment(date,data['indices'],settings['role_seed']);prediction=data['codes'][data['prediction']]
        fit.append((data,prediction,selected));calibration.append((path,data,selected))
        fit_ids.extend(f'{date}:{int(i)}' for i in data['indices'][selected]);guard_ids.extend(f'{date}:{int(i)}' for i in data['indices'][~selected])
    if len(guard_ids)<1000:raise ValueError('Insufficient disjoint initial risk-calibration rows')
    if set(fit_ids)&set(guard_ids) or len(set(fit_ids))!=len(fit_ids) or len(set(guard_ids))!=len(guard_ids):raise ValueError('Reference-fit and risk-calibration rows overlap')
    predictions=np.concatenate([p[s] for _,p,s in fit]);maps={}
    for score in settings['base_scores']:
        values=np.concatenate([d[score][s] for d,_,s in fit])
        if settings.get('kind') == 'additive_quantile':
            maps[score]=fit_shift_maps(values,predictions,known.tolist(),settings['probabilities'][0],settings['source_prior_count'])
        else:
            maps[score]=fit_maps(values,predictions,known.tolist(),settings['probabilities'],settings['source_prior_count'])
    output.mkdir(exist_ok=False);scores_dir=output/'scores';scores_dir.mkdir()
    state=dict(status='FIXED_REFERENCE_MAPS_FIT',base_scores=settings['base_scores'],maps=maps,fit_rows=len(fit_ids),guard_rows=len(guard_ids),role_assignment='SHA256 fixed source-row ID split; fit if low bit is zero',role_seed=settings['role_seed'],fit_ids_sha256=hashlib.sha256(json.dumps(fit_ids,separators=(',',':')).encode()).hexdigest(),guard_ids_sha256=hashlib.sha256(json.dumps(guard_ids,separators=(',',':')).encode()).hexdigest(),upstream_model_sha256=prepared['model_sha256'],upstream_config_sha256=prepared['config_sha256'])
    (output/'reference-maps.json').write_text(json.dumps(state,indent=2)+'\n')
    for path,data,selected in calibration:
        prediction=data['codes'][data['prediction']]
        for score in settings['base_scores']:data['rqp_'+score]=transform(data[score],prediction,maps[score])
        saved={k:v if k=='codes' else v[~selected] for k,v in data.items()}
        np.savez_compressed(scores_dir/path.name,**saved)
    for path in sorted((folder/'scores').glob('dev-*.npz')):
        date=path.stem.split('-',1)[1]
        if date>=config['protected_start']:raise ValueError('Protected future score cache')
        with np.load(path,allow_pickle=False) as archive:data={k:archive[k] for k in archive.files}
        prediction=data['codes'][data['prediction']]
        for score in settings['base_scores']:data['rqp_'+score]=transform(data[score],prediction,maps[score])
        np.savez_compressed(scores_dir/path.name,**data)
    report=dict(prepared)
    report.update(status='DEVELOPMENT_TRANSFORMED_SCORES_COMPLETE',config_sha256=digest(target_path),upstream_config_sha256=prepared['config_sha256'],upstream_output_run=folder.name,upstream_prepare_runner_sha256=prepared['runner_sha256'],runner_sha256=digest(__file__),preparation_role='CACHED_SCORE_TRANSFORM_NO_CLASSIFIER_FIT',reference_maps_sha256=digest(output/'reference-maps.json'),normalization_fit_rows=len(fit_ids),calibration_rows=len(guard_ids),calibration_policy='Initial labelled pool split by fixed row-ID hash into reference fitting and disjoint risk calibration. All deployment audit streams and costs unchanged.',converter_raw_flow_tables_read=[])
    report['model_artifact']=str((folder/'model.pt').relative_to(root))
    report['geometry_artifact']=str((folder/'geometry-state.npz').relative_to(root)) if prepared.get('geometry_sha256') else None
    report['source_rows_artifact']=str((folder/'source-rows.json').relative_to(root))
    report['upstream_calibration_population']={key:prepared.get(key) for key in ['calibration_rows','calibration_source_overlap_rows','calibration_repeated_input_rows']}
    report.pop('calibration_source_overlap_rows',None);report.pop('calibration_repeated_input_rows',None)
    (output/'prepare-complete.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],reference_fit_rows=len(fit_ids),risk_calibration_rows=len(guard_ids),output=output.name)),flush=True)


if __name__=='__main__':main()
