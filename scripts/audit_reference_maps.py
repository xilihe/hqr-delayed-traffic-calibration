"""Reconstruct reference maps and transformed caches through separate arithmetic."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from configio import study_config


def empirical_quantiles(values,p):
    ordered=np.sort(np.asarray(values,dtype=np.float64));locations=p*(len(ordered)-1)
    left=np.floor(locations).astype(int);right=np.ceil(locations).astype(int)
    return ordered[left]+(ordered[right]-ordered[left])*(locations-left)


def independent_transform(scores,prediction,maps):
    output=np.empty(len(scores),dtype=np.float64)
    for code,mapping in maps.items():
        selected=prediction==int(code);raw=scores[selected].astype(np.float64)
        x=np.asarray(mapping['x']);z=np.asarray(mapping['z'])
        segment=np.clip(np.searchsorted(x,raw,side='right')-1,0,len(x)-2)
        output[selected]=z[segment]+(raw-x[segment])*(z[segment+1]-z[segment])/(x[segment+1]-x[segment])
    return output


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--config',required=True);args=parser.parse_args();root=args.root
    target_path,config,folder=study_config(root,args.config);source_path,_,source_folder=study_config(root,config['upstream_config'])
    prepared=json.loads((folder/'prepare-complete.json').read_text());state=json.loads((folder/'reference-maps.json').read_text());settings=config['reference_quantiles']
    if prepared['protected_flow_tables_read'] or prepared['config_sha256']!=hashlib.sha256(target_path.read_bytes()).hexdigest():raise ValueError('Target identity or future boundary differs')
    if state['upstream_config_sha256']!=hashlib.sha256(source_path.read_bytes()).hexdigest():raise ValueError('Reference maps use a different source run')
    samples=[];roles={};fit_ids=[];guard_ids=[]
    for path in sorted((source_folder/'scores').glob('cal-*.npz')):
        date=path.stem.split('-',1)[1]
        with np.load(path,allow_pickle=False) as archive:data={k:archive[k] for k in archive.files}
        indices=[i for i,row in enumerate(data['indices']) if int(hashlib.sha256(('reference-role:'+str(settings['role_seed'])+':'+date+':'+str(int(row))).encode()).hexdigest()[:2],16)%2==0]
        fit_mask=np.zeros(len(data['indices']),dtype=bool);fit_mask[indices]=True;roles[path.name]=~fit_mask
        fit_ids.extend(date+':'+str(int(i)) for i in data['indices'][fit_mask]);guard_ids.extend(date+':'+str(int(i)) for i in data['indices'][~fit_mask])
        if not data['known'].all() or date>=config['development_start']:raise ValueError('Reference fitting uses unavailable or non-known labels')
        samples.append({**{s:data[s][fit_mask] for s in settings['base_scores']},'prediction':data['codes'][data['prediction'][fit_mask]]})
    if set(fit_ids)&set(guard_ids):raise ValueError('Reference fitting overlaps risk calibration')
    for name,ids in [('fit',fit_ids),('guard',guard_ids)]:
        if state[name+'_ids_sha256']!=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest():raise ValueError('Row-role checksum differs')
    if prepared['calibration_rows']!=len(guard_ids) or prepared['normalization_fit_rows']!=len(fit_ids):raise ValueError('Historical label counts differ')
    prediction=np.concatenate([d['prediction'] for d in samples]);p=np.asarray(settings['probabilities']);target=np.log(p/(1-p))
    for score in settings['base_scores']:
        values=np.concatenate([d[score] for d in samples]);global_x=empirical_quantiles(values,p)
        for code,mapping in state['maps'][score].items():
            local=values[prediction==int(code)];n=len(local);weight=n/(n+settings['source_prior_count'])
            x=weight*empirical_quantiles(local,p)+(1-weight)*global_x if n else global_x
            unique,inverse=np.unique(x,return_inverse=True);z=np.bincount(inverse,weights=target)/np.bincount(inverse)
            if settings.get('kind') == 'additive_quantile':
                center=float(x[0]);unique=np.asarray([center-1,center+1]);z=np.asarray([-1.,1.])
            if len(unique)<2:
                scale=max(float(np.max(values.astype(np.float64))-np.min(values.astype(np.float64))),1.0);center=float(unique[0]);unique=np.asarray([center-scale,center+scale]);z=np.asarray([-1.,1.])
            if mapping['source_rows']!=n or mapping['source_weight']!=weight or not np.allclose(unique,mapping['x'],rtol=1e-11,atol=1e-11) or not np.allclose(z,mapping['z'],rtol=1e-11,atol=1e-11):raise ValueError('Quantile map differs from independent empirical-order reconstruction')
    checked=[];max_error=0.
    for path in sorted((source_folder/'scores').glob('*.npz')):
        date=path.stem.split('-',1)[1]
        if date>=config['protected_start']:raise ValueError('Protected cache in transformation audit')
        with np.load(path,allow_pickle=False) as archive:original={k:archive[k] for k in archive.files}
        with np.load(folder/'scores'/path.name,allow_pickle=False) as archive:transformed={k:archive[k] for k in archive.files}
        selected=roles[path.name] if path.name.startswith('cal-') else np.ones(len(original['indices']),dtype=bool)
        for key,values in original.items():
            expected=values if key=='codes' else values[selected]
            if not np.array_equal(expected,transformed[key]):raise ValueError('Reference transformation changed original labels, predictions, row identity or raw scores')
        prediction=transformed['codes'][transformed['prediction']]
        for score in settings['base_scores']:
            values=independent_transform(transformed[score],prediction,state['maps'][score]);stored=transformed['rqp_'+score]
            if not np.allclose(values,stored,rtol=1e-10,atol=1e-10):raise ValueError('Transformed scores differ from separate piecewise-linear arithmetic')
            max_error=max(max_error,float(np.max(np.abs(values-stored),initial=0.)))
        checked.append(dict(file=path.name,sha256=hashlib.sha256((folder/'scores'/path.name).read_bytes()).hexdigest()))
    report=dict(status='PASS_INDEPENDENT_REFERENCE_MAP_AND_SCORE_RECONSTRUCTION',fit_rows=len(fit_ids),risk_calibration_rows=len(guard_ids),files=checked,max_absolute_arithmetic_difference=max_error,protected_flow_tables_read=[],checks=['Disjoint deterministic historical roles','Independent empirical-order quantile anchors','Source-only prediction-group maps','All raw scores, labels, predictions and row identities unchanged','Separate piecewise-linear reconstruction of all derived scores'])
    (folder/'normalization-audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='files'}),flush=True)


if __name__=='__main__':main()
