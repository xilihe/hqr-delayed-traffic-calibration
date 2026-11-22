"""Score the frozen future cohort with an already-fitted source classifier.

No fitting, label-based future sampling, or calibration updates occur here.
The sealed implementation and historical artifacts are checked before any
confirmation flow table is opened. Prediction caches are for offline replay.
"""
import argparse
import json
from pathlib import Path
import sys
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import tables
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from frozen_protocol import verified_protocol, sha256
from reference_quantiles import transform as map_scores
from model_adapters import build_model
from cesnet_adapter import transform
from prepare import inputs_at, signatures, infer


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--seed',type=int,required=True);args=parser.parse_args();root=args.root
    protocol,seal=verified_protocol(root)
    entries=[e for e in protocol['models'] if e['seed']==args.seed]
    if len(entries)!=1:raise ValueError('Model seed is absent or repeated in frozen protocol')
    entry=entries[0];source=root/entry['raw_run'];prepared=json.loads((source/'prepare-complete.json').read_text())
    source_config=json.loads((root/entry['source_config']).read_text())
    if prepared['protected_flow_tables_read'] or sha256(root/entry['source_config'])!=prepared['config_sha256']:
        raise ValueError('Historical model identity or protected-date history differs')
    if prepared['source_rows_sha256']!=protocol['source_rows_sha256'] or prepared['known_codes']!=protocol['known_codes']:
        raise ValueError('Model replication uses a different source cohort or class partition')
    dataset=root/'data/CESNET-QUICEXT-25-XS.h5';inventory=json.loads((root/'outputs/inventory.json').read_text())
    if sha256(dataset)!=protocol['dataset_sha256']:raise ValueError('Dataset version differs')
    out=root/entry['confirmation_run'];out.mkdir(exist_ok=False)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type!='cuda':raise ValueError('Scheduled GPU is required for frozen classifier scoring')
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    checkpoint=torch.load(source/'model.pt',map_location='cpu',weights_only=False)
    known=np.asarray(protocol['known_codes']);model=build_model(len(known),source_config).to(device)
    model.load_state_dict(checkpoint['state_dict']);model.eval()
    if list(checkpoint['known_codes'])!=known.tolist():raise ValueError('Checkpoint class identity differs')
    with np.load(source/'geometry-state.npz',allow_pickle=False) as archive:geometry={k:archive[k] for k in archive.files}
    maps={}
    for variant in ['reference','shift']:
        folder=root/entry[variant+'_run'];state=json.loads((folder/'reference-maps.json').read_text())
        if state['upstream_model_sha256']!=prepared['model_sha256']:raise ValueError('Historical map uses another classifier')
        maps[variant]=state['maps']
    for variant in ['raw','reference','shift']:(out/variant/'scores').mkdir(parents=True)
    background={int(v) for k,v in inventory['apps'].items() if k.casefold().endswith('background')}
    futures=[t for t in inventory['tables'] if protocol['confirmation_start']<=t['path'].rsplit('/D',1)[1]<=protocol['confirmation_end']]
    if [t['path'] for t in futures]!=protocol['confirmation_tables']:raise ValueError('Frozen confirmation table list differs')
    source_rows=json.loads((source/'source-rows.json').read_text());input_hashes=set();ledger=[];cache_files=[]
    with tables.open_file(dataset,mode='r') as database:
        for path,ids in source_rows.items():
            raw,_=inputs_at(database,path,ids);input_hashes.update(signatures(raw))
        for path in sorted((source/'scores').glob('cal-*.npz')):
            date=path.stem.split('-',1)[1]
            with np.load(path,allow_pickle=False) as archive:ids=archive['indices']
            raw,_=inputs_at(database,'/flows/D'+date,ids);input_hashes.update(signatures(raw))
        for table in futures:
            date=table['path'].rsplit('/D',1)[1]
            # The date seed and row-count cap match development sampling.
            import hashlib
            sampling_seed=int.from_bytes(hashlib.sha256(date.encode()).digest()[:4],'big')
            rng=np.random.default_rng(sampling_seed)
            ids=np.sort(rng.choice(table['rows'],min(protocol['flows_per_day'],table['rows']),replace=False))
            raw,labels=inputs_at(database,table['path'],ids)
            overlap=np.asarray([s in input_hashes for s in signatures(raw)],dtype=bool)
            values=infer(model,transform(raw,checkpoint['preprocessing']).astype(np.float32),device,geometry)
            values.update(labels=labels,known=np.isin(labels,known),background=np.isin(labels,list(background)),indices=ids,codes=known,source_overlap=overlap)
            prediction=known[values['prediction']]
            for variant in ['raw','reference','shift']:
                saved=dict(values)
                if variant!='raw':
                    for score,mapping in maps[variant].items():saved['rqp_'+score]=map_scores(values[score],prediction,mapping)
                path=out/variant/'scores'/('test-'+date+'.npz')
                np.savez_compressed(path,**saved)
                cache_files.append(dict(path=str(path.relative_to(root)),sha256=sha256(path)))
            ledger.append(dict(table=table['path'],rows=len(labels),input_overlap=int(overlap.sum()),known_n=int(values['known'].sum()),heldout_n=int((~values['known'] & ~values['background']).sum()),background_n=int(values['background'].sum()),known_correct=int(np.count_nonzero(values['known'] & (prediction==labels)))))
            print(json.dumps(dict(scored_confirmation_date=date,rows=len(labels))),flush=True)
    # A changed dependency cannot silently yield a valid completion receipt.
    verified_protocol(root)
    report=dict(status='FROZEN_CONFIRMATION_SCORE_CACHE_COMPLETE',model_seed=args.seed,
                protocol_sha256=seal['protocol_sha256'],model_sha256=prepared['model_sha256'],
                dataset_sha256=protocol['dataset_sha256'],confirmation_tables_read=ledger,
                future_labels_used_for_fitting=False,future_label_based_sampling=False,
                reference_maps_fixed=True,source_input_signature_count=len(input_hashes),files=cache_files,
                torch=torch.__version__,numpy=np.__version__,device=torch.cuda.get_device_name(device))
    (out/'score-complete.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in {'files','confirmation_tables_read'}}),flush=True)


if __name__=='__main__':main()
