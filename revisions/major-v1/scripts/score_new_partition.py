"""Score the sole predeclared supplementary service split using fitted models.

This is post-confirmation supplementary evidence. It is not a new untouched
confirmation claim and does not modify the original frozen protocol/results.
"""
import argparse,hashlib,json,os,sys
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import tables
import torch
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from frozen_protocol import sha256
from reference_quantiles import transform as map_scores
from model_adapters import build_model
from cesnet_adapter import transform
from prepare import inputs_at,signatures,infer

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--seed',type=int,required=True);a=parser.parse_args();root=a.root
 plan_path=root/'revisions/major-v1/new-partition-plan.json';plan=json.loads(plan_path.read_text());entry=next(e for e in plan['models'] if e['seed']==a.seed);original=json.loads((root/'research/frozen/confirmation-v1.json').read_text())
 source=root/entry['raw_run'];prepared=json.loads((source/'prepare-complete.json').read_text());config=json.loads((root/entry['raw_config']).read_text());assert prepared['known_codes']==plan['known_codes'] and prepared['training_rows']==plan['expected_source_rows'];assert not prepared['protected_flow_tables_read'];assert config['class_selection_seed']==plan['class_selection_seed']
 dataset=root/'data/CESNET-QUICEXT-25-XS.h5';assert sha256(dataset)==plan['dataset_sha256'];inventory=json.loads((root/'outputs/inventory.json').read_text());out=root/entry['confirmation_run'];out.mkdir(exist_ok=False)
 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');assert device.type=='cuda'
 torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 checkpoint=torch.load(source/'model.pt',map_location='cpu',weights_only=False);known=np.array(plan['known_codes']);model=build_model(len(known),config).to(device);model.load_state_dict(checkpoint['state_dict']);model.eval();assert list(checkpoint['known_codes'])==known.tolist()
 with np.load(source/'geometry-state.npz',allow_pickle=False) as f:geometry={k:f[k] for k in f.files}
 maps={};historical=[]
 for variant in ['shift','reference']:
  folder=root/entry[variant+'_run'];state=json.loads((folder/'reference-maps.json').read_text());assert state['upstream_model_sha256']==prepared['model_sha256'];assert state['fit_rows']+state['guard_rows']==30000
  maps[variant]=state['maps'];historical.append(dict(variant=variant,H_rows=state['fit_rows'],C_rows=state['guard_rows'],map_sha256=sha256(folder/'reference-maps.json')))
 for variant in ['raw','shift','reference']:(out/variant/'scores').mkdir(parents=True)
 background={int(v) for k,v in inventory['apps'].items() if k.endswith('-background')};tables_list=[t for t in inventory['tables'] if original['confirmation_start']<=t['path'].rsplit('/D',1)[1]<=original['confirmation_end']];assert [t['path'] for t in tables_list]==original['confirmation_tables']
 input_hashes=set();source_rows=json.loads((source/'source-rows.json').read_text());ledger=[];files=[]
 with tables.open_file(dataset,mode='r') as db:
  for path,ids in source_rows.items():raw,_=inputs_at(db,path,ids);input_hashes.update(signatures(raw))
  for path in sorted((source/'scores').glob('cal-*.npz')):
   date=path.stem.split('-',1)[1]
   with np.load(path,allow_pickle=False) as f:ids=f['indices']
   raw,_=inputs_at(db,'/flows/D'+date,ids);input_hashes.update(signatures(raw))
  for table in tables_list:
   date=table['path'].rsplit('/D',1)[1];rng=np.random.default_rng(int.from_bytes(hashlib.sha256(date.encode()).digest()[:4],'big'));ids=np.sort(rng.choice(table['rows'],min(original['flows_per_day'],table['rows']),replace=False));raw,labels=inputs_at(db,table['path'],ids);overlap=np.array([v in input_hashes for v in signatures(raw)])
   values=infer(model,transform(raw,checkpoint['preprocessing']).astype(np.float32),device,geometry);values.update(labels=labels,known=np.isin(labels,known),background=np.isin(labels,list(background)),indices=ids,codes=known,source_overlap=overlap);pred=known[values['prediction']]
   for variant in ['raw','shift','reference']:
    saved=dict(values)
    if variant!='raw':
     for score,mapping in maps[variant].items():saved['rqp_'+score]=map_scores(values[score],pred,mapping)
    path=out/variant/'scores'/('test-'+date+'.npz');np.savez_compressed(path,**saved);files.append(dict(path=str(path.relative_to(root)),sha256=sha256(path)))
   ledger.append(dict(date=date,rows=len(labels),known_n=int(values['known'].sum()),heldout_n=int((~values['known']&~values['background']).sum()),background_n=int(values['background'].sum()),indices_sha256=hashlib.sha256(ids.astype('<i8').tobytes()).hexdigest(),input_overlap=int(overlap.sum())))
   if len(ledger)%30==0:print(json.dumps(dict(posthoc_split_scored_dates=len(ledger),seed=a.seed)),flush=True)
 report={'status':'POST_CONFIRMATION_NEW_PARTITION_SCORE_CACHE_COMPLETE','model_seed':a.seed,'class_selection_seed':plan['class_selection_seed'],'known_codes':known.tolist(),'plan_sha256':sha256(plan_path),'runner_sha256':sha256(__file__),'source_model_sha256':prepared['model_sha256'],'data_sha256':plan['dataset_sha256'],'historical_maps':historical,'files':files,'confirmation_dates':ledger,'future_labels_used_for_model_fitting':False,'independent_network_validation':False,'torch':torch.__version__,'numpy':np.__version__,'device':torch.cuda.get_device_name(device)}
 (out/'score-complete.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ['files','confirmation_dates']}),flush=True)
if __name__=='__main__':main()
