"""Seal the selected protocol and historical evidence before future-row access.

This is an internal, timestamped analysis freeze, not an external registration.
An existing seal is never replaced. Only development and historical artifacts
are read; future date names and row counts come from the metadata inventory.
"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from frozen_protocol import sha256,verified_protocol


def read(path):return json.loads(path.read_text())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);args=parser.parse_args();root=args.root
    target=root/'research/frozen';
    if target.exists():raise FileExistsError('An existing internal freeze cannot be replaced')
    inventory=read(root/'outputs/inventory.json');original=read(root/'outputs/pilot-transformer-v1/prepare-complete.json')
    selection=read(root/'research/verification/aci-selection.json')
    if selection['confirmation_scores_read']:raise ValueError('ACI selection consulted confirmation')
    preflight=read(root/'outputs/confirmation-engine-preflight-v2.json')
    if preflight['status']!='PASS_CONFIRMATION_ENGINE_TEST_ONLY_SIMULATION' or preflight['result_claim_allowed']:
        raise ValueError('Confirmation engineering preflight missing or mislabelled')
    for run in ['pilot-transformer-v1','pilot-transformer-rqp-v1','pilot-transformer-shift-v1']:
        if read(root/'outputs'/run/'audit-complete.json')['status']!='PASS_INDEPENDENT_NUMPY_RECOUNT_AND_FEEDBACK_LEDGER':raise ValueError('Development comparison lacks independent audit')
    models=[];artifacts=[]
    for run in ['pilot-transformer-v1','pilot-transformer-rqp-v1','pilot-transformer-shift-v1']:
        artifacts += [root/'outputs'/run/name for name in ['audit-complete.json','period-summary.csv','period-summary-provenance.json']]
    for run in ['pilot-transformer-v1','pilot-transformer-rqp-v1']:
        artifacts += [root/'outputs'/run/'aci-comparison'/name for name in ['complete.json','summary.csv']]
    for seed in [42,43,44]:
        entry=dict(seed=seed,source_config='configs/transformer-v1.json' if seed==42 else f'configs/transformer-seed{seed}-v1.json',
                   raw_config='configs/transformer-v1.json' if seed==42 else f'configs/transformer-seed{seed}-v1.json',
                   raw_run='outputs/pilot-transformer-v1' if seed==42 else f'outputs/replication-transformer-seed{seed}-v1',
                   reference_config='configs/transformer-rqp-v1.json' if seed==42 else f'configs/transformer-rqp-seed{seed}-v1.json',
                   reference_run='outputs/pilot-transformer-rqp-v1' if seed==42 else f'outputs/historical-rqp-transformer-seed{seed}-v1',
                   shift_config='configs/transformer-shift-full-v1.json' if seed==42 else f'configs/transformer-shift-seed{seed}-v1.json',
                   shift_run=f'outputs/historical-shift-transformer-seed{seed}-v1',confirmation_run=f'outputs/confirmation-transformer-seed{seed}-v1')
        raw=root/entry['raw_run'];prepared=read(raw/'prepare-complete.json')
        if prepared['config_sha256']!=sha256(root/entry['source_config']) or prepared['model_sha256']!=sha256(raw/'model.pt'):raise ValueError('Model/configuration identity differs')
        for field in ['source_rows_sha256','known_codes','calibration_rows']:
            if prepared[field]!=original[field]:raise ValueError('Replicated source/calibration cohort differs')
        if prepared['protected_flow_tables_read'] or any(p.rsplit('/D',1)[1]>='20241001' for p in prepared['flow_tables_read']):raise ValueError('Historical fitting exposed future flow values')
        if prepared['geometry_sha256']!=sha256(raw/'geometry-state.npz'):raise ValueError('Source feature geometry differs')
        for variant in ['raw','reference','shift']:
            folder=root/entry[variant+'_run'];report=read(folder/'prepare-complete.json')
            if report['config_sha256']!=sha256(root/entry[variant+'_config']):raise ValueError('Variant configuration differs')
            artifacts.extend(sorted((folder/'scores').glob('*.npz')));artifacts.append(folder/'prepare-complete.json')
            if variant!='raw':
                audit=read(folder/'normalization-audit.json')
                if audit['status']!='PASS_INDEPENDENT_REFERENCE_MAP_AND_SCORE_RECONSTRUCTION':raise ValueError('Historical map reconstruction missing')
                state=read(folder/'reference-maps.json')
                if state['upstream_model_sha256']!=prepared['model_sha256']:raise ValueError('Historical map uses another model')
                if state['fit_rows']!=14875 or state['guard_rows']!=15125:raise ValueError('Historical roles changed')
                artifacts.extend([folder/'reference-maps.json',folder/'normalization-audit.json'])
        artifacts.extend([raw/'model.pt',raw/'geometry-state.npz',raw/'source-rows.json'])
        if (root/entry['confirmation_run']).exists():raise ValueError('Confirmation output already exists before freeze')
        models.append(entry)
    future=[t['path'] for t in inventory['tables'] if '20241001'<=t['path'].rsplit('/D',1)[1]<='20250531']
    if len(future)!=238:raise ValueError('Confirmation metadata date count differs')
    protocol=dict(status='FROZEN_BEFORE_CONFIRMATION_FLOW_ACCESS',dataset='CESNET-QUICEXT-25-XS',dataset_sha256=inventory['sha256'],
                  confirmation_start='20241001',confirmation_end='20250531',confirmation_tables=future,flows_per_day=6000,warmup_dates=85,
                  source_rows_sha256=original['source_rows_sha256'],known_codes=original['known_codes'],models=models,
                  scores=['negative_energy','one_minus_msp','mahalanobis'],budgets=[32,128,512],delays=[1,7],audit_seeds=[101,102,103],
                  alpha=.05,window_days=28,primary=dict(method='historical_quantile_recentering',score='negative_energy',budget=32,delay=7),
                  variant_methods=dict(raw=['fixed_global','fixed_classwise','rolling_global','rolling_classwise','pooled_classwise'],shift=['rolling_global'],reference=['rolling_global']),
                  aci_gamma=selection['selected'],known_frr_operational_margin=.005,
                  resampling=dict(block_lengths=[7,28,56],primary_block=28,replicates=2000,seed=20261122),
                  analysis_plan='research/confirmation-design-working.md',selection_evidence='research/verification/aci-selection.json',
                  primary_estimands=['mean_daily_known_frr','period_background_macro_detection','period_heldout_service_macro_detection','mean_daily_known_correct_accepted'],
                  claim_boundary='Application of historical conditional-quantile recentering; background-rejection hypothesis from development, with held-out-service detection reported separately. No first rectification/feedback/delayed-conformal claim; no arbitrary-drift per-day guarantee.',
                  stopping_rule='One frozen confirmation; preserve all outcomes and original failures. Repairs require a documented amendment and affected-output audit.')
    # Scientific implementations/configurations and all historical score caches
    # are bound. Runtime receipts and manuscript prose may continue to change.
    scientific_scripts=['prepare.py','normalize_scores.py','audit_reference_maps.py','score_confirmation.py','evaluate_confirmation.py','analyse_confirmation.py','freeze_confirmation.py','evaluate.py','audit.py','evaluate_aci.py','summarize_periods.py','configio.py','verify_confirmation_engine.py']
    artifacts+=[root/'scripts'/name for name in scientific_scripts]+sorted(root.glob('src/*.py'))+sorted(root.glob('configs/*.json'))
    artifacts+=[root/'research/confirmation-design-working.md',root/'research/verification/aci-selection.json',root/'research/verification/development-review.json',root/'outputs/confirmation-engine-preflight-v2.json',root/'outputs/inventory.json']
    files={str(path.relative_to(root)):sha256(path) for path in sorted(set(artifacts))}
    target.mkdir();protocol_path=target/'confirmation-v1.json';protocol_path.write_text(json.dumps(protocol,indent=2)+'\n')
    seal=dict(status='INTERNAL_PROTOCOL_FROZEN_BEFORE_CONFIRMATION',frozen_at_utc=datetime.now(timezone.utc).isoformat(),external_preregistration=False,
              protocol=str(protocol_path.relative_to(root)),protocol_sha256=sha256(protocol_path),files=files,protected_future_flow_tables_read=[])
    (target/'confirmation-v1-seal.json').write_text(json.dumps(seal,indent=2)+'\n');verified_protocol(root)
    print(json.dumps(dict(status=seal['status'],frozen_at_utc=seal['frozen_at_utc'],protocol_sha256=seal['protocol_sha256'],historical_bound_files=len(files),confirmation_dates=len(future))),flush=True)


if __name__=='__main__':main()
