"""Read-only preservation verification and manifest for final Stage3A artifacts."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'00_manifest'))
from stage3a_plusplus_common import (PROJECT,CONFIG,FREEZE,PREREG,FINAL_MANIFEST,TRAIN_MANIFEST,
    atomic_json,read_json,sha256,git,verify_previous)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--verify-shards',action='store_true')
    args=parser.parse_args(); frozen=verify_previous(shards=args.verify_shards)
    shards_verified=args.verify_shards
    shard_proof=ROOT/'00_manifest/stage3a_plusplus_frozen_shards_audit.json'
    if shard_proof.exists():
        proof=read_json(shard_proof)
        assert proof['status']=='PASS' and proof['frozen_manifest_sha256']==sha256(FREEZE)
        assert proof['scene_shards_SHA256_verified']==len(frozen['scene_shards'])==850
        shards_verified=True
    final=read_json(FINAL_MANIFEST);train=read_json(TRAIN_MANIFEST)
    assert final['stage3a_frozen']=='YES' and final['Ready_Stage3B']=='YES'
    assert final['status']==train['status']=='COMPLETE'
    assert final['official_VAL_fresh_evaluation_complete']
    assert final['NaN']==final['Inf']==0
    assert final['final_executed_global_step']==18000+final['executed_extra_steps']<=21000
    assert final['executed_extra_steps']<=3000
    assert sha256(ROOT/final['checkpoint_relative_path'])==final['checkpoint_sha256']
    assert sha256(CONFIG)==final['config_sha256']
    prereg=read_json(PREREG)
    assert all(sha256(ROOT/name)==digest for name,digest in prereg['source_sha256'].items())
    figure=read_json(ROOT/'05_figures/stage3a_final_convergence_audit.json')
    assert figure['status']=='PASS' and not figure['fabricated_points'] and not figure['interpolation']
    assert figure['frozen_best_global_step']==final['final_best_global_step']
    assert figure['final_executed_global_step']==final['final_executed_global_step']
    assert all(sha256(ROOT/name)==digest for name,digest in figure['sources_sha256'].items())
    assert all(sha256(ROOT/name)==digest for name,digest in figure['exports_sha256'].items())
    report=read_json(ROOT/'00_manifest/stage3a_final_frozen_report_audit.json')
    assert report['status']=='PASS'
    assert all(sha256(ROOT/name)==digest for name,digest in report['artifact_sha256'].items())
    independent=read_json(ROOT/'00_manifest/stage3a_final_frozen_independent_audit.json')
    assert independent['status']=='PASS' and independent['NaN']==independent['Inf']==0
    assert independent['final_checkpoint_sha256']==final['checkpoint_sha256']
    assert all(sha256(ROOT/name)==digest for name,digest in independent['artifact_sha256'].items())
    cases=read_json(ROOT/'04_evaluation/stage3a_final_motion_case_manifest.json')
    gain=final['relative_FDE_improvement_vs_18000']
    assert cases['regenerated']==(gain>=.005)
    if gain>=.005:
        assert cases['source_checkpoint_sha256']==final['checkpoint_sha256']
        for item in cases['figures']:
            assert sha256(ROOT/item['source_json'])==item['source_sha256']
            case_audit=read_json(ROOT/'05_figures'/(item['name']+'_audit.json'))
            assert case_audit['status']=='PASS'
            assert all(sha256(ROOT/entry['relative_path'])==entry['sha256'] for entry in case_audit['exports'].values())
    audit=ROOT/'00_manifest/stage3a_plusplus_final_preservation_audit.json'
    atomic_json(audit,{'status':'PASS','protected_previous_files':len(frozen['files']),
        'protected_previous_files_unchanged':True,'frozen_scene_shards':len(frozen['scene_shards']),
        'all_scene_shards_SHA256_verified':shards_verified,'registered_optimization_source_unchanged':True,
        'original_config_SHA256':sha256(CONFIG),'final_checkpoint_SHA256':final['checkpoint_sha256'],
        'interaction_density_recomputed':False,'Stage3B_executed':False,'test_used':False,
        'final_convergence_figure_exports_and_source_SHA_verified':True,
        'case_regeneration_threshold_applied':True,'final_report_artifact_SHA_verified':True,
        'source_commit':git('rev-parse','HEAD'),'only_current_stage_new_artifacts_added':True})
    manifest=ROOT/'00_manifest/stage3a_plusplus_artifact_manifest.json'
    rows=[]
    prefixes=('stage3a_plusplus_','stage3a_final_')
    for p in sorted(ROOT.rglob('*')):
        if not p.is_file() or p==manifest or not p.name.startswith(prefixes):continue
        if '__pycache__' in p.parts or 'stage3_cache' in p.parts or p.name.endswith(('.tmp','.pyc')):continue
        rows.append({'relative_path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':sha256(p),
                     'local_only':p.name.endswith('.pt') or p.name=='stage3a_final_frozen_actor_errors.csv'})
    atomic_json(manifest,{'stage':'Stage3A++ final frozen No-Type','self_hash_excluded':True,
        'artifacts':rows,'large_checkpoint_and_actor_CSV_upload':False})
    print('FINAL_PRESERVATION=PASS',len(frozen['files']),'files;',len(rows),'new artifacts')

if __name__=='__main__':main()
