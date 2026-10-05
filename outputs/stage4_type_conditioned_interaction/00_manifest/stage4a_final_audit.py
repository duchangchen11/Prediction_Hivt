"""Read-only protocol, original-artifact preservation and export provenance audit."""
import csv
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'00_manifest'))
STAGE3_ROOT = ROOT.parent / 'stage3_multitype_hivt'
sys.path.insert(0, str(STAGE3_ROOT / '00_manifest'))
sys.path.insert(0, str(STAGE3_ROOT / '04_evaluation'))
sys.path.insert(0, str(ROOT / '00_manifest'))
from stage4a_common import (CONFIG,FREEZE,PREREG,SUMMARY,CURVE,BEST,LAST,GROUPS,
    config,model_new,read_json,atomic_json,sha256,verify_previous,state_digest,PROJECT,git)
import torch
import yaml

def main():
    torch.set_num_threads(4)
    training=read_json(SUMMARY);prereg=read_json(PREREG)
    relocation=read_json(ROOT/'00_manifest/stage4a_relocation_audit.json')
    assert relocation['status']=='PASS' and relocation['training_completed_before_migration']
    assert relocation['final_evaluation_completed_before_migration'] and not relocation['retraining']
    assert relocation['best_checkpoint_SHA256']==training['checkpoint_sha256']
    assert Path(relocation['new_stage4_root'])==ROOT
    assert relocation['registered_training_source_snapshots_verified']
    assert all(relocation['core_function_AST_checks'].values())
    for name,digest in relocation['post_training_path_adapter_SHA256'].items():assert sha256(ROOT/name)==digest
    assert not [p for p in STAGE3_ROOT.rglob('stage4a_*') if p.is_file()]
    architecture=read_json(ROOT/'00_manifest/stage4a_architecture_review.json');assert architecture['status']=='PASS'
    subgroup=read_json(ROOT/'04_evaluation/stage4a_interaction_subgroup_audit.json')
    assert subgroup['registered_before_formal_training'] and subgroup['formal_optimizer_updates_at_registration']==0
    assert subgroup['preregistration_sha256']==sha256(PREREG)
    assert not subgroup['future_or_prediction_used_to_define_membership']
    assert read_json(ROOT/'00_manifest/stage4a_training_sources.json')['preregistration_sha256']==sha256(PREREG)
    assert training['status']=='COMPLETE' and training['warmup_steps']==5000
    assert training['NLL_steps']+5000==training['final_executed_global_step']<=21000
    assert training['NLL_steps']%500==0 and training['from_scratch'] and not training['Stage3B_trained_weights_loaded']
    assert not training['test_used'] and not training['Stage4B_executed']
    assert training['NaN']==training['Inf']==0 and training['TypeEmbedding_gradient_verified']
    assert sha256(CONFIG)==prereg['Stage4A_config_sha256'] and sha256(BEST)==training['checkpoint_sha256']
    old=yaml.safe_load((STAGE3_ROOT/'00_manifest/stage3b_config.yaml').read_text());new=config()
    assert {k:v for k,v in new.items() if k not in prereg['only_configuration_additions']}==old
    assert {k:new[k] for k in prereg['only_configuration_additions']}==prereg['only_configuration_additions']
    assert old['type_embedding'] is True and new['type_embedding'] is True
    for name,digest in read_json(ROOT/'00_manifest/stage4a_training_sources.json')['training_source_sha256'].items():assert sha256(ROOT/'00_manifest/stage4a_training_snapshot'/name)==digest
    with CURVE.open() as f:curve=list(csv.DictReader(f))
    steps=[int(float(r['global_step'])) for r in curve]
    assert steps==list(range(500,training['final_executed_global_step']+1,500))
    best=None;bad=0;best_step=None
    for r in curve:
        step=int(float(r['global_step']));warm=step<=5000
        assert r['phase']==('fixed_scale' if warm else 'original_nll')
        assert float(r['learning_rate'])==(.001 if warm else .0001)
        assert float(r['TypeEmbedding_gradient_norm_min'])>0
        for tag in ('pair','first','final'):
            assert float(r['Relation_'+tag+'_gradient_norm_max'])>0
            assert float(r['Relation_'+tag+'_gradient_norm_min'])>=0
        measured=read_json(ROOT/f'04_evaluation/stage4a_val_step_{step:05d}.json')
        assert len(measured['scenes'])==150 and measured['windows']==3603 and not measured['test_used']
        for group in GROUPS:
            for metric,tag in (('count','count'),('minADE6','ADE'),('minFDE6','FDE'),('MR6','MR'),('Top1ADE6','Top1ADE'),('Top1FDE6','Top1FDE'),('NLL','NLL')):
                assert abs(float(r['VAL_'+group+'_'+tag])-measured['metrics']['full_horizon'][group][metric])<1e-12
        if step==5500:best=None;bad=0
        value=float(r['VAL_overall_FDE']);improved=best is None or value<best
        if improved:best=value;best_step=step;bad=0
        else:bad+=1
        assert int(float(r['improved']))==int(improved) and int(float(r['best_step']))==best_step
        assert float(r['best_overall_FDE'])==best and int(float(r['consecutive_nonimprovements']))==bad
        if not warm and bad>=5:assert step==training['final_executed_global_step'],'Training continued after patience'
    assert best_step==training['best_global_step']
    assert training['converged_by_patience']==(bad>=5)
    assert training['stopped_by_budget']==(training['final_executed_global_step']==21000)
    transition=read_json(ROOT/'00_manifest/stage4a_phase_transition_audit.json')
    assert transition['status']=='PASS' and transition['optimizer_state_preserved'] and transition['model_RNG_restored']
    assert transition['own_warmup_best_source_step']==training['warmup_best_source_step']
    assert transition['NLL_sampler_cursor_reset_like_Stage3A']=={'epoch':0,'next_batch':0}
    checkpoint_audits={}
    for label,path in (('best',BEST),('last',LAST)):
        saved=torch.load(path,map_location='cpu',weights_only=False);meta=saved['metadata'];nll_step=meta['global_step']-5000
        assert meta['phase']=='original_nll' and meta['config_sha256']==sha256(CONFIG)
        assert meta['git_commit_SHA']==training['training_code_git_commit'] and meta['warmup_executed_steps']==5000
        expected_epoch=(nll_step-1)//1057;expected_next=(nll_step-1)%1057+1
        assert saved['iterator']=={'epoch':expected_epoch,'next_batch':expected_next}
        model=model_new(device='cpu');model.load_state_dict(saved['state_dict']);optimizer=model.optimizer(.0001,config()['weight_decay'])
        optimizer.load_state_dict(saved['optimizer_state_dict'])
        embedding_step=int(optimizer.state[model.type_embedding.weight]['step'])
        for param in (model.global_interactor.pair_embedding.weight,model.global_interactor.relation_mlp[0].weight,model.global_interactor.relation_mlp[-1].weight):
            assert int(optimizer.state[param]['step'])==embedding_step
        assert embedding_step==training['warmup_best_source_step']+nll_step
        assert all(torch.isfinite(p).all() for p in model.parameters())
        if label=='best':assert state_digest(saved['state_dict'])==training['model_state_content_sha256']
        checkpoint_audits[label]={'global_step':meta['global_step'],'NLL_steps':nll_step,
            'TypeEmbedding_AdamW_updates':embedding_step,'cursor':saved['iterator'],'sha256':sha256(path)}
        del saved,model,optimizer
    for name in ('stage4a_initialization_audit.json','stage4a_unit_tests.json','stage4a_neutral_initialization_audit.json'):
        assert read_json(ROOT/'00_manifest'/name)['status']=='PASS'
    for name in ('stage4a_tiny_overfit.json','stage4a_gradient_audit.json','stage4a_pairing_audit.json','stage4a_metrics.json','stage4a_bootstrap_ci.json','stage4a_qualitative_case_manifest.json','stage4a_interaction_subgroup_audit.json','stage4a_relation_bias_statistics.json','stage4a_efficiency_audit.json'):
        assert read_json(ROOT/'04_evaluation'/name)['status']=='PASS'
    assert read_json(ROOT/'04_evaluation/stage4a_pairing_audit.json')['paired_full_horizon_actors']==54990
    actual_sha=sha256(ROOT/'04_evaluation/stage4a_actor_errors.csv')
    base_sha=sha256(STAGE3_ROOT/'04_evaluation/stage3b_type_embedding_actor_errors.csv')
    member_sha=sha256(ROOT/subgroup['membership_relative_path'])
    assert member_sha==subgroup['membership_sha256']
    fresh=read_json(ROOT/'04_evaluation/stage4a_metrics.json')
    assert fresh['fresh_complete_official_VAL'] and fresh['NaN']==fresh['Inf']==0
    assert fresh['actor_errors_sha256']==actual_sha and fresh['checkpoint_sha256']==training['checkpoint_sha256']
    reconciliation=read_json(ROOT/'04_evaluation/stage4a_fresh_val_reconciliation.json')
    assert reconciliation['status']=='PASS' and reconciliation['checkpoint_sha256']==training['checkpoint_sha256']
    assert len(reconciliation['checks'])==len(GROUPS)*6 and all(r['passed'] for r in reconciliation['checks'])
    for group in GROUPS:
        for metric in ('minADE6','minFDE6','MR6','Top1ADE6','Top1FDE6','NLL'):
            tolerance=1e-4 if metric in ('Top1ADE6','Top1FDE6') else 1e-6
            assert abs(fresh['metrics']['full_horizon'][group][metric]-training['selected_full_horizon_metrics'][group][metric])<tolerance
    pair=read_json(ROOT/'04_evaluation/stage4a_pairing_audit.json')
    ci=read_json(ROOT/'04_evaluation/stage4a_bootstrap_ci.json')
    for artifact in (pair,ci):
        assert artifact['B_actor_errors_sha256']==base_sha and artifact['C_actor_errors_sha256']==actual_sha
        assert artifact['interaction_membership_sha256']==member_sha
    assert pair['paired_partial_future_actors']==30037 and pair['paired_total_actor_windows']==85027
    assert ci['replicates']==1000 and ci['seed']==2022 and ci['scene_count']==150
    efficiency=read_json(ROOT/'04_evaluation/stage4a_efficiency_audit.json')
    assert efficiency['Stage4A_checkpoint_sha256']==training['checkpoint_sha256']
    assert efficiency['Stage3B_checkpoint_sha256']==sha256(STAGE3_ROOT/'07_checkpoints/stage3b_best_overall_minfde.pt')
    assert efficiency['paired_measured_batches']>=500 and efficiency['eval_mode'] and efficiency['identical_input_tensors_per_pair']
    assert efficiency['additional_parameters']==1608 and efficiency['parameters_C']==647609
    assert efficiency['batch_timing_source_sha256']==sha256(ROOT/'06_tables/stage4a_efficiency_batch_timings.csv')
    bias=read_json(ROOT/'04_evaluation/stage4a_relation_bias_statistics.json')
    assert bias['checkpoint_sha256']==training['checkpoint_sha256'] and not bias['causal_interpretation']
    assert bias['source_csv_sha256']==sha256(ROOT/'06_tables/stage4a_relation_bias_statistics.csv')
    assert bias['aggregate_csv_sha256']==sha256(ROOT/'06_tables/stage4a_relation_bias_pair_aggregate.csv')
    cases=read_json(ROOT/'04_evaluation/stage4a_qualitative_case_manifest.json')
    assert cases['checkpoint_SHA256']=={'Stage3B':efficiency['Stage3B_checkpoint_sha256'],'Stage4A':training['checkpoint_sha256']}
    assert cases['actor_CSV_SHA256']=={'Stage3B':base_sha,'Stage4A':actual_sha,'membership':member_sha}
    if training['converged_by_patience']:assert training['stop_reason']=='patience_5'
    else:assert training['stopped_by_budget'] and training['stop_reason']=='global21000_budget'

    exports=[]
    for directory in ('05_figures','03_type_interaction'):
        for path in sorted((ROOT/directory).glob('stage4a_*_audit.json')):
            audit=read_json(path);assert audit['status']=='PASS'
            if 'source_json' in audit:assert sha256(ROOT/audit['source_json'])==audit['source_sha256']
            for name,digest in audit.get('source_SHA256',{}).items():assert sha256(ROOT/name)==digest
            for name,digest in audit['exports_SHA256'].items():assert sha256(ROOT/name)==digest
            exports.append(str(path.relative_to(ROOT)))
    assert len(exports)>=9
    scope=read_json(ROOT/'00_manifest/stage4a_scope_preservation.json')
    assert len(scope['untouched_Stage2C_untracked_files'])==5
    for name,digest in scope['untouched_Stage2C_untracked_files'].items():assert sha256(PROJECT/name)==digest
    assert git('rev-parse','stage3b/type-embedding')==prereg['frozen_Stage3B_commit']
    assert git('branch','--show-current')=='stage4a/type-conditioned-interaction'
    frozen=verify_previous(shards=True)
    output={'status':'PASS','training_and_validation_protocol':'PASS','same_seed_shared_initialization':'PASS',
        'config_only_five_relation_bias_keys_added':True,'training_code_unchanged_since_preregistration':True,
        'training_code_git_commit':training['training_code_git_commit'],'all_actual500step_VAL_points_verified':len(curve),
        'separate_Stage4_root':str(ROOT),'Stage3_has_no_Stage4_files':True,
        'post_training_directory_migration':'PASS','registered_training_source_snapshots_verified':True,
        'TypeEmbedding_gradient_min':min(float(r['TypeEmbedding_gradient_norm_min']) for r in curve),
        'relation_adapter_first10_gradient_audit':'PASS',
        'relation_gradient_maxima':{tag:max(float(r['Relation_'+tag+'_gradient_norm_max']) for r in curve) for tag in ('pair','first','final')},
        'checkpoint_optimizer_and_cursor':checkpoint_audits,'frozen_prior_files_verified':len(frozen['files']),
        'frozen_scene_shards_verified':len(frozen['scene_shards']),'old_results_unchanged':True,
        'Stage2C_untracked_redraw_files_unchanged':5,'frozen_Stage3B_branch_unchanged':True,
        'paired_full_horizon_actor_windows':54990,'figure_audits_verified':exports,
        'test_used':False,'Stage4B_executed':False,'scientific_support_is_separate_from_technical_PASS':True}
    atomic_json(ROOT/'00_manifest/stage4a_final_protocol_audit.json',output)
    print('STAGE4A_FINAL_PROTOCOL_AUDIT=PASS',len(curve),'VAL points;',len(exports),'figure audits',flush=True)

if __name__=='__main__':main()
