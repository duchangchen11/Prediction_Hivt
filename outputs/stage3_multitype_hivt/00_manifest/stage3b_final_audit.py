"""Read-only protocol, original-artifact preservation and export provenance audit."""
import csv
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'00_manifest'))
from stage3b_common import (CONFIG,FREEZE,PREREG,SUMMARY,CURVE,BEST,LAST,GROUPS,
    config,model_new,read_json,atomic_json,sha256,verify_previous,state_digest,PROJECT,git)
import torch
import yaml

def main():
    torch.set_num_threads(4)
    training=read_json(SUMMARY);prereg=read_json(PREREG)
    assert training['status']=='COMPLETE' and training['warmup_steps']==5000
    assert training['NLL_steps']+5000==training['final_executed_global_step']<=21000
    assert training['NLL_steps']%500==0 and training['from_scratch'] and not training['NoType_trained_weights_loaded']
    assert not training['test_used'] and not training['Stage4_executed']
    assert training['NaN']==training['Inf']==0 and training['TypeEmbedding_gradient_verified']
    assert sha256(CONFIG)==prereg['Type_config_sha256'] and sha256(BEST)==training['checkpoint_sha256']
    old=yaml.safe_load((ROOT/'00_manifest/stage3_config.yaml').read_text());new=config()
    assert {k:v for k,v in old.items() if k!='type_embedding'}=={k:v for k,v in new.items() if k!='type_embedding'}
    assert old['type_embedding'] is False and new['type_embedding'] is True
    for name,digest in prereg['training_source_sha256'].items():assert sha256(ROOT/name)==digest
    with CURVE.open() as f:curve=list(csv.DictReader(f))
    steps=[int(float(r['global_step'])) for r in curve]
    assert steps==list(range(500,training['final_executed_global_step']+1,500))
    best=None;bad=0;best_step=None
    for r in curve:
        step=int(float(r['global_step']));warm=step<=5000
        assert r['phase']==('fixed_scale' if warm else 'original_nll')
        assert float(r['learning_rate'])==(.001 if warm else .0001)
        assert float(r['TypeEmbedding_gradient_norm_min'])>0
        measured=read_json(ROOT/f'04_evaluation/stage3b_val_step_{step:05d}.json')
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
    transition=read_json(ROOT/'00_manifest/stage3b_phase_transition_audit.json')
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
        assert embedding_step==training['warmup_best_source_step']+nll_step
        assert all(torch.isfinite(p).all() for p in model.parameters())
        if label=='best':assert state_digest(saved['state_dict'])==training['model_state_content_sha256']
        checkpoint_audits[label]={'global_step':meta['global_step'],'NLL_steps':nll_step,
            'TypeEmbedding_AdamW_updates':embedding_step,'cursor':saved['iterator'],'sha256':sha256(path)}
        del saved,model,optimizer
    for name in ('stage3b_initialization_audit.json','stage3b_unit_tests.json'):
        assert read_json(ROOT/'00_manifest'/name)['status']=='PASS'
    for name in ('stage3b_tiny_overfit.json','stage3b_pairing_audit.json','stage3b_type_embedding_metrics.json','stage3b_bootstrap_ci.json','stage3b_qualitative_case_manifest.json'):
        assert read_json(ROOT/'04_evaluation'/name)['status']=='PASS'
    assert read_json(ROOT/'04_evaluation/stage3b_pairing_audit.json')['paired_full_horizon_actors']==54990
    exports=[]
    for directory in ('05_figures','03_type_embedding'):
        for path in sorted((ROOT/directory).glob('stage3b_*_audit.json')):
            audit=read_json(path);assert audit['status']=='PASS'
            if 'source_json' in audit:assert sha256(ROOT/audit['source_json'])==audit['source_sha256']
            for name,digest in audit.get('source_SHA256',{}).items():assert sha256(ROOT/name)==digest
            for name,digest in audit['exports_SHA256'].items():assert sha256(ROOT/name)==digest
            exports.append(str(path.relative_to(ROOT)))
    assert len(exports)>=10
    scope=read_json(ROOT/'00_manifest/stage3b_scope_preservation.json')
    assert len(scope['untouched_Stage2C_untracked_files'])==5
    for name,digest in scope['untouched_Stage2C_untracked_files'].items():assert sha256(PROJECT/name)==digest
    assert git('rev-parse','stage3/multitype-hivt')==prereg['frozen_NoType_commit']
    assert git('branch','--show-current')=='stage3b/type-embedding'
    frozen=verify_previous(shards=True)
    output={'status':'PASS','training_and_validation_protocol':'PASS','same_seed_shared_initialization':'PASS',
        'config_only_type_embedding_changed':True,'training_code_unchanged_since_preregistration':True,
        'training_code_git_commit':training['training_code_git_commit'],'all_actual500step_VAL_points_verified':len(curve),
        'TypeEmbedding_gradient_min':min(float(r['TypeEmbedding_gradient_norm_min']) for r in curve),
        'checkpoint_optimizer_and_cursor':checkpoint_audits,'frozen_prior_files_verified':len(frozen['files']),
        'frozen_scene_shards_verified':len(frozen['scene_shards']),'old_results_unchanged':True,
        'Stage2C_untracked_redraw_files_unchanged':5,'frozen_NoType_branch_unchanged':True,
        'paired_full_horizon_actor_windows':54990,'figure_audits_verified':exports,
        'test_used':False,'Stage4_executed':False,'scientific_support_is_separate_from_technical_PASS':True}
    atomic_json(ROOT/'00_manifest/stage3b_final_protocol_audit.json',output)
    print('STAGE3B_FINAL_PROTOCOL_AUDIT=PASS',len(curve),'VAL points;',len(exports),'figure audits',flush=True)

if __name__=='__main__':main()
