"""Freeze Stage3B and register Stage4A before any optimizer updates."""
import argparse
import shutil
from pathlib import Path
import sys
import yaml
ROOT=Path(__file__).resolve().parents[1]
STAGE3_ROOT = ROOT.parent / 'stage3_multitype_hivt'
sys.path.insert(0, str(STAGE3_ROOT / '00_manifest'))
sys.path.insert(0, str(STAGE3_ROOT / '04_evaluation'))
sys.path.insert(0, str(ROOT / '00_manifest'))
sys.path.insert(0,str(ROOT/'00_manifest'))
from stage3b_common import PROJECT,atomic_json,read_json,sha256,git,verify_previous

BASE='824c87d7c1bdb89ef1b79572f463bf864a4b840a'
CKPT_SHA='461dd9fc8ccc4a03bb72e6a92f3e328e34e1fefb6ebf890ff87eedffaa718547'
ADDED={'type_conditioned_interaction':True,'pair_embedding_dim':16,
       'relation_bias_hidden_dim':32,'directed_type_pair':True,'relation_bias_zero_init':True}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--requirements',required=True,type=Path)
    a=parser.parse_args()
    assert git('branch','--show-current')=='stage4a/type-conditioned-interaction'
    assert git('rev-parse','HEAD')==BASE
    freeze=ROOT/'00_manifest/stage4a_frozen_stage3b.json'
    prereg=ROOT/'00_manifest/stage4a_preregistration.json'
    assert not freeze.exists() and not prereg.exists()
    previous=verify_previous(shards=True)
    old_artifacts=read_json(STAGE3_ROOT/'00_manifest/stage3b_artifact_manifest.json')
    for item in old_artifacts['artifacts']:
        assert sha256(STAGE3_ROOT/item['relative_path'])==item['sha256'],item['relative_path']
    protected=dict(previous['files'])
    for name in git('ls-files').splitlines():protected[name]=sha256(PROJECT/name)
    for item in old_artifacts['artifacts']:
        p=STAGE3_ROOT/item['relative_path'];protected[str(p.relative_to(PROJECT))]=sha256(p)
    checkpoint=STAGE3_ROOT/'07_checkpoints/stage3b_best_overall_minfde.pt'
    assert sha256(checkpoint)==CKPT_SHA
    decision=read_json(STAGE3_ROOT/'04_evaluation/stage3b_scientific_decision.json')
    assert decision['Stage3B']=='PASS' and decision['Ready_Stage4']=='YES'
    assert len(previous['scene_shards'])==850
    explicit={name:sha256(STAGE3_ROOT/name) for name in (
        '00_manifest/stage3b_model.py','00_manifest/stage3b_common.py',
        '00_manifest/stage3b_config.yaml','03_type_embedding/stage3b_train.py',
        '09_reports/stage3b_final_report.md','06_tables/stage3b_type_embedding_main_results.csv',
        '04_evaluation/stage3b_bootstrap_ci.json')}
    runtime='models/hivt_runtime/global_interactor.py'
    atomic_json(freeze,{'status':'FROZEN','Stage3B_frozen':'YES','base_commit':BASE,
        'Stage3B_commit':BASE,'Stage3B_checkpoint_path':str(checkpoint.relative_to(PROJECT)),
        'Stage3B_checkpoint_SHA256':CKPT_SHA,'Stage3B_explicit_artifacts_SHA256':explicit,
        'runtime_global_interactor_SHA256':sha256(PROJECT/runtime),
        'files':protected,'scene_shards':previous['scene_shards'],'scene_shard_count':850})
    c=yaml.safe_load((STAGE3_ROOT/'00_manifest/stage3b_config.yaml').read_text())
    assert c['type_embedding'] and not set(ADDED).intersection(c)
    c.update(ADDED);config_path=ROOT/'00_manifest/stage4a_config.yaml'
    config_path.write_text(yaml.safe_dump(c,sort_keys=False))
    shutil.copyfile(a.requirements,ROOT/'00_manifest/stage4a_requirements.txt')
    pairs={str(3*target+source):f'{t} <- {s}'
           for target,t in enumerate(('V','P','B')) for source,s in enumerate(('V','P','B'))}
    atomic_json(ROOT/'00_manifest/stage4a_type_pair_mapping.json',{
        'status':'REGISTERED','edge_index_0':'source j','edge_index_1':'target i',
        'pair_id':'3 * target_type + source_type','mapping':pairs,'directed':True})
    atomic_json(prereg,{'status':'REGISTERED_BEFORE_TRAINING','seed':2022,
        'frozen_Stage3B_commit':BASE,'Stage3B_checkpoint_SHA256':CKPT_SHA,
        'Stage3B_actor_errors_sha256':sha256(STAGE3_ROOT/'04_evaluation/stage3b_type_embedding_actor_errors.csv'),
        'Stage3B_config_sha256':sha256(STAGE3_ROOT/'00_manifest/stage3b_config.yaml'),
        'Stage4A_config_sha256':sha256(config_path),'only_configuration_additions':ADDED,
        'model_change':{'pair_embedding':[9,16],'relation_feature_dim':20,
            'relative_position':'source minus target, transformed by existing target rotation, divided by50m',
            'relative_heading':'source angle minus target angle, cos/sin',
            'MLP':[20,32,24],'layers':3,'heads':8,'reshape':['E',3,8],
            'insertion':'add scalar relation bias to original attention score immediately before softmax',
            'zero_final_weight_and_bias':True,'new_parameters_expected':1608,
            'original_values_rel_embed_topology_and_decoder_unchanged':True},
        'shared_initialization':'canonical frozen Stage3B seed2022 step0; exact shared name/shape copy; no trained weights',
        'formal_fixed_scale_warmup_steps':5000,'formal_NLL_maximum_steps':16000,
        'formal_maximum_global_step':21000,'validation_interval':500,'NLL_patience':5,
        'warmup_to_NLL_transition':'restore own warmup best model+AdamW+RNG, only LR0.001->0.0001; reset NLL sampler cursor as Stage3B',
        'warmup_selection':'full-horizon official VAL overall minFDE6 strict improvement',
        'final_selection':'post-update original NLL full-horizon official VAL overall minFDE6 strict improvement',
        'budget_note':'exact Stage3B actual registered5000+16000 execution; copied YAML historical budget fields preserved unchanged',
        'tiny_hyperparameter_selection':False,'hyperparameter_search':False,'extra_training_seeds':False,
        'interaction_subgroup':{'name':'Heterogeneous-20m','radius_m':20.0,
            'definition':'t0-visible target has >=1 different-type t0-visible vehicle/pedestrian/bicycle context actor within Euclidean20m inclusive',
            'VP_context':'vehicle target with pedestrian neighbor OR pedestrian target with vehicle neighbor',
            'exclude_self':True,'ego_node':False,'future_used_for_membership':False,
            'prediction_used_for_membership':False,'registered_before_formal_training':True},
        'bootstrap':{'replicates':1000,'seed':2022,'resampling_unit':'paired official VAL scene',
            'scenes':150,'confidence':0.95,'weighting':'pool actor-window deltas in resampled clusters'},
        'primary_endpoint':'overall full-horizon minFDE6; delta=Stage4A-Stage3B',
        'major_class_degradation_guard':{'classes':['vehicle','pedestrian','bicycle'],
            'abnormal':'any class minFDE6 paired scene-bootstrap CI95 lower>0',
            'rule':'SUPPORTED and PARTIAL require no reliable major-class degradation; otherwise NOT SUPPORTED and ReadyNO',
            'Bicycle_note':'small samples; used only as conservative harm guard, no strong improvement claim'},
        'scientific_decision':{
            'SUPPORTED':'overall deltaFDE<0 and CI95 upper<0 and no major-class harm guard',
            'PARTIAL':'overall CI contains0 without reliable overall harm; >=1 registered interaction-relevant group deltaFDE<0 and CI95 upper<0; no major-class harm guard',
            'interaction_relevant_groups':['Heterogeneous-20m','VP-context-20m','Pedestrian >5m','Vehicle >5m'],
            'NOT_SUPPORTED':'reliable overall/major-class degradation, or neither primary nor interaction-relevant reliable improvement'},
        'technical_PASS_separate_from_effect_direction':True,
        'Ready_Stage4B_rule':'technicalPASS and SUPPORTED/PARTIAL and no reliable overall/major-class harm',
        'relation_bias_diagnostics':'model internal statistics; no causal effect or physical importance claim',
        'class_balanced_loss':False,'oversampling':False,'data_reprocessed':False,'test_used':False,
        'Reliability_Head':False,'Intent_Head':False,'dynamic_graph_reconstruction':False,'Stage4B_executed':False})
    old_scope=read_json(STAGE3_ROOT/'00_manifest/stage3b_scope_preservation.json')
    atomic_json(ROOT/'00_manifest/stage4a_scope_preservation.json',{
        'status':'REGISTERED_FOR_FINAL_CHECK','untouched_Stage2C_untracked_files':old_scope['untouched_Stage2C_untracked_files'],
        'baseline_branch':'stage3b/type-embedding','baseline_commit':BASE,
        'current_branch':'stage4a/type-conditioned-interaction','Stage4B_executed':False})
    exclude=PROJECT/'.git/info/exclude';text=exclude.read_text()
    pattern='/outputs/stage4_type_conditioned_interaction/04_evaluation/stage4a_actor_errors.csv'
    if pattern not in text.splitlines():exclude.write_text(text.rstrip()+'\n'+pattern+'\n')
    print('STAGE4A_REGISTERED',len(protected),'protected files;',len(previous['scene_shards']),'verified frozen shards',flush=True)

if __name__=='__main__':main()
