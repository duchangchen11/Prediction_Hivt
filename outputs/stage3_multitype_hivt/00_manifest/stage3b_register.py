"""Freeze Stage3A and preregister the single-variable Type Embedding experiment."""
import argparse
import shutil
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'00_manifest'))
from stage3b_common import PROJECT,CONFIG,BASE_CONFIG,FREEZE,PREREG,atomic_json,read_json,sha256,git
import yaml

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--requirements',required=True,type=Path);a=parser.parse_args()
    assert git('branch','--show-current')=='stage3b/type-embedding'
    assert git('rev-parse','HEAD')=='baf8b59b44047657dca0a64460dd4005c323af76'
    assert not FREEZE.exists() and not PREREG.exists()
    previous=read_json(ROOT/'00_manifest/stage3a_plusplus_frozen_previous.json');protected=dict(previous['files'])
    for name in git('ls-files',str(ROOT.relative_to(PROJECT))).splitlines():protected[name]=sha256(PROJECT/name)
    for p in list((ROOT/'07_checkpoints').glob('stage3*.pt'))+[ROOT/'04_evaluation/stage3a_final_frozen_actor_errors.csv']:
        if not p.name.startswith('stage3b_'):protected[str(p.relative_to(PROJECT))]=sha256(p)
    assert all(sha256(PROJECT/n)==d for n,d in protected.items())
    frozen=read_json(ROOT/'07_checkpoints/stage3a_final_frozen_checkpoint_manifest.json')
    assert frozen['stage3a_frozen']=='YES' and frozen['Ready_Stage3B']=='YES'
    assert sha256(ROOT/frozen['checkpoint_relative_path'])==frozen['checkpoint_sha256']
    atomic_json(FREEZE,{'base_commit':git('rev-parse','HEAD'),'files':protected,'scene_shards':previous['scene_shards'],
                       'NoType_frozen_checkpoint_sha256':frozen['checkpoint_sha256'],'status':'FROZEN'})
    c=yaml.safe_load(BASE_CONFIG.read_text());assert not c['type_embedding'];c['type_embedding']=True
    CONFIG.write_text(yaml.safe_dump(c,sort_keys=False))
    shutil.copyfile(a.requirements,ROOT/'00_manifest/stage3b_requirements.txt')
    atomic_json(PREREG,{'status':'REGISTERED_BEFORE_TRAINING','seed':2022,'frozen_NoType_commit':git('rev-parse','HEAD'),
        'NoType_checkpoint_sha256':frozen['checkpoint_sha256'],'NoType_actor_errors_sha256':frozen['final_actor_errors_sha256'],
        'NoType_config_sha256':sha256(BASE_CONFIG),'Type_config_sha256':sha256(CONFIG),
        'only_configuration_value_changed':'type_embedding: false -> true',
        'model_change':{'nn_Embedding':[3,64],'additional_parameters':192,'insertion':'LocalEncoder output before GlobalInteractor',
                        'fusion':'local + type_embedding(agent_type)','decoder_receives_augmented_local':True},
        'shared_initialization':'Fresh No-Type seed2022 constructor; name/shape exact copy; no trained weights',
        'formal_maximum_global_step':21000,'formal_fixed_scale_warmup_steps':5000,'formal_NLL_maximum_steps':16000,
        'warmup_to_NLL_transition':'Restore own best warm-up model+AdamW+RNG; lower LR to1e-4; reset NLL sampler cursor, as Stage3A',
        'warmup_selection':'full-horizon official VAL overall minFDE6 strict improvement',
        'final_selection':'post-update original-NLL full-horizon official VAL overall minFDE6 strict improvement',
        'NLL_patience':5,'validation_interval':500,'optimizer_state_preserved_at_phase_transition':True,
        'budget_note':'Execution limits follow this Stage3B request and actual Stage3A 5000+16000 budget; original frozen YAML historical maximum fields are preserved.',
        'tiny_reuses_exact_Stage3A_windows':True,'tiny_minimum_relative_ADE_FDE_regression_decrease':0.5,
        'bootstrap':{'replicates':1000,'seed':2022,'resampling_unit':'paired VAL scene','scenes':150,'confidence':0.95,
                     'weighting':'pool all actor-window deltas within resampled scene clusters'},
        'primary_endpoint':'Overall full-horizon minFDE6; delta=Type-NoType',
        'scientific_decision':{'SUPPORTED':'Overall deltaFDE<0 and its CI95 upper<0',
            'PARTIAL':'Overall CI includes0, and Vehicle or Pedestrian deltaFDE CI95 upper<0, without significant overall harm',
            'NOT_SUPPORTED':'Significant overall harm or no reliable main-group FDE improvement'},
        'auxiliary_Top1_improvement_does_not_change_primary_support_decision':True,
        'bicycle_gt5_inference':'descriptive only; limited unique bicycle instances/scenes; no strong significance claim',
        'Stage3B_PASS_rule':'Integration+tiny pass; unchanged data/hypers; finite completed training and freshVAL; exact pairing; all requested evidence audited, regardless of favorable effect',
        'Ready_Stage4_rule':'Technical evidence complete and TypeEmbedding SUPPORTED/PARTIAL; otherwise NO; no Stage4 execution authorized',
        'class_weights':False,'oversampling':False,'new_map_semantics':False,'dynamic_interaction':False,
        'test_used':False,'Stage4_executed':False})
    exclude=PROJECT/'.git/info/exclude';old=exclude.read_text()
    pattern='/outputs/stage3_multitype_hivt/04_evaluation/stage3b_type_embedding_actor_errors.csv'
    if pattern not in old.splitlines():exclude.write_text(old.rstrip()+'\n'+pattern+'\n')
    print('STAGE3B_REGISTERED',len(protected),'protected files;',len(previous['scene_shards']),'frozen shards')

if __name__=='__main__':main()
