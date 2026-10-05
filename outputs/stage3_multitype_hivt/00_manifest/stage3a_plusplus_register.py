"""Register the final 18000→21000 budget without changing prior experiments."""
from pathlib import Path
import shutil
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'00_manifest'))
from stage3a_plusplus_common import (PROJECT,CONFIG,SOURCE,FREEZE,PREREG,atomic_json,read_json,sha256,git)

def main():
    assert not FREEZE.exists() and not PREREG.exists(), 'Registration already exists'
    assert git('branch','--show-current') == 'stage3/multitype-hivt'
    original=read_json(ROOT/'00_manifest/stage3a_plus_frozen_stage3a.json')
    protected=dict(original['files'])
    for name in git('ls-files',str(ROOT.relative_to(PROJECT))).splitlines():
        protected[name]=sha256(PROJECT/name)
    for p in (SOURCE,ROOT/'07_checkpoints/stage3a_plus_last_checkpoint.pt',
              ROOT/'02_preprocessed/stage3_cache/stage3a_plus_best_actor_errors.csv'):
        protected[str(p.relative_to(PROJECT))]=sha256(p)
    assert all(sha256(PROJECT/name)==digest for name,digest in protected.items())
    atomic_json(FREEZE,{'status':'FROZEN_BEFORE_FINAL_EXTENSION','base_commit':git('rev-parse','HEAD'),
        'files':protected,'scene_shards':original['scene_shards'],
        'interaction_density_recalculated':False,'purpose':'Read-only preservation of all previous experiments'})
    files=['00_manifest/stage3a_plusplus_common.py','00_manifest/stage3a_plusplus_register.py',
           '03_no_type_baseline/stage3a_plusplus_extend_nll.py']
    atomic_json(PREREG,{'status':'REGISTERED_BEFORE_TRAINING','source_checkpoint_sha256':sha256(SOURCE),
        'source_global_step':18000,'source_overall_minFDE6':1.382144824708836,
        'config_sha256':sha256(CONFIG),'source_sha256':{name:sha256(ROOT/name) for name in files},
        'maximum_extra_optimizer_steps':3000,'maximum_global_step':21000,'validation_interval':500,
        'patience':5,'checkpoint_selection':'full-horizon official VAL overall minFDE6; strict improvement',
        'restore_model_optimizer_CPU_CUDA_RNG_sampler_cursor':True,
        'LR_batch_loss_data_architecture_changed':False,'new_warmup':False,
        'freeze_rule':'Freeze YES on patience stop or hard global21000 budget; do not extend again',
        'convergence_rule':'converged_by_patience requires five consecutive nonimprovements',
        'readiness_rule':'finite valid checkpoint and completed fresh official VAL; existing vehicle overall/moving FDE ratio<=1.25',
        'optional_case_relative_FDE_gain_threshold':0.005,'interaction_density_recalculated':False,
        'test_used':False,'Stage3B_executed':False})
    shutil.copyfile('/home/lrj/.codex/attachments/ba5f558b-ba66-4a4b-9788-dbfda71b4169/已粘贴的文本.txt',
                    ROOT/'00_manifest/stage3a_plusplus_requirements.txt')
    exclude=PROJECT/'.git/info/exclude'; pattern='/outputs/stage3_multitype_hivt/04_evaluation/stage3a_final_frozen_actor_errors.csv'
    old=exclude.read_text() if exclude.exists() else ''
    if pattern not in old.splitlines(): exclude.write_text(old.rstrip()+'\n'+pattern+'\n')
    print('REGISTERED',len(protected),'protected files;',len(original['scene_shards']),'existing frozen shards')

if __name__ == '__main__': main()
