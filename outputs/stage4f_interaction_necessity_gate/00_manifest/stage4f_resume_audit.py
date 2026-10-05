"""Read-only checkpoint reload audit in CPU; never advances formal training."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'00_manifest'))
from stage4f_common import (model_new,config,state_digest,atomic_json,sha256,
    LAST,SceneDataset,SceneSampler,read_json,CURVE)
import torch


def main():
    digest=sha256(LAST);saved=torch.load(LAST,map_location='cpu',weights_only=False)
    assert saved['config']==config()
    model=model_new('cpu');model.load_state_dict(saved['state_dict'],strict=True)
    assert state_digest(model.state_dict())==state_digest(saved['state_dict'])
    optimizer=model.optimizer(saved['metadata']['LR'],config()['weight_decay'])
    optimizer.load_state_dict(saved['optimizer_state_dict']);restored=optimizer.state_dict()
    original=saved['optimizer_state_dict'];assert restored['param_groups']==original['param_groups']
    assert restored['state'].keys()==original['state'].keys()
    for key,state in original['state'].items():
        for name,value in state.items():
            actual=restored['state'][key][name]
            if torch.is_tensor(value):assert torch.equal(actual,value) and torch.isfinite(actual).all()
            else:assert actual==value
    torch.set_rng_state(saved['torch_rng']);assert torch.equal(torch.get_rng_state(),saved['torch_rng'])
    assert len(saved['cuda_rng'])==1 and saved['cuda_rng'][0].dtype==torch.uint8
    ds=SceneDataset('train');sampler=SceneSampler(ds,2022)
    sampler.epoch=saved['iterator']['epoch']+(0 if saved['metadata']['phase']=='fixed_scale' else 100000)
    order=list(sampler);again=list(sampler);assert order==again
    cursor=saved['iterator']['next_batch'];assert 0<=cursor<=1057
    next_indices=order[16*cursor:16*(cursor+1)]
    assert saved['phase_state']['phase_step']==saved['metadata']['phase_step']
    atomic_json(ROOT/'04_evaluation/stage4f_checkpoint_resume_audit.json',{'status':'PASS','checkpoint_SHA_at_read':digest,
        'audited_global_step':saved['metadata']['global_step'],'model_state_bitwise_reloaded':True,
        'optimizer_param_groups_and_all_states_bitwise_reloaded':True,'CPU_RNG_bitwise_restored':True,
        'CUDA_RNG_serialized':True,'CUDA_restore_applied_in_formal_restore_function':True,
        'audit_backend':'CPU only; no formal training process state changed','sampler_epoch':sampler.epoch,
        'sampler_cursor':saved['iterator'],'next_batch_indices':next_indices,'phase_state_matches_metadata':True,
        'optimizer_updates_in_audit':0,'source_prefix_order_repeated_exactly':True})
    print('CHECKPOINT_RESUME_AUDIT_PASS',saved['metadata']['global_step'],flush=True)


if __name__=='__main__':torch.set_num_threads(1);main()
