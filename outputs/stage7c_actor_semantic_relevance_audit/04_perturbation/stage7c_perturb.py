"""Exactly two OOD inference passes; frozen ON/ZERO are reused, never retrained."""
from pathlib import Path
import sys,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'00_manifest'))
from stage7c_common import *
from stage7a_evaluation_core import evaluate

class ShuffleDataset(Stage7ASemanticDataset):
    def __init__(self):
        super().__init__('val');self.permutations={}
    def __getitem__(self,index):
        g=super().__getitem__(index);s=g.lane_semantic
        key=g.scene_token+g.sample_token
        sub=int.from_bytes(hashlib.sha256(key.encode()).digest()[:4],'big')
        rng=np.random.default_rng(np.random.SeedSequence([2022,sub]))
        perm=rng.permutation(len(s))
        new=s[torch.from_numpy(perm)].clone()
        assert torch.equal(s.sum(0),new.sum(0))
        codes=(s.numpy().astype(np.int32)*(1<<np.arange(9))).sum(1)
        new_codes=(new.numpy().astype(np.int32)*(1<<np.arange(9))).sum(1)
        assert np.array_equal(np.sort(codes),np.sort(new_codes))
        self.permutations[key]={'scene_token':g.scene_token,'sample_token':g.sample_token,
            'lanes':len(s),'semantic_rows_changed':int((codes!=new_codes).sum()),
            'permutation_sha256':hashlib.sha256(perm.astype('<i8').tobytes()).hexdigest()}
        g.lane_semantic=new
        return g

@torch.no_grad()
def main():
    torch.set_num_threads(4)
    assert read_json(ROOT/'00_manifest/stage7c_attention_capture_integrity.json')['status']=='PASS'
    completed={}
    for name in ['SHUFFLE','CENTERED']:
        model=load_model('Stage7A');before=state_digest(model.state_dict())
        hook=None
        if name=='SHUFFLE':ds=ShuffleDataset()
        else:
            ds=Stage7ASemanticDataset('val')
            mlp=model.local_encoder.al_encoder.semantic_mlp
            r0=mlp(torch.zeros((1,9),device='cuda')).detach().clone()
            def centered(module,args,output):return output-r0
            hook=mlp.register_forward_hook(centered)
        path=ROOT/f'04_perturbation/stage7c_{name.lower()}_actor_errors.csv'
        measured=evaluate(ds,model,actor_path=path,progress=True)
        if hook:hook.remove()
        assert state_digest(model.state_dict())==before
        assert measured['metrics']['full_horizon']['overall']['count']==54990
        assert measured['metrics']['partial_future']['overall']['count']==30037
        measured.update(checkpoint_sha256=BEST_SHA,OOD_diagnostic=True,not_formal_baseline=True,
            perturbation=name,training=False,new_checkpoint=False,model_state_unchanged=True)
        atomic_json(ROOT/f'04_perturbation/stage7c_{name.lower()}_metrics.json',measured)
        if name=='SHUFFLE':
            assert len(ds.permutations)==3603
            pd.DataFrame(ds.permutations.values()).to_csv(ROOT/'06_tables/stage7c_shuffle_permutation_audit.csv',index=False)
        completed[name]=measured['metrics']['full_horizon']['overall']
        print('PERTURBATION_COMPLETE',name,completed[name],flush=True)
        del model,ds;torch.cuda.empty_cache()
    atomic_json(ROOT/'04_perturbation/stage7c_perturbation_run_audit.json',{
        'status':'PASS','forward_runs':2,'reused_frozen_runs':['Stage3B','Stage7A-ON','Stage7A-ZERO'],
        'completed':completed,'VAL_scenes':150,'supervised_windows':3603,'full_targets':54990,
        'training_updates':0,'new_checkpoints':0,'test_used':False,'Stage7B_executed':False})

if __name__=='__main__':main()
