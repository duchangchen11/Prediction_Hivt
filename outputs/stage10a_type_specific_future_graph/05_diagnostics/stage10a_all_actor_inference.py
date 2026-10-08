"""Real TRAIN window: infer every current-valid actor without future-target filtering."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'01_training/00_code'))
from stage10a_common import *
from stage10a_model import DualExpert
import copy

@torch.no_grad()
def main():
    seed(); verify(); assert read_json(FROZEN)['status']=='FROZEN_ALL_COMPLETE'
    experts={e:fresh(e) for e in EXPERTS}
    for e,m in experts.items():
        m.load_state_dict(torch.load(ROOT/'02_checkpoints'/CPNAMES[e],map_location='cpu',weights_only=False)['state_dict'])
        m.eval().requires_grad_(False)
    r2=frozen_r2(); model=DualExpert(experts['vehicle'],experts['pedestrian'],r2).eval()
    records=[r for r in read_json(S6/'01_cache/stage6a_cache_manifest.json')['batches'] if r['split']=='train']
    selected_window=None
    # Fixed first TRAIN window containing all three current-valid types.
    for rec in records:
        block=torch.load(S6/rec['path'],map_location='cpu',weights_only=False)
        for w in block['windows']:
            current=~w['history_padding'][:,4]
            if all(bool(((w['agent_type']==t)&current).any()) for t in range(3)):
                assert sha256(S6/rec['path'])==rec['sha256']; selected_window=w; break
        if selected_window is not None: break
    assert selected_window is not None
    w=selected_window; targets=torch.where(~w['history_padding'][:,4])[0]
    args,_,_=graph_from_window(w,targets); args=tuple(a.cuda() for a in args)
    norm=read_json(S6/'02_features/stage6a_normalization.json')
    feature,flags,_,_=observable_features(*[w[k].cuda() for k in ('history','history_padding','agent_type','ego_prediction','mode_logits','mode_prob')])
    feature=r2normalize(feature.cpu(),flags.cpu(),norm,'R2')[targets].cuda()
    candidate=w['ego_prediction'][targets].cuda(); before=tensor_sha(candidate)
    out=model(*args,feature,candidate)
    assert out['candidates'] is candidate and tensor_sha(candidate)==before
    assert out['mode_logits'].shape==out['mode_prob'].shape==(len(targets),6)
    assert torch.isfinite(out['mode_prob']).all()
    assert torch.allclose(out['mode_prob'].sum(-1),torch.ones(len(targets),device='cuda'),atol=1e-6,rtol=0.)
    poisoned=copy.deepcopy(w); poisoned['GT'].fill_(float('nan'))
    poisoned['future_mask']=~poisoned['future_mask']; poisoned['target_mask']=~poisoned['target_mask']
    pp,_,_=graph_from_window(poisoned,targets); pp=tuple(a.cuda() for a in pp)
    assert all(torch.equal(a,b) for a,b in zip(args,pp))
    second=model(*pp,feature,candidate)
    assert all(torch.equal(out[k],second[k]) for k in out)
    ty=w['agent_type'][targets]; bike=(ty==2).cuda(); ref=r2(feature,args[3])
    assert torch.equal(out['mode_logits'][bike],ref['mode_logits'][bike]) and torch.equal(out['mode_prob'][bike],ref['mode_prob'][bike])
    # Exercise guard logic directly: it must reject the forbidden pathname before any I/O.
    import stage10a_common
    caught=0
    for path in FORBIDDEN_DATA:
        try: stage10a_common._data_guard('open',(path+'/probe.pt','r',0))
        except RuntimeError: caught+=1
    assert caught==len(FORBIDDEN_DATA)
    atomic_json(ROOT/'05_diagnostics/stage10a_all_actor_inference.json',{'status':'PASS','split':'official TRAIN only',
        'scene_token':w['scene_token'],'sample_token':w['sample_token'],'dataset_index':w['dataset_index'],
        'current_valid_targets':len(targets),'counts':{c:int((ty==t).sum()) for t,c in enumerate(CLASSES)},
        'full_horizon_targets':int((w['target_mask'][targets]&w['future_mask'][targets].all(-1)).sum()),
        'target_future_mask_not_used_to_select_inference_targets':True,'all_actor_output_K':6,
        'GT_poison_outputs_bitwise_identical':True,'Bicycle_logits_probabilities_candidates_bitwise_R2':True,
        'candidate_requires_grad':False,'guard_logic_rejections_without_file_access':caught,
        'official_VAL_opened':False,'test_opened':False})
    print('STAGE10_ALL_ACTOR_INFERENCE_PASS',len(targets),flush=True)

if __name__=='__main__': main()
