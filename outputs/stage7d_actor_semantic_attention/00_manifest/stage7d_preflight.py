"""Identity, canonical neutral step0, semantic gradients and fixed tiny health."""
from pathlib import Path
import sys, random, math, os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'00_manifest'))
from stage7d_common import *
from stage3b_common import model_new as baseline_new
from stage7d_dataset import semantic_vector
sys.path.insert(0,str(ROOT.parent/'stage7c_actor_semantic_relevance_audit/01_attention_capture'))
from stage7c_hooks import LaneCapture
import numpy as np
import torch
from torch_geometric.data import Batch

def equal(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
    if isinstance(a,np.ndarray):return np.array_equal(a,b)
    return a==b

def main():
    assert read_json(PREREG)['training_authorized'] and not CURVE.exists()
    identity=[];rng=random.Random(2022)
    for split in ('train','val'):
        base=SceneDataset(split);ds=Stage7DSemanticDataset(split)
        chosen=rng.sample(range(len(ds)),100)
        chosen.sort(key=lambda i:(ds.rows[i]['file_path'],i))
        for i in chosen:
            a=base[i];b=ds[i];assert set(b.keys())-set(a.keys())=={'lane_semantic'}
            assert all(equal(a[k],b[k]) for k in a.keys())
            identity.append({'split':split,'dataset_index':i,'scene_token':a.scene_token,'sample_token':a.sample_token,
                'all_original_fields_bitwise_equal':True,'original_fields':sorted(a.keys())})
        base.clear();ds.clear()
    atomic_json(ROOT/'01_data_audit/stage7d_data_identity_audit.json',{'status':'PASS','seed':2022,'TRAIN_windows':100,'VAL_windows':100,'only_new_field':'lane_semantic','windows':identity})
    ds=Stage7DSemanticDataset('train');order=list(range(len(ds)));rng=random.Random(2022);rng.shuffle(order)
    # Prioritize bicycle-bearing rows without consulting any prediction error.
    candidates=[i for i in order if int(ds.rows[i]['full_bicycle'])>0]+[i for i in order if int(ds.rows[i]['full_bicycle'])==0]
    chosen=[];graphs=[];special=False;types=set();moving=False
    for i in candidates:
        g=ds[i];s=g.lane_semantic
        contextual=bool((s[:,1]+s[:,3]).any() and s[:,5:8].any() and s[:,8].any())
        t=set(g.agent_type[g.target_mask].tolist());mv=any(g.t0_motion_state[j]=='vehicle.moving' for j in torch.where(g.target_mask)[0].tolist())
        if not graphs or (contextual and not special) or not t.issubset(types) or (mv and not moving):
            chosen.append(i);graphs.append(g);special|=contextual;types|=t;moving|=mv
        if special and types=={0,1,2} and moving:break
    assert special and types=={0,1,2} and moving and len(graphs)<=6
    for i in order:
        if len(graphs)==6:break
        if i not in chosen:chosen.append(i);graphs.append(ds[i])
    ds.clear();assert len(graphs)==6
    selection=[{'index':i,'scene_token':g.scene_token,'sample_token':g.sample_token,
      'targets_by_type':{c:int(((g.agent_type==t)&g.target_mask).sum()) for t,c in enumerate(CLASSES)},
      'moving_targets':sum(g.t0_motion_state[j]=='vehicle.moving' for j in torch.where(g.target_mask)[0].tolist()),
      'turn_segments':int((g.lane_semantic[:,1]+g.lane_semantic[:,3]).sum()),
      'control_segments':int(g.lane_semantic[:,5:8].any(-1).sum()),'crosswalk_segments':int(g.lane_semantic[:,8].sum())} for i,g in zip(chosen,graphs)]
    atomic_json(ROOT/'00_manifest/stage7d_tiny_selection.json',{'seed':2022,'selection':'TRAIN attributes/semantics only','windows':selection})
    baseline=baseline_new();canonical_rng=torch.get_rng_state().clone()
    model=model_new();assert torch.equal(canonical_rng,torch.get_rng_state())
    shared=baseline.state_dict();own=model.state_dict()
    assert all(torch.equal(t,own[name]) for name,t in shared.items())
    parameter_diff=max(float((p-dict(model.named_parameters())[n]).abs().max()) for n,p in baseline.named_parameters())
    assert parameter_diff==0
    batch=Batch.from_data_list(graphs).cuda();baseline.eval();model.eval()
    captures=[{},{}];handles=[]
    for model_id,m in enumerate((baseline,model)):
        for name in ('lin_q','lin_k','lin_v'):
            def observe(module,args,out,mid=model_id,n=name):captures[mid][n]=out.detach().clone()
            handles.append(getattr(m.local_encoder.al_encoder,name).register_forward_hook(observe))
    capture=LaneCapture(model,representations=False)
    bias_inputs=[];bias_handle=model.local_encoder.al_encoder.semantic_score_mlp.register_forward_pre_hook(
        lambda module,args:bias_inputs.append(args[0].detach().clone()))
    torch.use_deterministic_algorithms(True)
    with torch.no_grad():
        a=baseline(batch);b=model(batch)
        diffs={k:float((a[k]-b[k]).abs().max()) for k in ('raw_prediction','mode_logits','mode_prob')}
        scores=[(c['lin_q'].reshape(-1,8,8)*c['lin_k'].reshape(-1,8,8)).sum(-1)/(8**.5) for c in captures]
        attention_score_diff=float((scores[0]-scores[1]).abs().max());assert attention_score_diff==0
        initial_alpha=capture.data['alpha'].copy()
        z=bias_inputs[-1];assert torch.count_nonzero(model.local_encoder.al_encoder.semantic_score_mlp(z))==0
        edge=torch.as_tensor(capture.data['edge_index'],device=batch.x.device)
        rotated=torch.bmm(batch.lane_actor_vectors.new_tensor(capture.data['edge_attr']).unsqueeze(-2),b['rotation'][edge[1]]).squeeze(-2)
        assert torch.equal(z[:,:9],batch.lane_semantic[edge[0]])
        assert torch.equal(z[:,9:11],rotated/50)
        assert torch.equal(z[:,11:],torch.nn.functional.one_hot(batch.agent_type[edge[1]],3).to(z.dtype))
        original={k:v.clone() for k,v in model.local_encoder.al_encoder.semantic_score_mlp.state_dict().items()}
        model.local_encoder.al_encoder.semantic_score_mlp[2].weight.fill_(.125)
        _=model(batch);kv_original={n:captures[1][n].clone() for n in ('lin_k','lin_v')}
        changed=batch.clone();changed.lane_semantic=1-batch.lane_semantic
        _=model(changed)
        kv_diff={n:float((captures[1][n]-kv_original[n]).abs().max()) for n in kv_original}
        assert all(v==0 for v in kv_diff.values())
        model.local_encoder.al_encoder.semantic_score_mlp.load_state_dict(original)
    capture.close();bias_handle.remove()
    for h in handles:h.remove()
    assert all(v<1e-6 for v in diffs.values())
    torch.use_deterministic_algorithms(False)
    count=sum(p.numel() for p in model.parameters());basecount=sum(p.numel() for p in baseline.parameters())
    atomic_json(ROOT/'00_manifest/stage7d_neutral_initialization_audit.json',{
      'status':'PASS','max_shared_parameter_diff':parameter_diff,'shared_buffers_bitwise_equal':True,
      'output_max_differences':diffs,'baseline_shared_state_sha256':state_digest(shared),
      'parameters':count,'baseline_parameters':basecount,'additional_parameters':count-basecount,
      'increase_percent':100*(count-basecount)/basecount,'real_TRAIN_batch_graphs':6,
      'trained_checkpoint_loaded':False,'canonical_constructor':'stage3b_common.model_new(audit=False)',
      'device':'CUDA','deterministic_algorithms_for_neutral_comparison':True,
      'CUBLAS_WORKSPACE_CONFIG':':4096:8','training_keeps_original_Stage3B_kernel_policy':True,
      'original_attention_score_max_diff':attention_score_diff,'initial_bias_exact_zero':True,
      'post_constructor_CPU_RNG_identical':True,'edge_z_semantic_rotation_actor_type_exact':True,
      'nonzero_bias_semantic_perturbation_KV_max_differences':kv_diff})
    del baseline;torch.cuda.empty_cache()
    optimizer=model.optimizer(0.001,0.0001);gradients=[];curve=[]
    semantic=model.local_encoder.al_encoder.semantic_score_mlp
    for step in range(1,201):
        model.train();optimizer.zero_grad(set_to_none=True);out=model(batch)
        values=loss_diagnostics(model,out,batch,'fixed_scale');assert all(torch.isfinite(v) for v in values.values())
        values['loss'].backward();assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        norms={label:sum(float(p.grad.square().sum()) for p in module.parameters() if p.grad is not None)**0.5
          for label,module in [('semantic_first',semantic[0]),('semantic_last',semantic[2]),('TypeEmbedding',model.type_embedding),
                               ('LocalEncoder',model.local_encoder),('GlobalInteractor',model.global_interactor),('Decoder',model.decoder)]}
        if step<=10:gradients.append({'step':step,**norms})
        optimizer.step()
        curve.append({'step':step,'loss':float(values['loss'].detach()),'mean_abs_semantic_bias':float(semantic(z).abs().mean().detach()),**norms})
        if step%50==0:print('TINY',step,curve[-1]['loss'],flush=True)
    assert gradients[0]['semantic_last']>0
    assert all(any(r[k]>0 for r in gradients) for k in gradients[0] if k!='step')
    assert np.mean([r['loss'] for r in curve[-20:]])<np.mean([r['loss'] for r in curve[:20]])
    assert curve[-1]['mean_abs_semantic_bias']>0
    model.eval();capture=LaneCapture(model,representations=False)
    with torch.no_grad():_=model(batch)
    final_alpha=capture.data['alpha'];capture.close()
    assert np.isfinite(final_alpha).all() and np.max(np.abs(final_alpha-initial_alpha))>0
    write_csv(ROOT/'03_training/stage7d_tiny_curve.csv',curve)
    atomic_json(ROOT/'03_training/stage7d_tiny_overfit.json',{'status':'PASS','updates':200,'fixed_TRAIN_windows':6,
      'initial20_loss_mean':float(np.mean([r['loss'] for r in curve[:20]])),
      'final20_loss_mean':float(np.mean([r['loss'] for r in curve[-20:]])),
      'final_mean_abs_semantic_bias':curve[-1]['mean_abs_semantic_bias'],
      'initial_bias_mean_abs':0.,'attention_max_abs_change':float(np.max(np.abs(final_alpha-initial_alpha))),
      'NaN':0,'Inf':0,
      'formal_training_uses_fresh_constructor':True,'structure_tuned':False})
    atomic_json(ROOT/'00_manifest/stage7d_formal_health_audit.json',{'status':'PASS','first10_gradients':gradients,
      'last_linear_nonzero_first_update':True,'first_and_last_linear_nonzero_within10':True,'all_shared_modules_finite_nonzero':True})
    print('PREFLIGHT_PASS',diffs,'parameters',count,'tiny_loss',curve[0]['loss'],'->',curve[-1]['loss'],flush=True)

if __name__=='__main__':torch.set_num_threads(4);main()
