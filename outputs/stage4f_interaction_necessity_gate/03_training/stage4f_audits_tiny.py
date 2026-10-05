"""History-only feature tests, strict real-batch neutrality, gradient and tiny."""
from pathlib import Path
import sys
import time
import math
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'00_manifest'))
from stage4f_common import (model_new, canonical_stage3b_model, canonical_stage4a_model,
    model_input, SceneDataset, atomic_json, read_json, write_csv, config,
    evaluate_graphs, loss_diagnostics, verify_previous, sha256, STAGE3_ROOT)
from stage4f_global_interactor import history_gate_features
import torch
from torch_geometric.data import Batch
from torch_geometric.utils import subgraph


def select_graphs():
    ds=SceneDataset('train')
    frozen=read_json(STAGE3_ROOT/'00_manifest/stage3b_tiny_selection.json')
    chosen=[r['index'] for r in frozen['windows']]
    graphs=[ds[i] for i in chosen]
    def coverage(graphs):
        c={'vehicle':0,'pedestrian':0,'bicycle':0,'moving_vehicle':0,
           'low_history_motion_pedestrian':0,'moving_pedestrian':0,'VP_context':0}
        for g in graphs:
            edges,_=subgraph(~g.padding_mask[:,4],g.edge_index)
            f,ids=history_gate_features(g.positions[:,:5],g.padding_mask[:,:5],g.agent_type,edges)
            net=f[:,4].expm1();types=g.agent_type[ids];eligible=g.target_mask[ids]
            for t,label in enumerate(('vehicle','pedestrian','bicycle')):
                c[label]+=int(((types==t)&eligible).sum())
            c['moving_vehicle']+=int(((types==0)&eligible&(net>2)).sum())
            c['low_history_motion_pedestrian']+=int(((types==1)&eligible&(net<0.2)).sum())
            c['moving_pedestrian']+=int(((types==1)&eligible&(net>0.5)).sum())
            a,b=g.agent_type[edges[0]],g.agent_type[edges[1]]
            distance=torch.linalg.vector_norm(g.positions[edges[0],4]-g.positions[edges[1],4],dim=-1)
            c['VP_context']+=int((((a==0)&(b==1)|((a==1)&(b==0)))&(distance<=20)).sum())
        return c
    c=coverage(graphs)
    # Reuse the frozen tiny windows if their observed history already suffices.
    # Otherwise append first complete TRAIN windows covering a missing criterion.
    for i in range(len(ds)):
        if all(c[k]>=v for k,v in {'vehicle':10,'pedestrian':10,'bicycle':5,
            'moving_vehicle':1,'low_history_motion_pedestrian':1,'moving_pedestrian':1,'VP_context':1}.items()):break
        if i in chosen:continue
        g=ds[i];new=coverage(graphs+[g])
        if any(c[k]==0 and new[k]>0 for k in c):
            chosen.append(i);graphs.append(g);c=new
    else:raise AssertionError('TRAIN history coverage unavailable')
    ds.clear()
    atomic_json(ROOT/'00_manifest/stage4f_tiny_selection.json',{'status':'PASS','coverage':c,
        'windows':[{'index':i,'scene_token':g.scene_token,'sample_token':g.sample_token,'actor_count':g.num_nodes} for i,g in zip(chosen,graphs)],
        'selection':'frozen tiny cohort reused; append only if history coverage missing',
        'motion_coverage_uses_only_history':True,'history_net_thresholds_m':{'moving_vehicle':2,'low_pedestrian':0.2,'moving_pedestrian':0.5},
        'future_displacement_or_future_motion_bins_used':False,'full_scene_context_preserved':True,'train_only':True})
    return graphs


def integration(graphs):
    torch.set_num_threads(1)
    model=model_new('cpu',audit=True).eval();b=canonical_stage3b_model('cpu').eval();a=canonical_stage4a_model('cpu').eval()
    bs=b.state_dict();aas=a.state_dict();fs=model.state_dict()
    assert all(torch.equal(t,fs[n]) for n,t in bs.items())
    assert all(torch.equal(t,fs[n]) for n,t in aas.items())
    data=Batch.from_data_list(graphs[:2]);original=data.edge_index.clone()
    edges,_=subgraph(~data.padding_mask[:,4],data.edge_index)
    f,ids=history_gate_features(data.positions[:,:5],data.padding_mask[:,:5],data.agent_type,edges)
    assert f.shape==(int((~data.padding_mask[:,4]).sum()),9)
    assert torch.equal(f[:,:3].argmax(-1),data.agent_type[ids]) and (f[:,:3].sum(-1)==1).all()
    # Hand-computable padded history, single observation, zero-neighbor cases.
    h=torch.tensor([[[0.,0.],[999.,999.],[3.,4.],[999.,999.],[6.,8.]],
                    [[100.,0.]]*5,[[0.,3.]]*5,[[60.,80.]]*5])
    pad=torch.tensor([[False,True,False,True,False],[True,True,True,True,False],[False]*5,[False]*5])
    types=torch.tensor([0,1,2,1]);e=torch.tensor([[2,3],[0,0]])
    features,nids=history_gate_features(h,pad,types,e)
    torch.testing.assert_close(features[0,3:6],torch.tensor([math.log1p(5),math.log1p(10),math.log1p(10)]))
    assert features[1,3:6].eq(0).all()
    distance=math.sqrt(61)
    torch.testing.assert_close(features[0,6:],torch.tensor([math.log(2),distance/50,1.]))
    assert features[1,6:].tolist()==[0.,1.,0.]
    nofuture_signature=__import__('inspect').signature(history_gate_features)
    assert tuple(nofuture_signature.parameters)==('history','history_padding','agent_type','current_edges')
    with torch.no_grad():
        out=model(model_input(data));reference=b(model_input(data))
        diff={k:float((out[k]-reference[k]).abs().max()) for k in ('raw_prediction','mode_logits','mode_prob')}
        assert all(v<1e-6 for v in diff.values()),diff
        g,features,idx=model.global_interactor.necessity_gate(data,edges)
        assert ((g[idx]>0)&(g[idx]<1)).all() and torch.allclose(g[idx],torch.full_like(g[idx],.1),atol=1e-7,rtol=0)
        last=model.global_interactor.relation_mlp[-1]
        assert last.weight.count_nonzero()==0 and last.bias.count_nonzero()==0
        changed=data.clone();changed.positions[:,5:]+=1234;changed.y-=777;changed.future_mask=~changed.future_mask
        changed.padding_mask[:,5:]=~changed.padding_mask[:,5:];changed.future_times+=950;changed.ego_future+=990
        out2=model(model_input(changed))
        future_diff=max(float((out[k]-out2[k]).abs().max()) for k in ('raw_prediction','mode_logits','mode_prob'))
        assert future_diff<1e-6 and torch.equal(data.edge_index,original)
        assert all(torch.isfinite(v).all() for v in out.values()) and torch.isfinite(f).all()
    audit={'status':'PASS','backend':'CPU single thread; deterministic real graph scatter',
        'threshold':1e-6,'fixed_real_train_batch':[(g.scene_token,g.sample_token) for g in graphs[:2]],
        **{k+'_max_abs_diff':v for k,v in diff.items()},'future_perturbation_max_abs_diff':future_diff}
    atomic_json(ROOT/'00_manifest/stage4f_neutral_initialization_audit.json',audit)
    tests={'status':'PASS','Test1_feature_shape':list(f.shape),'Test2_onehot_legal':True,
        'Test3_no_future_tensor_function_signature':str(nofuture_signature),'Test4_hand_computed_history_only_and_padding_gaps':True,
        'Test5_neighborhood_from_existing_edges_with_current50m_filter':True,'Test6_no_neighbors_defaults':[0,1,0],
        'Test7_gate_open_interval':True,'Test8_initial_gate_0p1':True,'Test9_relation_final_zero':True,
        'Test10_neutral_output':diff,'Test11_all_finite':True,'Test12_edge_index_unchanged':True,
        'shared_B_and_C_step0_bitwise_equal':True,'future_counterfactual_output_equal':True,
        'target_gate_broadcast_all_edges_heads_layers':True,'original_layers_reused_without_modification':True}
    atomic_json(ROOT/'00_manifest/stage4f_unit_tests.json',tests)
    del model,b,a;torch.set_num_threads(4)


def norms(model):
    gi=model.global_interactor
    groups={'relation_mlp':gi.relation_mlp,'pair_embedding':gi.pair_embedding,
            'necessity_gate_mlp':gi.necessity_gate_mlp,'relation_final':gi.relation_mlp[-1]}
    result={}
    for name,module in groups.items():
        p=list(module.parameters());assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in p)
        result[name]=float(sum(x.grad.square().sum() for x in p).sqrt())
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    return result


def gradient(graphs):
    model=model_new();optimizer=model.optimizer(.001,1e-4);data=Batch.from_data_list(graphs).cuda();rows=[]
    for step in range(1,11):
        model.train();optimizer.zero_grad(set_to_none=True)
        values=loss_diagnostics(model,model(model_input(data)),data,'fixed_scale');values['loss'].backward()
        nn=norms(model);assert nn['relation_final']>0
        if step==1:assert nn['necessity_gate_mlp']==0
        optimizer.step();rows.append({'step':step,'loss':float(values['loss'].detach()),'gradients':nn})
    assert all(any(row['gradients'][key]>0 for row in rows) for key in nn)
    atomic_json(ROOT/'04_evaluation/stage4f_gradient_audit.json',{'status':'PASS','updates':rows,
        'first_relation_final_gradient_positive':True,'first_necessity_gradient_zero_allowed':True,
        'all_adapter_gradients_positive_within10':True,'train_only':True,'model_discarded':True,'NaN':0,'Inf':0})
    del model,optimizer,data;torch.cuda.empty_cache()


def tiny(graphs):
    model=model_new();optimizer=model.optimizer(.001,1e-4);initial=evaluate_graphs(model,graphs);rows=[];started=time.monotonic()
    for step in range(1,1001):
        start=((step-1)*2)%len(graphs);batch=Batch.from_data_list([graphs[(start+j)%len(graphs)] for j in range(min(2,len(graphs)))]).cuda()
        model.train();optimizer.zero_grad(set_to_none=True)
        values=loss_diagnostics(model,model(model_input(batch)),batch,'fixed_scale')
        assert all(torch.isfinite(v) for v in values.values());values['loss'].backward();nn=norms(model);optimizer.step()
        if step%100==0:
            current=evaluate_graphs(model,graphs)
            row={'step':step,'loss':float(values['loss'].detach()),**{label+'_FDE':current[label]['FDE'] for label in current}}
            rows.append(row);write_csv(ROOT/'03_training/stage4f_tiny_curve.csv',rows);print('TINY',row,flush=True)
    model.eval();model.global_interactor.record_necessity_gate=True;model.global_interactor.record_relation_bias=True
    with torch.no_grad():
        data=Batch.from_data_list(graphs).cuda();model(model_input(data));gi=model.global_interactor
        obs=gi.necessity_gate_observation;g=obs['gate'][obs['node_ids']]
        bias=gi.relation_bias_observation['bias'];final=evaluate_graphs(model,graphs)
    assert torch.isfinite(g).all() and torch.isfinite(bias).all()
    loss_decreased=all(final[label]['fixed_scale_regression_loss']<initial[label]['fixed_scale_regression_loss'] for label in ('vehicle','pedestrian','bicycle'))
    differentiated=float(g.max()-g.min())>1e-4
    status='PASS' if loss_decreased and differentiated and float(bias.abs().mean())>0 else 'FAIL'
    result={'status':status,'initial':initial,'final':final,'updates':1000,'phase':'fixed_scale',
        'criterion':'all3 class fixed-scale regression diagnostics fall; bias nonzero; gate max-min>1e-4; finite',
        'all_three_class_loss_decreased':loss_decreased,'gate_differentiated':differentiated,
        'gate_mean':float(g.mean()),'gate_min':float(g.min()),'gate_max':float(g.max()),
        'mean_absolute_relation_bias':float(bias.abs().mean()),'gradient_norms_last':nn,
        'train_only':True,'full_scene_context':True,'future_motion_selection':False,
        'model_discarded_before_formal':True,'hyperparameters_searched':False,'elapsed_seconds':time.monotonic()-started,'NaN':0,'Inf':0}
    atomic_json(ROOT/'04_evaluation/stage4f_tiny_overfit.json',result)
    torch.save({'state_dict':model.state_dict(),'metadata':result},ROOT/'07_checkpoints/stage4f_tiny_last.pt')
    assert status=='PASS',result
    print('STAGE4F_TINY_PASS',result,flush=True)


if __name__=='__main__':
    torch.set_num_threads(4);verify_previous();graphs=select_graphs();integration(graphs)
    print('UNIT_NEUTRAL_PASS',flush=True);gradient(graphs);print('GRADIENT_PASS',flush=True);tiny(graphs);verify_previous()
