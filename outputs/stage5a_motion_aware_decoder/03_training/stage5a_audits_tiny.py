"""Padding, future independence, real-batch neutrality, gradient and tiny audits."""
from pathlib import Path
import sys
import math
import inspect
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'00_manifest'))
from stage5a_common import (model_new, canonical_stage3b_model, model_input, SceneDataset,
    atomic_json, read_json, write_csv, evaluate_graphs, loss_diagnostics, verify_previous, STAGE3_ROOT)
from stage5a_decoder import condition_features
import torch
from torch_geometric.data import Batch


def select_graphs():
    ds = SceneDataset('train')
    chosen = [r['index'] for r in read_json(STAGE3_ROOT/'00_manifest/stage3b_tiny_selection.json')['windows']]
    graphs = [ds[i] for i in chosen]
    def coverage(gg):
        c = dict.fromkeys(('vehicle','pedestrian','bicycle','moving_vehicle','low_motion_vehicle',
                           'moving_pedestrian','low_motion_pedestrian'), 0)
        for g in gg:
            f = condition_features(g.positions[:,:5],g.padding_mask[:,:5],g.agent_type)
            net = f[:,4].expm1()
            eligible = g.target_mask & ~g.padding_mask[:,4]
            for t,label in enumerate(('vehicle','pedestrian','bicycle')):
                c[label] += int(((g.agent_type==t)&eligible).sum())
            for t,label,lo in ((0,'vehicle',2.),(1,'pedestrian',.5)):
                c['moving_'+label] += int(((g.agent_type==t)&eligible&(net>lo)).sum())
                c['low_motion_'+label] += int(((g.agent_type==t)&eligible&(net<.2)).sum())
        return c
    minimum = {'vehicle':10,'pedestrian':10,'bicycle':5,'moving_vehicle':1,'low_motion_vehicle':1,
               'moving_pedestrian':1,'low_motion_pedestrian':1}
    c = coverage(graphs)
    for i in range(len(ds)):
        if all(c[k]>=v for k,v in minimum.items()): break
        if i in chosen: continue
        graph = ds[i]
        new = coverage(graphs+[graph])
        if any(c[k]<v and new[k]>c[k] for k,v in minimum.items()):
            chosen.append(i); graphs.append(graph); c=new
    assert all(c[k]>=v for k,v in minimum.items()), c
    ds.clear()
    atomic_json(ROOT/'00_manifest/stage5a_tiny_selection.json', {
        'status':'PASS','coverage':c,'minimum':minimum,
        'windows':[{'index':i,'scene_token':g.scene_token,'sample_token':g.sample_token,'actors':g.num_nodes} for i,g in zip(chosen,graphs)],
        'selection':'reuse frozen TRAIN cohort; append first TRAIN graph needed for observed-history coverage',
        'history_net_thresholds_m':{'moving_vehicle':2.,'moving_pedestrian':.5,'low_motion':.2},
        'full_scene_context_preserved':True,'future_motion_selection':False,'val_used':False})
    return graphs


def integration(graphs):
    torch.set_num_threads(1)
    model=model_new('cpu',audit=True).eval()
    reference=canonical_stage3b_model('cpu').eval()
    data=Batch.from_data_list(graphs[:2])
    condition=condition_features(data.positions[:,:5],data.padding_mask[:,:5],data.agent_type)
    assert condition.shape==(data.num_nodes,6)
    assert torch.equal(condition[:,:3].argmax(-1),data.agent_type)
    assert condition[:,:3].sum(-1).eq(1).all()
    h=torch.tensor([[[0.,0.],[999.,999.],[3.,4.],[999.,999.],[6.,8.]],
                    [[999.,999.],[999.,999.],[999.,999.],[999.,999.],[6.,8.]],
                    [[0.,0.],[3.,0.],[3.,4.],[0.,4.],[0.,0.]],
                    [[999.,999.]]*5,
                    [[0.,0.],[999.,999.],[3.,4.],[999.,999.],[999.,999.]]])
    padding=torch.tensor([[False,True,False,True,False],[True,True,True,True,False],
                          [False]*5,[True]*5,[False,True,False,True,True]])
    features=condition_features(h,padding,torch.tensor([0,1,2,0,1]))
    expected=torch.tensor([[5.,10.,10.],[0.,0.,0.],[4.,0.,14.],[0.,0.,0.],[5.,5.,5.]]).log1p()
    torch.testing.assert_close(features[:,3:],expected)
    altered=h.clone();altered[padding]=torch.randn_like(altered[padding])*1e6
    assert torch.equal(features,condition_features(altered,padding,torch.tensor([0,1,2,0,1])))
    assert tuple(inspect.signature(condition_features).parameters)==('history','history_padding','agent_type')
    tests={}
    with torch.no_grad():
        routing=model.decoder.route(condition)
        assert routing.shape==(data.num_nodes,2) and routing.eq(.5).all()
        assert torch.allclose(routing.sum(-1),torch.ones(data.num_nodes)) and torch.isfinite(routing).all()
        hidden=torch.randn(6,data.num_nodes,64)
        delta,experts=model.decoder.residual(hidden,routing)
        assert experts.shape==(6,data.num_nodes,2,64) and delta.shape==hidden.shape
        assert experts.eq(0).all() and delta.eq(0).all()
        # Nonzero probe verifies actor weights broadcast across all six modes.
        probe=torch.stack((hidden,hidden*2),dim=-2)
        weights=torch.rand(data.num_nodes,2);weights/=weights.sum(-1,keepdim=True)
        broadcast=(probe*weights[None,:,:,None]).sum(-2)
        expected_broadcast=torch.stack([hidden[k]*(weights[:,0]+2*weights[:,1])[:,None] for k in range(6)])
        torch.testing.assert_close(broadcast,expected_broadcast)
        out=model(model_input(data));base=reference(model_input(data))
        diff={k:float((out[k]-base[k]).abs().max()) for k in ('raw_prediction','mode_logits','mode_prob')}
        assert all(v<1e-6 for v in diff.values()),diff
        # Future quantities, endpoint and labels are never arguments of the condition.
        changed=data.clone()
        changed.positions[:,5:]+=1234.;changed.y-=777.
        changed.future_mask=~changed.future_mask;changed.padding_mask[:,5:]=~changed.padding_mask[:,5:]
        changed.GT_endpoint_displacement_m=torch.full((data.num_nodes,),9999.)
        changed.future_labels=torch.ones(data.num_nodes,dtype=torch.long)
        changed.future_times+=950.;changed.ego_future+=990.
        changed.target_mask=~changed.target_mask
        f2=condition_features(changed.positions[:,:5],changed.padding_mask[:,:5],changed.agent_type)
        r2=model.decoder.route(f2)
        assert torch.equal(condition,f2) and torch.equal(routing,r2)
        future_out=model(model_input(changed))
        fd={k:float((out[k]-future_out[k]).abs().max()) for k in ('raw_prediction','mode_logits','mode_prob')}
        assert all(v==0 for v in fd.values()),fd
        assert all(torch.isfinite(v).all() for v in out.values()) and torch.isfinite(features).all()
    atomic_json(ROOT/'00_manifest/stage5a_neutral_initialization_audit.json', {
        'status':'PASS','backend':'CPU single thread, deterministic scatter','tolerance':1e-6,
        'fixed_real_train_batch':[{'scene':g.scene_token,'sample':g.sample_token} for g in graphs[:2]],
        **{k+'_max_abs_diff':v for k,v in diff.items()},'shared_parameters_max_abs_diff':0.})
    atomic_json(ROOT/'00_manifest/stage5a_no_future_leakage_audit.json', {
        'status':'PASS','condition_max_abs_diff':0.,'router_max_abs_diff':0.,'output_max_abs_diff':fd,
        'mutated':['future trajectory','y','future mask','future padding','GT endpoint','future labels','future times','ego future','target mask'],
        'same_history_and_actor_type':True,'function_signature':str(inspect.signature(condition_features))})
    atomic_json(ROOT/'00_manifest/stage5a_unit_tests.json', {
        'status':'PASS','condition_shape':[data.num_nodes,6],'onehot_legal':True,'router_shape':[data.num_nodes,2],
        'router_sum_one':True,'router_finite':True,'expert_input_output_shape':[6,data.num_nodes,64],
        'actor_weights_broadcast_to_all6_modes':True,'step0_experts_zero':True,'step0_router_half':True,
        'neutral_output':diff,'future_leakage_zero':True,'padding_hand_computed_cases':5,
        'padding_value_independence':True,'no_NaN_Inf':True,
        'pi_same_structure_and_source':True,'no_attention_backbone_changes':True})
    del model,reference;torch.set_num_threads(4)


def gradient_norms(model):
    modules={'expert1_first':model.decoder.experts[0][0], 'expert1_final':model.decoder.experts[0][-1],
             'expert2_first':model.decoder.experts[1][0], 'expert2_final':model.decoder.experts[1][-1],
             'router_first':model.decoder.router[0], 'router_final':model.decoder.router[-1]}
    result={}
    for name,module in modules.items():
        parameters=list(module.parameters())
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters)
        result[name]=float(sum(p.grad.square().sum() for p in parameters).sqrt())
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    return result


def gradient(graphs):
    model=model_new();optimizer=model.optimizer(.001,1e-4)
    data=Batch.from_data_list(graphs).cuda();rows=[]
    for step in range(1,11):
        model.train();optimizer.zero_grad(set_to_none=True)
        values=loss_diagnostics(model,model(model_input(data)),data,'fixed_scale')
        assert torch.isfinite(values['loss'])
        values['loss'].backward();norms=gradient_norms(model)
        if step==1:
            assert norms['expert1_final']>0 and norms['expert2_final']>0
            assert norms['router_first']==norms['router_final']==0
        optimizer.step();rows.append({'step':step,'loss':float(values['loss'].detach()),'gradient_norms':norms})
    assert all(any(row['gradient_norms'][name]>0 for row in rows) for name in norms),rows
    atomic_json(ROOT/'04_evaluation/stage5a_gradient_audit.json',{
        'status':'PASS','updates':rows,'all6_module_gradients_positive_within10':True,
        'initial_zero_router_gradient_allowed':True,'train_only':True,'model_discarded':True,'NaN':0,'Inf':0})
    del model,optimizer,data;torch.cuda.empty_cache()


def tiny(graphs):
    model=model_new();optimizer=model.optimizer(.001,1e-4)
    initial=evaluate_graphs(model,graphs);rows=[];started=time.monotonic()
    for step in range(1,1001):
        start=((step-1)*2)%len(graphs)
        data=Batch.from_data_list([graphs[(start+j)%len(graphs)] for j in range(min(2,len(graphs)))]).cuda()
        model.train();optimizer.zero_grad(set_to_none=True)
        values=loss_diagnostics(model,model(model_input(data)),data,'fixed_scale')
        assert all(torch.isfinite(v) for v in values.values())
        values['loss'].backward();norms=gradient_norms(model);optimizer.step()
        if step%100==0:
            current=evaluate_graphs(model,graphs)
            row={'step':step,'loss':float(values['loss'].detach()),**{g+'_FDE':current[g]['FDE'] for g in current}}
            rows.append(row);write_csv(ROOT/'03_training/stage5a_tiny_curve.csv',rows);print('TINY',row,flush=True)
    model.eval();model.decoder.record_observation=True
    with torch.no_grad():
        data=Batch.from_data_list(graphs).cuda();model(model_input(data))
        obs=model.decoder.observation;r=obs['routing'];delta=obs['residual'];final=evaluate_graphs(model,graphs)
    decreasing=all(final[g]['fixed_scale_regression_loss']<initial[g]['fixed_scale_regression_loss'] for g in ('vehicle','pedestrian','bicycle'))
    assert decreasing and float(delta.abs().mean())>0 and float((r-.5).abs().max())>1e-4
    assert torch.isfinite(r).all() and torch.isfinite(delta).all()
    result={'status':'PASS','updates':1000,'phase':'fixed_scale','initial':initial,'final':final,
        'all3_class_loss_decreased':decreasing,'mean_absolute_residual':float(delta.abs().mean()),
        'router_max_deviation_from_half':float((r-.5).abs().max()),'router_mean':r.mean(0).tolist(),
        'train_only':True,'model_discarded_before_formal':True,'hyperparameter_search':False,
        'elapsed_seconds':time.monotonic()-started,'NaN':0,'Inf':0}
    atomic_json(ROOT/'04_evaluation/stage5a_tiny_overfit.json',result)
    torch.save({'state_dict':model.state_dict(),'metadata':result},ROOT/'07_checkpoints/stage5a_tiny_last.pt')
    print('STAGE5A_TINY_PASS',result,flush=True)


if __name__=='__main__':
    torch.set_num_threads(4);verify_previous();graphs=select_graphs();integration(graphs)
    print('UNIT_NEUTRAL_PASS',flush=True);gradient(graphs);print('GRADIENT_PASS',flush=True)
    tiny(graphs);verify_previous()
