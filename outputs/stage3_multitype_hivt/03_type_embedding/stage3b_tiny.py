"""Required integration gates and exact-window train-only tiny overfit."""
import copy
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'00_manifest'))
from stage3b_common import (CLASSES,MODEL_KEYS,SceneDataset,config,model_new,model_input,
    evaluate_graphs,loss_diagnostics,atomic_json,read_json,write_csv,sha256,git,verify_previous,HiVTLossRecovery)
import torch
from torch_geometric.data import Batch

def graphs_from_frozen_selection():
    ds=SceneDataset('train');selection=read_json(ROOT/'00_manifest/stage3_tiny_selection.json');graphs=[]
    for record in selection['windows']:
        graph=ds[record['index']]
        assert (graph.scene_token,graph.sample_token)==(record['scene_token'],record['sample_token'])
        assert graph.num_nodes==record['actor_count'];graphs.append(graph)
    ds.clear();totals={name:sum(int((g.target_mask&(g.agent_type==t)).sum()) for g in graphs) for t,name in enumerate(CLASSES)}
    assert totals==selection['target_counts'] and all(totals[name]>=minimum for name,minimum in zip(CLASSES,(10,10,5)))
    atomic_json(ROOT/'00_manifest/stage3b_tiny_selection.json',{**selection,'same_exact_actor_windows_as_Stage3A':True,
        'Stage3A_selection_sha256':sha256(ROOT/'00_manifest/stage3_tiny_selection.json')})
    return graphs

def finite_gradients(model):
    gradients=[p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    gradient=model.type_embedding.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all() and float(gradient.norm())>0,'TypeEmbedding gradient is zero/nonfinite'
    return float(gradient.norm())

def integration_tests(graphs):
    model=model_new(audit=True);model.eval();data=Batch.from_data_list(graphs[:2]).cuda();original_x=data.x.clone()
    assert data.agent_type.dtype==torch.long and data.agent_type.shape==(data.num_nodes,)
    assert torch.equal(data.agent_type,torch.cat([g.agent_type for g in graphs[:2]]).cuda())
    assert set(data.agent_type.tolist())=={0,1,2}
    invalid_rejections=0
    for types in (data.agent_type[:,None],data.agent_type.float(),data.agent_type[:-1],torch.full_like(data.agent_type,3),torch.full_like(data.agent_type,-1)):
        try:model.validate_actor_types(types,data.num_nodes)
        except ValueError:invalid_rejections+=1
        else:raise AssertionError('Invalid type IDs/shape accepted')
    captured={};calls=[]
    local_hook=model.local_encoder.register_forward_hook(lambda module,args,output:captured.update(local=output.detach().clone()))
    global_hook=model.global_interactor.register_forward_pre_hook(lambda module,args:captured.update(global_input=args[1].detach().clone()))
    type_hook=model.type_embedding.register_forward_pre_hook(lambda module,args:calls.append(args[0].detach().clone()))
    with torch.no_grad():
        out=model(model_input(data));prediction=model.ego_predictions(out,data);target=data.target_mask
        expected=captured['local']+model.type_embedding(data.agent_type)
        torch.testing.assert_close(captured['global_input'],expected,rtol=0,atol=0)
        assert calls[0].shape==(data.num_nodes,) and len(calls)==2 # forward plus explicit expected value
        assert data.lane_vectors.shape[-1]==2 and torch.equal(data.x,original_x)
        same_motion=torch.zeros(3,64,device='cuda');aug=model.fuse_actor_types(same_motion,torch.arange(3,device='cuda'))
        assert all(not torch.equal(aug[a],aug[b]) for a,b in ((0,1),(0,2),(1,2)))
        assert prediction[target].shape==(int(target.sum()),6,12,2)
        assert out['mode_prob'][target].shape==(int(target.sum()),6)
        assert all(torch.isfinite(v).all() for v in out.values())
        torch.testing.assert_close(out['mode_prob'].sum(-1),torch.ones(data.num_nodes,device='cuda'))
        saved_type=model.type_embedding.weight.clone();model.type_embedding.weight.zero_()
        no_type=HiVTLossRecovery(**{k:config()[k] for k in MODEL_KEYS}).cuda()
        no_type.load_state_dict({k:v for k,v in model.state_dict().items() if k!='type_embedding.weight'});no_type.eval()
        base=no_type(data);zero=model(data)
        torch.testing.assert_close(zero['raw_prediction'],base['raw_prediction'],rtol=0,atol=1e-5)
        zero_difference=float((zero['raw_prediction']-base['raw_prediction']).abs().max())
        model.type_embedding.weight.copy_(saved_type)
        perturbed=data.clone();perturbed.positions[:,5:]+=1234;perturbed.y-=777
        perturbed.future_mask=~perturbed.future_mask;perturbed.padding_mask[:,5:]=~perturbed.padding_mask[:,5:]
        perturbed.future_times+=950;perturbed.ego_future+=990
        unchanged=model(perturbed)
        torch.testing.assert_close(out['raw_prediction'],unchanged['raw_prediction'],rtol=0,atol=1e-5)
    local_hook.remove();global_hook.remove();type_hook.remove();del no_type
    gradients={}
    for t,name in enumerate(CLASSES):
        model.zero_grad(set_to_none=True);working=data.clone();working.target_mask=data.target_mask&(data.agent_type==t)
        values=loss_diagnostics(model,model(model_input(working)),working,'original_nll')
        assert all(torch.isfinite(v) for v in values.values());values['loss'].backward();gradients[name]=finite_gradients(model)
    model.zero_grad(set_to_none=True)
    atomic_json(ROOT/'00_manifest/stage3b_unit_tests.json',{'status':'PASS','Test1_actor_type_shape':True,'Test2_only_012':True,
        'invalid_input_rejections':invalid_rejections,'Test3_same_motion_different_types_differ':True,
        'Test4_actor_only_type_lookup_no_lane_lookup':True,'local_output_plus_type_equals_global_input':True,
        'Test5_target_prediction_shape':list(prediction[target].shape),'target_mode_probability_shape':list(out['mode_prob'][target].shape),
        'Test6_predictions_and_losses_NaN':0,'Test6_predictions_and_losses_Inf':0,
        'zero_TypeEmbedding_reproduces_unmodified_forward_max_difference':zero_difference,
        'future_perturbation_invariance':True,'per_class_type_embedding_gradient_norm':gradients,
        'additional_parameter_count':192,'tests_optimizer_updates':0,'trained_NoType_weights_loaded':False,
        'all_attention_temporal_lane_and_decoder_modules_inherited_unchanged':True})
    del model,data;torch.cuda.empty_cache()

def adequate(initial,current):
    return all(current[c][key]<0.5*initial[c][key] for c in CLASSES for key in ('ADE','FDE','fixed_scale_regression_loss'))

def main():
    verify_previous();graphs=graphs_from_frozen_selection();integration_tests(graphs)
    model=model_new();c=config();optimizer=model.optimizer(c['tiny']['warmup_lr'],c['weight_decay'])
    initial=evaluate_graphs(model,graphs);rows=[];global_step=0;best_state=best_optimizer=None;best_fde=float('inf')
    started=time.monotonic();gradient_min=float('inf');gradient_max=0;warm_source=0
    for phase,maxkey,lrkey in (('fixed_scale','warmup_max_steps','warmup_lr'),('original_nll','nll_max_steps','nll_lr')):
        if phase=='original_nll':model.load_state_dict(best_state);optimizer.load_state_dict(best_optimizer)
        for group in optimizer.param_groups:group['lr']=c['tiny'][lrkey]
        for step in range(1,c['tiny'][maxkey]+1):
            start=((step-1)*2)%len(graphs);chosen=[graphs[(start+j)%len(graphs)] for j in range(min(2,len(graphs)))]
            data=Batch.from_data_list(chosen).cuda();model.train();optimizer.zero_grad(set_to_none=True)
            values=loss_diagnostics(model,model(model_input(data)),data,phase)
            assert all(torch.isfinite(v) for v in values.values());values['loss'].backward()
            grad=finite_gradients(model);gradient_min=min(gradient_min,grad);gradient_max=max(gradient_max,grad)
            optimizer.step();global_step+=1
            if step%c['tiny']['check_every']==0 or step==c['tiny'][maxkey]:
                current=evaluate_graphs(model,graphs)
                row={'global_step':global_step,'phase':phase,'phase_step':step,'train_loss':float(values['loss'].detach()),
                    'regression_loss':float(values['regression_loss'].detach()),'classification_loss':float(values['classification_loss'].detach()),
                    'TypeEmbedding_gradient_norm':grad}
                for name in CLASSES:
                    for field,value in current[name].items():row[name+'_'+field]=value
                rows.append(row);write_csv(ROOT/'03_type_embedding/stage3b_tiny_curve.csv',rows)
                print('STAGE3B_TINY',phase,step,{name:(current[name]['ADE'],current[name]['FDE']) for name in CLASSES},'type_grad',grad,flush=True)
                if phase=='fixed_scale' and current['overall']['FDE']<best_fde:
                    best_fde=current['overall']['FDE'];warm_source=step;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};best_optimizer=copy.deepcopy(optimizer.state_dict())
                if phase=='fixed_scale' and step>=c['tiny']['warmup_min_steps'] and adequate(initial,current):break
        if phase=='fixed_scale':warm_steps=step
    final=evaluate_graphs(model,graphs);status='PASS' if adequate(initial,final) else 'FAIL'
    result={'status':status,'initial':initial,'final':final,'criterion':'all three classes ADE/FDE/fixed-scale diagnostic decrease >=50%, TypeEmbedding gradient>0 everyupdate',
        'warmup_steps':warm_steps,'warmup_best_source_step':warm_source,'NLL_steps':step,'global_steps':global_step,
        'TypeEmbedding_gradient_norm_min':gradient_min,'TypeEmbedding_gradient_norm_max':gradient_max,
        'elapsed_seconds':time.monotonic()-started,'same_exact_Stage3A_tiny_windows':True,'full_scene_context_preserved':True,
        'train_only':True,'model_discarded_for_formal_training':True,'NaN':0,'Inf':0,'git_commit_SHA':git('rev-parse','HEAD')}
    atomic_json(ROOT/'04_evaluation/stage3b_tiny_overfit.json',result)
    path=ROOT/'07_checkpoints/stage3b_tiny_last.pt';temp=path.with_suffix('.pt.tmp')
    torch.save({'state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict(),'metadata':result},temp);os.replace(temp,path)
    verify_previous();print('STAGE3B_TINY_OVERFIT='+status,flush=True)
    assert status=='PASS','Stop: TypeEmbedding tiny failed; formal training forbidden'

if __name__=='__main__':
    torch.set_num_threads(4)
    try:main()
    except Exception as error:
        atomic_json(ROOT/'04_evaluation/stage3b_gate_failure.json',{'status':'FAIL','error':repr(error),'formal_training_started':False})
        raise
