"""From-scratch, unchanged Protocol1; fixed5000 warm-up and hard21000 total budget."""
import csv
import math
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'00_manifest'))
from stage4a_common import (CONFIG,PREREG,FREEZE,BEST,LAST,SUMMARY,CURVE,CLASSES,GROUPS,
    SceneDataset,SceneSampler,config,model_new,model_input,loss_diagnostics,evaluate,
    atomic_json,read_json,write_csv,sha256,git,verify_previous,state_digest)
import torch
from torch_geometric.loader import DataLoader

WARM_BEST=ROOT/'07_checkpoints/stage4a_warmup_best_overall.pt'

def checkpoint_save(path,model,optimizer,metadata,state,iterator):
    temp=path.with_suffix('.pt.tmp')
    torch.save({'state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict(),
        'metadata':metadata,'phase_state':dict(state),'iterator':dict(iterator),'config':config(),
        'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},temp)
    os.replace(temp,path)

def restore(path):
    saved=torch.load(path,map_location='cpu',weights_only=False)
    assert saved['config']==config() and saved['metadata']['config_sha256']==sha256(CONFIG)
    model=model_new();model.load_state_dict(saved['state_dict'],strict=True)
    optimizer=model.optimizer(saved['metadata']['LR'],config()['weight_decay'])
    optimizer.load_state_dict(saved['optimizer_state_dict'])
    assert state_digest(model.state_dict())==state_digest(saved['state_dict'])
    torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
    return model,optimizer,saved

def recover_rows():
    if not CURVE.exists():return []
    with CURVE.open() as f:return [{k:(v if k=='phase' else float(v)) for k,v in row.items()} for row in csv.DictReader(f)]

def train_phase(phase,training_commit):
    c=config();warm=phase=='fixed_scale';label='warmup' if warm else 'nll';lr=c['warmup' if warm else 'nll']['lr']
    budget=5000 if warm else 16000;offset=0 if warm else 5000
    last_path=ROOT/f'07_checkpoints/stage4a_{label}_last_checkpoint.pt'
    summary_path=ROOT/f'03_type_interaction/stage4a_{label}_summary.json'
    if summary_path.exists() and read_json(summary_path)['status']=='COMPLETE':return read_json(summary_path)
    rows=recover_rows();rows=[r for r in rows if r['global_step']<=offset or r['phase']==phase]
    state={'phase_step':0,'best_FDE':None,'best_step':None,'bad_validations':0,'completed':False}
    iterator={'epoch':0,'next_batch':0};resuming=False;warm_source=0
    if last_path.exists():
        model,optimizer,saved=restore(last_path);state=dict(saved['phase_state']);iterator=dict(saved['iterator'])
        warm_source=saved['metadata']['warmup_best_source_step'];resuming=state['phase_step']>0
        rows=[r for r in rows if r['global_step']<=offset+state['phase_step']]
        if 'last_curve_row' in state:
            recovered=state['last_curve_row']
            rows=[r for r in rows if r['global_step']!=recovered['global_step']]+[recovered]
            rows.sort(key=lambda r:r['global_step']);write_csv(CURVE,rows)
    elif warm:
        model=model_new();optimizer=model.optimizer(lr,c['weight_decay'])
    else:
        model,optimizer,saved=restore(WARM_BEST);warm_source=saved['metadata']['global_step']
        source_groups=[{k:v for k,v in group.items() if k not in ('lr','params')} for group in optimizer.param_groups]
        for group in optimizer.param_groups:group['lr']=lr
        assert source_groups==[{k:v for k,v in group.items() if k not in ('lr','params')} for group in optimizer.param_groups]
        atomic_json(ROOT/'00_manifest/stage4a_phase_transition_audit.json',{
            'status':'PASS','global_NLL_start':5000,'own_warmup_best_source_step':warm_source,
            'own_warmup_checkpoint_sha256':sha256(WARM_BEST),'optimizer_state_preserved':True,
            'model_RNG_restored':True,'NLL_sampler_cursor_reset_like_Stage3A':iterator,
            'only_authorized_phase_LR_change':{'from':0.001,'to':0.0001},
            'Stage3B_trained_weights_loaded':False,'Protocol1_Stage3B_transition_reused':True})
    assert all(group['lr']==lr for group in optimizer.param_groups)
    train=SceneDataset('train');val=SceneDataset('val');sampler=SceneSampler(train,c['seed'])
    steps_per_epoch=math.ceil(len(train)/16);assert len(train)==16898 and len(val)==3603 and steps_per_epoch==1057
    running={k:0.0 for k in ('loss','regression_loss','classification_loss','NLL')};running_count=0
    grad_min=float('inf');grad_max=0.0;relation_grad_min={n:float('inf') for n in ('pair','first','final')};relation_grad_max={n:0.0 for n in relation_grad_min};started=time.monotonic()
    print('PHASE_START',label,'step',state['phase_step'],'cursor',iterator,'warm_source',warm_source,flush=True)
    while state['phase_step']<budget and not state['completed']:
        sampler.epoch=iterator['epoch']+(0 if warm else 100000);order=list(sampler)
        batches=[order[i:i+16] for i in range(0,len(order),16)];start_batch=iterator['next_batch']
        loader=DataLoader(train,batch_sampler=batches[start_batch:],num_workers=0)
        rng=torch.get_rng_state();loader_iterator=iter(loader)
        if resuming:torch.set_rng_state(rng)
        resuming=False
        for batch_index,batch in enumerate(loader_iterator,start=start_batch):
            model.train();data=batch.cuda();optimizer.zero_grad(set_to_none=True)
            values=loss_diagnostics(model,model(model_input(data)),data,phase)
            assert all(torch.isfinite(v) for v in values.values()),'Nonfinite loss'
            values['loss'].backward()
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None),'Nonfinite gradient'
            grad=float(model.type_embedding.weight.grad.norm())
            assert math.isfinite(grad) and grad>0,'TypeEmbedding is disconnected or has nonfinite gradient'
            grad_min=min(grad_min,grad);grad_max=max(grad_max,grad)
            gi=model.global_interactor
            for tag,param in (('pair',gi.pair_embedding.weight),('first',gi.relation_mlp[0].weight),('final',gi.relation_mlp[-1].weight)):
                assert param.grad is not None and torch.isfinite(param.grad).all()
                norm=float(param.grad.norm());assert math.isfinite(norm)
                relation_grad_min[tag]=min(relation_grad_min[tag],norm);relation_grad_max[tag]=max(relation_grad_max[tag],norm)
            if warm and state['phase_step']==9:
                assert all(v>0 for v in relation_grad_max.values()),'Relation adapter has no gradient within first10 formal updates'
            optimizer.step();state['phase_step']+=1;iterator['next_batch']=batch_index+1
            for key in running:running[key]+=float(values[key].detach())
            running_count+=1;global_step=offset+state['phase_step'];assert global_step<=21000
            if state['phase_step']%100==0:print('TRAIN_STEP',label,'global',global_step,'type_gradient',grad,flush=True)
            if state['phase_step']%500==0:
                measured=evaluate(val,model,phase);assert len(measured['scenes'])==150 and measured['windows']==3603
                full=measured['metrics']['full_horizon'];fde=full['overall']['minFDE6']
                improved=state['best_FDE'] is None or fde<state['best_FDE']
                state['bad_validations']=0 if improved else state['bad_validations']+1
                if improved:state.update(best_FDE=fde,best_step=global_step)
                state['completed']=state['phase_step']>=budget or (not warm and state['bad_validations']>=5)
                row={'global_step':global_step,'phase':phase,'phase_step':state['phase_step'],'learning_rate':lr,
                     'train_loss':running['loss']/running_count,'regression_loss':running['regression_loss']/running_count,
                     'classification_loss':running['classification_loss']/running_count,'NLL':running['NLL']/running_count,
                     'TypeEmbedding_gradient_norm_min':grad_min,'TypeEmbedding_gradient_norm_max':grad_max,
                     'best_step':state['best_step'],'best_overall_FDE':state['best_FDE'],
                     'improved':int(improved),'consecutive_nonimprovements':state['bad_validations']}
                for tag in relation_grad_min:
                    row['Relation_'+tag+'_gradient_norm_min']=relation_grad_min[tag]
                    row['Relation_'+tag+'_gradient_norm_max']=relation_grad_max[tag]
                for group in GROUPS:
                    assert all(v is not None and math.isfinite(v) for v in full[group].values())
                    for metric,tag in (('count','count'),('minADE6','ADE'),('minFDE6','FDE'),('MR6','MR'),
                                       ('Top1ADE6','Top1ADE'),('Top1FDE6','Top1FDE'),('NLL','NLL')):
                        row['VAL_'+group+'_'+tag]=full[group][metric]
                metadata={'phase':phase,'global_step':global_step,'phase_step':state['phase_step'],
                    'epoch':iterator['epoch']+iterator['next_batch']/steps_per_epoch,'LR':lr,
                    'validation_ADE':full['overall']['minADE6'],'validation_FDE':fde,'validation_MR':full['overall']['MR6'],
                    'moving_ADE':full['vehicle.moving']['minADE6'],'moving_FDE':full['vehicle.moving']['minFDE6'],
                    'full_horizon_metrics':full,'warmup_executed_steps':5000 if not warm else state['phase_step'],
                    'warmup_best_source_step':warm_source,'config_sha256':sha256(CONFIG),
                    'git_commit_SHA':training_commit,'selection_metric':'overall minFDE6',
                    'selection_split':'official val','selection_horizon':'full_horizon','type_embedding':True,
                    'additional_parameter_count':1608,'parameter_count':647609,'type_conditioned_interaction':True,'optimizer_state_preserved':not warm}
                measured.update(checkpoint_metadata=metadata)
                state['last_curve_row']=dict(row)
                atomic_json(ROOT/f'04_evaluation/stage4a_val_step_{global_step:05d}.json',measured)
                if improved:checkpoint_save(WARM_BEST if warm else BEST,model,optimizer,metadata,state,iterator)
                checkpoint_save(last_path,model,optimizer,metadata,state,iterator)
                checkpoint_save(LAST,model,optimizer,metadata,state,iterator)
                # Persist each curve row only once; VAL JSON supports recovery after interrupted writes.
                rows=[r for r in rows if r['global_step']!=global_step];rows.append(row);rows.sort(key=lambda r:r['global_step'])
                write_csv(CURVE,rows)
                running={key:0.0 for key in running};running_count=0;grad_min=float('inf');grad_max=0.0;relation_grad_min={n:float('inf') for n in relation_grad_min};relation_grad_max={n:0.0 for n in relation_grad_max}
                print('FULL_VAL',label,'global',global_step,'FDE',fde,'best',state['best_step'],state['best_FDE'],'bad',state['bad_validations'],flush=True)
                if state['completed']:break
        if not state['completed']:iterator['epoch']+=1;iterator['next_batch']=0
    patience=not warm and state['bad_validations']>=5;budget_stop=not warm and offset+state['phase_step']==21000
    summary={'status':'COMPLETE','phase':phase,'phase_steps':state['phase_step'],'global_executed_step':offset+state['phase_step'],
        'best_global_step':state['best_step'],'best_overall_FDE':state['best_FDE'],
        'consecutive_nonimprovements':state['bad_validations'],'converged_by_patience':patience,'stopped_by_budget':budget_stop,
        'stop_reason':'warmup_fixed5000' if warm else ('patience_5' if patience else 'global21000_budget'),
        'warmup_best_source_step':warm_source,'final_iterator':iterator,'last_checkpoint_sha256':sha256(last_path),
        'elapsed_seconds_this_invocation':time.monotonic()-started,'NaN':0,'Inf':0,'training_code_git_commit':training_commit}
    atomic_json(summary_path,summary);del model,optimizer,train,val;torch.cuda.empty_cache()
    print('PHASE_COMPLETE',label,summary,flush=True);return summary

def main():
    prereg=read_json(PREREG);verify_previous()
    assert read_json(ROOT/'00_manifest/stage4a_unit_tests.json')['status']=='PASS'
    assert read_json(ROOT/'04_evaluation/stage4a_tiny_overfit.json')['status']=='PASS'
    assert read_json(ROOT/'04_evaluation/stage4a_gradient_audit.json')['status']=='PASS'
    assert read_json(ROOT/'00_manifest/stage4a_neutral_initialization_audit.json')['status']=='PASS'
    assert read_json(ROOT/'04_evaluation/stage4a_interaction_subgroup_audit.json')['registered_before_formal_training']
    assert read_json(ROOT/'00_manifest/stage4a_initialization_audit.json')['max_shared_parameter_diff']==0
    for name,digest in read_json(ROOT/'00_manifest/stage4a_training_sources.json')['training_source_sha256'].items():assert sha256(ROOT/name)==digest,'Registered training source changed'
    assert sha256(CONFIG)==prereg['Stage4A_config_sha256'] and git('branch','--show-current')=='stage4a/type-conditioned-interaction'
    if SUMMARY.exists() and read_json(SUMMARY)['status']=='COMPLETE':print('STAGE4A_ALREADY_COMPLETE');return
    commit=read_json(ROOT/'00_manifest/stage4a_training_registration.json')['training_code_git_commit']
    warm=train_phase('fixed_scale',commit);nll=train_phase('original_nll',commit)
    saved=torch.load(BEST,map_location='cpu',weights_only=False);full=saved['metadata']['full_horizon_metrics']
    summary={'status':'COMPLETE','warmup_steps':5000,'NLL_steps':nll['phase_steps'],
        'final_executed_global_step':nll['global_executed_step'],'best_global_step':saved['metadata']['global_step'],
        'stop_reason':nll['stop_reason'],'converged_by_patience':nll['converged_by_patience'],'stopped_by_budget':nll['stopped_by_budget'],
        'warmup_best_source_step':saved['metadata']['warmup_best_source_step'],
        'training_code_git_commit':commit,'config_sha256':sha256(CONFIG),'NaN':0,'Inf':0,'TypeEmbedding_gradient_verified':True,'RelationBias_gradient_verified':True,
        'checkpoint_relative_path':str(BEST.relative_to(ROOT)),'checkpoint_sha256':sha256(BEST),
        'model_state_content_sha256':state_digest(saved['state_dict']),'selected_full_horizon_metrics':full,
        'selection_rule':'post-update original-NLL full-horizon official VAL overall minFDE6 strict improvement',
        'maximum_global_step':21000,'validation_interval':500,'patience':5,'from_scratch':True,
        'Stage3B_trained_weights_loaded':False,'test_used':False,'Stage4B_executed':False}
    verify_previous();atomic_json(SUMMARY,summary)
    atomic_json(ROOT/'07_checkpoints/stage4a_best_checkpoint_manifest.json',summary)
    print('STAGE4A_FORMAL_TRAINING_COMPLETE',summary,flush=True)

if __name__=='__main__':torch.set_num_threads(4);main()
