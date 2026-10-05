"""Restore step18000 exactly; at most 3000 additional original-NLL updates."""
import csv
import math
import os
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '00_manifest'))
from stage3a_plusplus_common import (SOURCE, LAST, NEW_BEST, TRAIN_MANIFEST, FINAL_MANIFEST,
    PREREG, CURVE, CONFIG, CLASSES, SceneDataset, SceneSampler, config, atomic_json,
    read_json, sha256, git, write_csv, verify_previous, model_digest, state_equal)
from stage3_common import model_new, model_input, loss_diagnostics, evaluate
import torch
from torch_geometric.loader import DataLoader

def save_checkpoint(path, model, optimizer, metadata, state, iterator):
    temp = path.with_suffix('.pt.tmp')
    torch.save({'state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                'metadata': metadata, 'plusplus_state': dict(state), 'iterator': dict(iterator),
                'config': config(), 'torch_rng': torch.get_rng_state(),
                'cuda_rng': torch.cuda.get_rng_state_all()}, temp)
    os.replace(temp, path)

def main():
    c = config(); verify_previous(); prereg = read_json(PREREG)
    for name, digest in prereg['source_sha256'].items():
        assert sha256(ROOT / name) == digest, 'Registered source changed: ' + name
    assert git('branch', '--show-current') == 'stage3/multitype-hivt'
    assert c['batch_size'] == 16 and c['nll']['lr'] == 1e-4
    assert [c[k] for k in ('historical_steps','future_steps','num_modes','embed_dim')] == [5,12,6,64]
    source = torch.load(SOURCE, map_location='cpu', weights_only=False)
    original = source['metadata']; source_sha = sha256(SOURCE)
    assert source_sha == prereg['source_checkpoint_sha256']
    assert source['config'] == c and original['config_sha256'] == sha256(CONFIG)
    assert original['global_step'] == 18000 and original['phase'] == 'original_nll'
    assert original['selection_metric'] == 'overall minFDE6'
    if TRAIN_MANIFEST.exists() and read_json(TRAIN_MANIFEST)['status'] == 'COMPLETE':
        print('FINAL_NO_TYPE_BUDGET_ALREADY_COMPLETE', flush=True); return
    state = {'extension_step': 0, 'best_FDE': original['validation_FDE'], 'best_step': 18000,
             'bad_validations': 0, 'best_refreshed': False, 'completed': False}
    saved = source; iterator = dict(source['iterator']); rows = []
    if LAST.exists():
        saved = torch.load(LAST, map_location='cpu', weights_only=False)
        assert saved['metadata']['plusplus_source_checkpoint_sha256'] == source_sha
        state = dict(saved['plusplus_state']); iterator = dict(saved['iterator'])
        if CURVE.exists():
            with CURVE.open() as f:
                rows = [{k: float(v) for k,v in row.items()} for row in csv.DictReader(f)
                        if int(row['extension_step']) <= state['extension_step']]
    model = model_new(); model.load_state_dict(saved['state_dict'])
    assert model_digest(model.state_dict()) == model_digest(saved['state_dict'])
    optimizer = model.optimizer(c['nll']['lr'], c['weight_decay'])
    optimizer.load_state_dict(saved['optimizer_state_dict'])
    initial_groups = [{k:v for k,v in g.items() if k != 'params'}
                      for g in source['optimizer_state_dict']['param_groups']]
    assert [{k:v for k,v in g.items() if k != 'params'} for g in optimizer.param_groups] == initial_groups
    restored = optimizer.state_dict()
    for key, entry in saved['optimizer_state_dict']['state'].items():
        for name, value in entry.items():
            actual = restored['state'][key][name]
            assert torch.equal(actual.cpu(),value.cpu()) if torch.is_tensor(value) else actual == value
    torch.set_rng_state(saved['torch_rng']); torch.cuda.set_rng_state_all(saved['cuda_rng'])
    assert torch.equal(torch.get_rng_state(),saved['torch_rng'])
    assert all(torch.equal(a.cpu(),b.cpu()) for a,b in zip(torch.cuda.get_rng_state_all(),saved['cuda_rng']))
    train = SceneDataset('train'); val = SceneDataset('val'); sampler = SceneSampler(train,c['seed'])
    steps_per_epoch = math.ceil(len(train)/c['batch_size'])
    assert 0 <= iterator['next_batch'] <= steps_per_epoch and iterator['epoch'] >= 0
    if not state['extension_step']:
        atomic_json(ROOT/'00_manifest/stage3a_plusplus_resume_audit.json', {
            'status':'PASS', 'source_checkpoint':str(SOURCE.relative_to(ROOT)), 'source_global_step':18000,
            'checkpoint_SHA256':source_sha, 'model_SHA256':model_digest(model.state_dict()),
            'optimizer_restored':True, 'optimizer_state_bitwise_equal':True,
            'RNG_restored':True, 'CPU_RNG_bitwise_equal':True, 'CUDA_RNG_bitwise_equal':True,
            'cursor_restored':True, 'sampler_cursor':dict(iterator), 'sampler_seed':c['seed'],
            'sampler_epoch_offset':100000, 'first_recreated_loader_CPU_RNG_draw_cancelled':True,
            'LR':1e-4, 'batch_size':16, 'loss_mode':'original learnable-scale Laplace NLL',
            'config_SHA256':sha256(CONFIG), 'optimizer_param_groups_unchanged':True,
            'new_warmup':False, 'from_scratch':False})
    manifest = read_json(TRAIN_MANIFEST) if TRAIN_MANIFEST.exists() else {
        'status':'RUNNING', 'source_checkpoint_relative_path':str(SOURCE.relative_to(ROOT)),
        'source_checkpoint_sha256':source_sha, 'start_global_step':18000,
        'maximum_extra_optimizer_steps':3000, 'maximum_global_step':21000,
        'validation_interval':500, 'patience':5, 'config_sha256':sha256(CONFIG),
        'training_code_git_commit':git('rev-parse','HEAD'), 'initial_iterator':source['iterator'],
        'selection_rule':'full-horizon official VAL overall minFDE6; strict improvement',
        'Type_Embedding':False, 'test_used':False, 'LR':1e-4, 'batch_size':16,
        'loss_mode':'original learnable-scale Laplace NLL'}
    atomic_json(TRAIN_MANIFEST,manifest)
    print('RESTORED_18000',iterator,'extension_step',state['extension_step'],flush=True)
    started=time.monotonic(); resumed_loader=True
    while state['extension_step'] < 3000 and not state['completed']:
        sampler.epoch=iterator['epoch']+100000; order=list(sampler)
        batches=[order[i:i+c['batch_size']] for i in range(0,len(order),c['batch_size'])]
        start_batch=iterator['next_batch']
        loader=DataLoader(train,batch_sampler=batches[start_batch:],num_workers=0)
        rng=torch.get_rng_state(); loader_iterator=iter(loader)
        if resumed_loader: torch.set_rng_state(rng)
        resumed_loader=False
        for batch_index,batch in enumerate(loader_iterator,start=start_batch):
            model.train(); data=batch.cuda(); optimizer.zero_grad(set_to_none=True)
            output=model(model_input(data)); values=loss_diagnostics(model,output,data,'original_nll')
            assert all(torch.isfinite(v) for v in values.values()), 'Nonfinite loss'
            values['loss'].backward()
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None), 'Nonfinite gradient'
            optimizer.step(); state['extension_step']+=1; iterator['next_batch']=batch_index+1
            step=state['extension_step']; global_step=18000+step
            assert global_step <= 21000
            if step%100 == 0: print('NLL_STEP',step,'global',global_step,flush=True)
            if step%500 == 0:
                measured=evaluate(val,model,phase='original_nll')
                assert len(measured['scenes']) == 150 and measured['windows'] == 3603
                full=measured['metrics']['full_horizon']; fde=full['overall']['minFDE6']
                improved=fde < state['best_FDE']
                state['bad_validations']=0 if improved else state['bad_validations']+1
                if improved: state.update(best_FDE=fde,best_step=global_step,best_refreshed=True)
                state['completed']=state['bad_validations'] >= 5 or global_step == 21000
                row={'extension_step':step,'global_step':global_step,'learning_rate':optimizer.param_groups[0]['lr'],
                     'best_overall_FDE':state['best_FDE'],'best_step':state['best_step'],
                     'improved':int(improved),'consecutive_nonimprovements':state['bad_validations']}
                for name in ('overall',)+CLASSES+('vehicle.moving',):
                    assert all(v is not None and math.isfinite(v) for v in full[name].values())
                    for metric,short in (('count','count'),('minADE6','ADE'),('minFDE6','FDE'),('MR6','MR'),
                                         ('Top1ADE6','Top1ADE'),('Top1FDE6','Top1FDE'),('NLL','NLL')):
                        row['VAL_'+name+'_'+short]=full[name][metric]
                metadata={**original,'epoch':iterator['epoch']+iterator['next_batch']/steps_per_epoch,
                    'global_step':global_step,'phase_step':original['phase_step']+step,'plusplus_step':step,
                    'validation_ADE':full['overall']['minADE6'],'validation_FDE':fde,
                    'validation_MR':full['overall']['MR6'],'moving_ADE':full['vehicle.moving']['minADE6'],
                    'moving_FDE':full['vehicle.moving']['minFDE6'],'git_commit_SHA':manifest['training_code_git_commit'],
                    'plusplus_source_checkpoint_sha256':source_sha,'optimizer_state_preserved':True,
                    'per_class_full_horizon_metrics':{name:full[name] for name in CLASSES},
                    'full_horizon_metrics':full}
                measured.update(checkpoint_metadata=metadata,plusplus_step=step)
                atomic_json(ROOT/f'04_evaluation/stage3a_plusplus_val_step_{global_step:05d}.json',measured)
                if improved: save_checkpoint(NEW_BEST,model,optimizer,metadata,state,iterator)
                save_checkpoint(LAST,model,optimizer,metadata,state,iterator)
                manifest.update(executed_extra_steps=step,final_executed_global_step=global_step,
                    best_refreshed=state['best_refreshed'],final_best_global_step=state['best_step'],
                    consecutive_nonimprovements=state['bad_validations'],final_iterator=dict(iterator),
                    last_checkpoint_sha256=sha256(LAST),NaN=0,Inf=0)
                atomic_json(TRAIN_MANIFEST,manifest); rows.append(row); write_csv(CURVE,rows)
                print('FULL_VAL',row,flush=True)
                if state['completed']: break
        if not state['completed']: iterator['epoch']+=1; iterator['next_batch']=0
    patience=state['bad_validations'] >= 5; budget=18000+state['extension_step'] >= 21000
    manifest.update(status='COMPLETE',stop_reason='patience_5' if patience else 'global_step_21000_budget',
                    converged_by_patience=patience,stopped_by_budget=budget,
                    elapsed_seconds_this_invocation=time.monotonic()-started,stage3a_frozen='YES')
    chosen=NEW_BEST if state['best_refreshed'] else SOURCE
    chosen_saved=torch.load(chosen,map_location='cpu',weights_only=False)
    if state['best_refreshed']: full=chosen_saved['metadata']['full_horizon_metrics']
    else: full=read_json(ROOT/'04_evaluation/stage3a_plus_best_val_metrics.json')['metrics']['full_horizon']
    final={**manifest,'checkpoint_relative_path':str(chosen.relative_to(ROOT)),
        'checkpoint_sha256':sha256(chosen),'model_state_content_sha256':model_digest(chosen_saved['state_dict']),
        'final_best_global_step':state['best_step'],'overall_minADE6':full['overall']['minADE6'],
        'overall_minFDE6':full['overall']['minFDE6'],'overall_MR6':full['overall']['MR6'],
        'vehicle_minFDE6':full['vehicle']['minFDE6'],'pedestrian_minFDE6':full['pedestrian']['minFDE6'],
        'bicycle_minFDE6':full['bicycle']['minFDE6'],'vehicle_moving_minFDE6':full['vehicle.moving']['minFDE6'],
        'selected_full_horizon_metrics':full,'checkpoint_valid':True,
        'official_VAL_fresh_evaluation_complete':False,
        'relative_FDE_improvement_vs_18000':1-full['overall']['minFDE6']/original['validation_FDE'],
        'future_ablation_protocol':{'maximum_global_step':21000,'validation_interval':500,
                                   'checkpoint_selection':'full-horizon official VAL overall minFDE6; strict improvement'},
        'further_No_Type_training_allowed':False}
    verify_previous(); atomic_json(TRAIN_MANIFEST,manifest); atomic_json(FINAL_MANIFEST,final)
    print('FINAL_NO_TYPE_BUDGET_COMPLETE',final,flush=True)

if __name__ == '__main__':
    torch.set_num_threads(4)
    main()
