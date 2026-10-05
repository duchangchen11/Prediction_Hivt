"""Stage3B-only initialization, frozen-data checks and unchanged VAL definitions."""
import csv
import hashlib
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'00_manifest'))
from stage3_common import (PROJECT,PREVIOUS,CLASSES,GROUPS,HORIZONS,METRICS,MODEL_KEYS,
    SceneDataset,SceneSampler,Accumulator,errors_with_top1,loss_diagnostics,
    atomic_json,read_json,sha256,git,write_csv,seed_all,HiVTLossRecovery)
from stage3b_model import HiVTTypeEmbedding
import numpy as np
import torch
import yaml

CONFIG=ROOT/'00_manifest/stage3b_config.yaml'
BASE_CONFIG=ROOT/'00_manifest/stage3_config.yaml'
FREEZE=ROOT/'00_manifest/stage3b_frozen_stage3a.json'
PREREG=ROOT/'00_manifest/stage3b_preregistration.json'
CACHE=ROOT/'02_preprocessed/stage3_cache'
BEST=ROOT/'07_checkpoints/stage3b_best_overall_minfde.pt'
LAST=ROOT/'07_checkpoints/stage3b_last_checkpoint.pt'
SUMMARY=ROOT/'03_type_embedding/stage3b_training_summary.json'
CURVE=ROOT/'03_type_embedding/stage3b_training_curve.csv'
ACTORS=ROOT/'04_evaluation/stage3b_type_embedding_actor_errors.csv'
BASE_ACTORS=ROOT/'04_evaluation/stage3a_final_frozen_actor_errors.csv'
GT_LEDGER=CACHE/'stage3b_frozen_no_type_GT_ledger.csv'

def config():return yaml.safe_load(CONFIG.read_text())

def state_digest(state):
    digest=hashlib.sha256()
    for name,t in sorted(state.items()):
        digest.update(name.encode());digest.update(str(t.dtype).encode());digest.update(str(tuple(t.shape)).encode())
        digest.update(t.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()

def verify_previous(shards=False):
    frozen=read_json(FREEZE)
    for name,digest in frozen['files'].items():assert sha256(PROJECT/name)==digest,'Frozen file changed: '+name
    if shards:
        for name,digest in frozen['scene_shards'].items():assert sha256(ROOT/name)==digest,'Frozen scene shard changed: '+name
    return frozen

def model_new(device='cuda', audit=False):
    c=config();base=yaml.safe_load(BASE_CONFIG.read_text())
    assert {k:v for k,v in c.items() if k!='type_embedding'}=={k:v for k,v in base.items() if k!='type_embedding'}
    assert c['type_embedding'] and not base['type_embedding']
    seed_all(c['seed']);no_type=HiVTLossRecovery(**{k:c[k] for k in MODEL_KEYS})
    shared={name:t.detach().clone() for name,t in no_type.state_dict().items()}
    shared_parameters={name:t.detach().clone() for name,t in no_type.named_parameters()}
    canonical_rng=torch.get_rng_state().clone()
    model=HiVTTypeEmbedding(**{k:c[k] for k in MODEL_KEYS})
    missing,unexpected=model.load_state_dict(shared,strict=False)
    assert missing==['type_embedding.weight'] and not unexpected
    parameters=dict(model.named_parameters())
    assert all(torch.equal(t,parameters[name]) for name,t in shared_parameters.items())
    assert sum(t.numel() for t in shared_parameters.values())==645809
    assert sum(p.numel() for p in model.parameters())==646001
    assert all(torch.equal(t,model.state_dict()[name]) for name,t in shared.items())
    init={'status':'PASS','seed':c['seed'],'shared_parameter_count':645809,'shared_parameter_tensor_count':len(shared_parameters),
        'shared_parameter_hash':state_digest(shared_parameters),'shared_state_hash':state_digest(shared),
        'max_shared_parameter_diff':0.0,'shared_buffers_bitwise_equal':True,
        'type_embedding_shape':[3,64],'type_embedding_init_std':float(model.type_embedding.weight.detach().std()),
        'type_embedding_init':'nn.Embedding default normal; deterministic post-shared-construction RNG stream',
        'type_embedding_initial_hash':state_digest({'type_embedding.weight':model.type_embedding.weight}),
        'additional_parameter_count':192,'total_parameter_count':646001,
        'source':'fresh seed2022 No-Type constructor; no trained checkpoint loaded',
        'NoType_trained_checkpoint_loaded':False,'training_CPU_RNG_restored_to_NoType_post_constructor_state':True}
    torch.set_rng_state(canonical_rng)
    del no_type,shared,shared_parameters
    if audit:atomic_json(ROOT/'00_manifest/stage3b_initialization_audit.json',init)
    return model.to(device)

def model_input(data):
    working=data.clone()
    HiVTTypeEmbedding.validate_actor_types(working.agent_type,working.num_nodes)
    return working

def GT_fingerprint(positions,future_mask,agent_type):
    positions=positions.detach().cpu().contiguous();future_mask=future_mask.detach().cpu().contiguous()
    digest=hashlib.sha256()
    digest.update(str(positions.dtype).encode());digest.update(str(tuple(positions.shape)).encode())
    digest.update(positions.numpy().tobytes())
    return {'GT_trajectory_sha256':digest.hexdigest(),
            'future_mask_bits':''.join('1' if x else '0' for x in future_mask.tolist()),'agent_type_id':int(agent_type)}

@torch.no_grad()
def evaluate(dataset,model,phase='original_nll',actor_path=None,progress=False):
    from torch_geometric.loader import DataLoader
    model.eval();acc=Accumulator();scenes={r['scene_token']:Accumulator() for r in dataset.all_rows};seen=0
    fields=['scene_name','scene_token','sample_token','instance_token','node_in_graph','horizon','agent_type',
            'motion_state','valid_future_steps','GT_endpoint_displacement_m',*METRICS,'best_mode','top1_mode',
            'GT_trajectory_sha256','future_mask_bits','agent_type_id']
    path=Path(actor_path) if actor_path else None;temp=path.with_name(path.name+'.tmp') if path else None
    handle=open(temp,'w',newline='') if temp else None;writer=csv.DictWriter(handle,fields,lineterminator='\n') if handle else None
    if writer:writer.writeheader()
    try:
        for batch in DataLoader(dataset,batch_size=config()['batch_size'],shuffle=False,num_workers=0):
            data=batch.cuda();out=model(model_input(data));_,errors=errors_with_top1(model,out,data)
            assert torch.isfinite(out['raw_prediction']).all() and torch.isfinite(out['mode_prob']).all()
            keys=('minADE_K','minFDE_K','MR_K','Top1ADE6','Top1FDE6','NLL','independent_minADE_K')
            values=torch.stack([errors[k] for k in keys],-1).cpu().numpy()
            types=data.agent_type.cpu().numpy();graph_index=data.batch.cpu().numpy();ptr=data.ptr.cpu().numpy()
            valid=errors['valid_steps'].cpu().numpy()
            displacement=torch.linalg.vector_norm(data.y[torch.arange(data.num_nodes,device=data.y.device),errors['last_valid_index'].clamp(min=0)],dim=-1).cpu().numpy()
            best=errors['best_mode'].cpu().numpy();top=errors['top1_mode'].cpu().numpy()
            if writer:gt=data.positions[:,5:].cpu();mask=data.future_mask.cpu()
            for horizon in HORIZONS:
                selected=errors[horizon].cpu().numpy();assert np.isfinite(values[selected]).all()
                for node in np.flatnonzero(selected):
                    graph=int(graph_index[node]);local=int(node-ptr[graph]);cls=CLASSES[int(types[node])]
                    motion=data.t0_motion_state[graph][local];groups=['overall',cls]
                    if cls=='vehicle':groups.append(motion)
                    token=data.scene_token[graph]
                    for group in groups:acc.add(horizon,group,values[node]);scenes[token].add(horizon,group,values[node])
                    if writer:
                        row=dict(zip(fields,[data.scene_name[graph],token,data.sample_token[graph],data.instance_tokens[graph][local],local,horizon,
                            cls,motion,int(valid[node]),float(displacement[node]),*values[node].tolist(),int(best[node]),int(top[node])]))
                        row.update(GT_fingerprint(gt[node],mask[node],types[node]));writer.writerow(row)
            seen+=batch.num_graphs
            if progress and seen%200<config()['batch_size']:print('VAL_EVALUATE',seen,'/',len(dataset),flush=True)
    finally:
        if handle:handle.close()
        dataset.clear()
    assert seen==len(dataset)
    if path:os.replace(temp,path)
    return {'metrics':acc.summary(),'scenes':{token:a.summary() for token,a in scenes.items()},'windows':seen,
        'candidate_windows':len(dataset.all_rows),'empty_supervision_windows':len(dataset.empty_rows),'test_used':False,
        'metric_definition':'ADE of best-FDE mode; MR endpoint>2m; Top1 argmax probability; NLL original best-summed-L2 mode, per-actor valid-time coordinate density mean',
        'K':6,'split':dataset.split,'type_embedding':True}

@torch.no_grad()
def evaluate_graphs(model,graphs):
    from torch_geometric.data import Batch
    model.eval();sums={name:np.zeros(6) for name in ('overall',)+CLASSES};diagnostic={name:[] for name in CLASSES}
    for graph in graphs:
        data=Batch.from_data_list([graph]).cuda();out=model(model_input(data));_,errors=errors_with_top1(model,out,data)
        values=loss_diagnostics(model,out,data,'fixed_scale')
        for t,name in enumerate(CLASSES):
            mask=data.target_mask&(data.agent_type==t)
            if mask.any():diagnostic[name].append(float(values[name+'_regression_loss']))
            for i in torch.where(mask)[0].tolist():
                row=np.array([1,float(errors['minADE_K'][i]),float(errors['minFDE_K'][i]),float(errors['Top1ADE6'][i]),float(errors['Top1FDE6'][i]),float(errors['NLL'][i])])
                assert np.isfinite(row).all();sums[name]+=row;sums['overall']+=row
    return {name:{'count':int(s[0]),**{k:float(s[j]/s[0]) if s[0] else None for j,k in enumerate(('ADE','FDE','Top1ADE','Top1FDE','NLL'),start=1)},
        'fixed_scale_regression_loss':float(np.mean(diagnostic[name])) if name in diagnostic and diagnostic[name] else None} for name,s in sums.items()}
