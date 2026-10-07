"""Stage7A-isolated exact copy of canonical Stage3B full VAL metrics."""
import csv,os
from pathlib import Path
import numpy as np
import torch
from stage3b_common import Accumulator,CLASSES,HORIZONS,METRICS,errors_with_top1,model_input,GT_fingerprint
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
        for batch in DataLoader(dataset,batch_size=16,shuffle=False,num_workers=0):
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
            if progress and seen%200<16:print('VAL_EVALUATE',seen,'/',len(dataset),flush=True)
    finally:
        if handle:handle.close()
        dataset.clear()
    assert seen==len(dataset)
    if path:os.replace(temp,path)
    return {'metrics':acc.summary(),'scenes':{token:a.summary() for token,a in scenes.items()},'windows':seen,
        'candidate_windows':len(dataset.all_rows),'empty_supervision_windows':len(dataset.empty_rows),'test_used':False,
        'metric_definition':'ADE of best-FDE mode; MR endpoint>2m; Top1 argmax probability; NLL original best-summed-L2 mode, per-actor valid-time coordinate density mean',
        'K':6,'split':dataset.split,'type_embedding':True,'semantic_lane_residual':True}
