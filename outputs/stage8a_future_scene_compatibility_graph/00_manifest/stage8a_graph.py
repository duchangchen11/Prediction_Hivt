"""GT-free observable mode nodes and individual directed future relations."""
from dataclasses import dataclass
import numpy as np
import torch

NODE_FIELDS=('type_vehicle','type_pedestrian','type_bicycle','original_logit','original_probability',
    'history_recent_displacement','history_net_displacement','history_path_length','endpoint_x','endpoint_y',
    'candidate_net_displacement','candidate_path_length','candidate_mean_speed','endpoint_heading_sin','endpoint_heading_cos')
INTERACTION_FIELDS=('future_minimum_distance','closest_timestep_over_Tf','future_mean_distance','endpoint_distance',
    'initial_future_distance','relative_endpoint_heading_sin','relative_endpoint_heading_cos',
    'minimum_relative_step_displacement','closing_distance_over_6s','neighbor_original_probability',
    'target_original_probability','target_vehicle','target_pedestrian','target_bicycle',
    'neighbor_vehicle','neighbor_pedestrian','neighbor_bicycle')

@dataclass(frozen=True)
class ObservableWindow:
    history:torch.Tensor
    history_padding:torch.Tensor
    actor_type:torch.Tensor
    predicted:torch.Tensor
    original_logits:torch.Tensor
    original_probability:torch.Tensor
    scene_token:str
    sample_token:str
    instance_tokens:tuple
    map_location:str
    origin:np.ndarray
    yaw:float

def observable_window(cache,location,origin,yaw):
    # Explicit allowlist. Neither future target mask nor GT can be accessed here.
    return ObservableWindow(cache['history'],cache['history_padding'],cache['agent_type'],cache['ego_prediction'],
        cache['mode_logits'],cache['mode_prob'],cache['scene_token'],cache['sample_token'],
        tuple(cache['instance_tokens']),location,np.asarray(origin,dtype=np.float64),float(yaw))

def endpoint_heading(predicted):
    d=predicted[...,-1,:]-predicted[...,-3,:]
    return torch.atan2(d[...,1],d[...,0])

@torch.no_grad()
def node_features(w):
    h=w.history;pad=w.history_padding;y=w.predicted;n=len(h)
    assert h.shape==(n,5,2) and y.shape==(n,6,12,2) and w.actor_type.shape==(n,)
    assert w.original_logits.shape==w.original_probability.shape==(n,6)
    valid=~pad;prev=torch.zeros_like(h[:,0]);first=prev.clone();seen=torch.zeros(n,dtype=torch.bool)
    recent=h.new_zeros(n);path=recent.clone()
    for t in range(5):
        pair=valid[:,t]&seen;distance=(h[:,t]-prev).norm(dim=-1)
        recent=torch.where(pair,distance,recent);path+=torch.where(pair,distance,0.)
        first=torch.where((valid[:,t]&~seen)[:,None],h[:,t],first)
        prev=torch.where(valid[:,t,None],h[:,t],prev);seen|=valid[:,t]
    net=torch.where(valid.sum(-1)>=2,(prev-first).norm(dim=-1),0.)
    steps=torch.diff(torch.cat((h[:,4,None,None].expand(n,6,1,2),y),dim=2),dim=2).norm(dim=-1)
    out=y.new_zeros((n,6,15));out[:,:,:3]=torch.nn.functional.one_hot(w.actor_type,3)[:,None]
    out[:,:,3]=w.original_logits;out[:,:,4]=w.original_probability
    out[:,:,5:8]=torch.stack((recent,net,path),-1)[:,None];out[:,:,8:10]=y[:,:,-1]
    out[:,:,10]=(y[:,:,-1]-h[:,4,None]).norm(dim=-1);out[:,:,11]=steps.sum(-1);out[:,:,12]=steps.sum(-1)/6.
    heading=endpoint_heading(y);out[:,:,13]=heading.sin();out[:,:,14]=heading.cos()
    assert torch.isfinite(out).all() and not out.requires_grad
    return out

@torch.no_grad()
def neighbors(w):
    current=w.history[:,4];valid=~w.history_padding[:,4];n=len(current)
    distance=torch.cdist(current.double(),current.double())
    allowed=valid[:,None]&valid[None,:]&~torch.eye(n,dtype=torch.bool)&(distance<=50.)
    distance=distance.masked_fill(~allowed,float('inf'))
    m=min(8,n);idx=distance.argsort(dim=-1,stable=True)[:,:m];mask=allowed.gather(1,idx)
    result=torch.zeros((n,8),dtype=torch.long);keep=torch.zeros((n,8),dtype=torch.bool)
    result[:,:m]=idx;keep[:,:m]=mask
    assert not (keep & (result==torch.arange(n)[:,None])).any() and keep.sum(-1).max()<=8
    return result,keep

@torch.no_grad()
def interaction_edges(w,idx,keep,targets=None):
    # No neighbor pre-aggregation: retain all 6x6 pairwise relations separately.
    targets=torch.arange(len(w.history)) if targets is None else targets
    a=w.predicted[targets];b=w.predicted[idx[targets]]
    rel=a[:,:,None,None]-b[:,None]
    distance=rel.norm(dim=-1);minimum,timestep=distance.min(-1)
    ha=endpoint_heading(a);hb=endpoint_heading(b);angle=ha[:,:,None,None]-hb[:,None]
    step_a=torch.diff(torch.cat((w.history[targets,4,None,None].expand(-1,6,1,2),a),dim=2),dim=2)
    step_b=torch.diff(torch.cat((w.history[idx[targets],4,None,None].expand(-1,8,6,1,2),b),dim=3),dim=3)
    disp=(step_a[:,:,None,None]-step_b[:,None]).norm(dim=-1).min(-1).values
    p_neighbor=w.original_probability[idx[targets]][:,None].expand(-1,6,-1,-1)
    p_target=w.original_probability[targets][:,:,None,None].expand_as(p_neighbor)
    geom=torch.stack((minimum,(timestep+1).to(a.dtype)/12.,distance.mean(-1),distance[...,-1],distance[...,0],
        angle.sin(),angle.cos(),disp,(distance[...,0]-distance[...,-1])/6.,p_neighbor,p_target),-1)
    ta=torch.nn.functional.one_hot(w.actor_type[targets],3)[:,None,None,None].expand(-1,6,8,6,-1)
    tb=torch.nn.functional.one_hot(w.actor_type[idx[targets]],3)[:,None,:,None].expand(-1,6,-1,6,-1)
    out=torch.cat((geom,ta,tb),-1);out=out.masked_fill(~keep[targets,None,:,None,None],0.)
    assert out.shape==(len(targets),6,8,6,17) and torch.isfinite(out).all()
    return out
