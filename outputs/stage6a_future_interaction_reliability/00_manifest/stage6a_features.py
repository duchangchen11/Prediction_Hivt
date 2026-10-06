"""Inference-computable own-mode and probability-weighted future interactions."""
import torch
from stage5a_decoder import condition_features

FEATURE_NAMES=('type_vehicle','type_pedestrian','type_bicycle','log1p_recent_displacement',
    'log1p_net_displacement','log1p_path_length','base_logit','base_probability',
    'predicted_endpoint_displacement','predicted_path_length','mean_step_displacement','max_step_displacement',
    'minimum_expected_min_distance','mean_expected_min_distance','minimum_expected_mean_distance',
    'maximum_expected_conflict','mean_expected_conflict','maximum_vehicle_conflict','maximum_pedestrian_conflict')

@torch.no_grad()
def observable_features(history,history_padding,actor_type,predicted,base_logits,base_prob,interaction=True):
    """No GT, future mask, target mask, map, label or refined probability argument."""
    n=len(history);k=6;t=12
    assert predicted.shape==(n,k,t,2) and base_logits.shape==base_prob.shape==(n,k)
    assert torch.isfinite(predicted).all() and torch.isfinite(base_prob).all()
    own=condition_features(history,history_padding,actor_type)
    current=history[:,4]
    steps=torch.diff(torch.cat((current[:,None,None].expand(n,k,1,2),predicted),dim=2),dim=2).norm(dim=-1)
    features=predicted.new_zeros((n,k,19))
    features[:,:,:6]=own[:,None];features[:,:,6]=base_logits;features[:,:,7]=base_prob
    features[:,:,8]=(predicted[:,:,-1]-current[:,None]).norm(dim=-1)
    features[:,:,9]=steps.sum(-1);features[:,:,10]=steps.mean(-1);features[:,:,11]=steps.max(-1).values
    if not interaction:
        return features,torch.zeros((n,3),dtype=torch.bool,device=history.device),None,None
    valid=~history_padding[:,4]
    distances=torch.cdist(current.double(),current.double())
    allowed=valid[:,None]&valid[None,:]&~torch.eye(n,dtype=torch.bool,device=history.device)&(distances<=50.)
    distances=distances.masked_fill(~allowed,float('inf'))
    m=min(8,n);neighbors=distances.argsort(dim=-1,stable=True)[:,:m]
    neighbor_valid=allowed.gather(1,neighbors)
    neighbor_types=actor_type[neighbors]
    flags=torch.stack((neighbor_valid.any(-1),((neighbor_types==0)&neighbor_valid).any(-1),
                       ((neighbor_types==1)&neighbor_valid).any(-1)),-1)
    # Raw absent-neighbor distances are zero sentinels, excluded from distance
    # normalization statistics and explicitly set to one AFTER normalization.
    for start in range(0,n,64):
        stop=min(start+64,n);idx=neighbors[start:stop];mask=neighbor_valid[start:stop]
        distance=(predicted[start:stop,:,None,None]-predicted[idx][:,None]).norm(dim=-1)
        weights=base_prob[idx][:,None]
        expected_min=(distance.min(-1).values*weights).sum(-1)
        expected_mean=(distance.mean(-1)*weights).sum(-1)
        expected_conflict=(torch.exp(-distance.square()/8.).mean(-1)*weights).sum(-1)
        denominator=mask.sum(-1).clamp(min=1)[:,None]
        has=flags[start:stop,0][:,None]
        minimum=torch.where(has,expected_min.masked_fill(~mask[:,None],float('inf')).min(-1).values,0.)
        mean=(expected_min*mask[:,None]).sum(-1)/denominator
        minimum_mean=torch.where(has,expected_mean.masked_fill(~mask[:,None],float('inf')).min(-1).values,0.)
        conflict=expected_conflict*mask[:,None]
        vehicle=conflict*(neighbor_types[start:stop]==0)[:,None]
        pedestrian=conflict*(neighbor_types[start:stop]==1)[:,None]
        features[start:stop,:,12:]=torch.stack((minimum,mean,minimum_mean,conflict.max(-1).values,
            conflict.sum(-1)/denominator,vehicle.max(-1).values,pedestrian.max(-1).values),-1)
    assert features.shape==(n,6,19) and torch.isfinite(features).all()
    return features,flags,neighbors,neighbor_valid

@torch.no_grad()
def normalize(features,flags,statistics,variant='R2'):
    mean=torch.as_tensor(statistics['mean'],dtype=features.dtype,device=features.device)
    std=torch.as_tensor(statistics['std'],dtype=features.dtype,device=features.device)
    out=features.clone();out[:,:,3:]=(out[:,:,3:]-mean[3:])/(std[3:]+1e-6)
    absent=~flags[:,0]
    out[absent,:,12:15]=1.;out[absent,:,15:]=0.
    out[~flags[:,1],:,17]=0.;out[~flags[:,2],:,18]=0.
    if variant=='R1':out[:,:,12:]=0.
    assert variant in ('R1','R2') and torch.isfinite(out).all()
    return out
