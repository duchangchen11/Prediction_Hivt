"""One history-only target gate scales the unchanged Stage4A logit bias.

The frozen GlobalInteractor is a complete current-valid graph. Its radius50
setting affects the LocalEncoder, not global attention. We retain that topology
exactly; only gate neighborhood features select existing edges within50m.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from stage4a_global_interactor import TypeConditionedGlobalInteractor

FEATURE_NAMES = ('type_vehicle','type_pedestrian','type_bicycle',
    'log1p_recent_displacement','log1p_history_net_displacement',
    'log1p_history_path_length','log1p_neighbor_count',
    'min_neighbor_distance_div50','heterogeneous_neighbor_fraction')


def history_gate_features(history, history_padding, agent_type, current_edges):
    """Return [N_current,9], current node IDs using ONLY the five history points.

    Path length connects successive chronological valid observations, skipping
    padded observations. Fewer than two valid observations gives zero motion.
    Incoming neighbors are current-valid existing GI edges of distance<=50m.
    This pure function has no data object, future tensor or target access.
    """
    assert history.ndim==3 and history.shape[1:]==(5,2)
    assert history_padding.shape==history.shape[:2]
    n=history.shape[0]; valid=~history_padding
    ids=torch.where(valid[:,4])[0]
    p=history[ids]; v=valid[ids]; row=torch.arange(len(ids),device=p.device)
    t=torch.arange(5,device=p.device).expand(len(ids),5)
    first=torch.where(v,t,5).min(-1).values
    previous=torch.where(v[:,:4],t[:,:4],-1).max(-1).values
    recent=torch.linalg.vector_norm(p[:,4]-p[row,previous.clamp(min=0)],dim=-1)
    recent=torch.where(previous>=0,recent,torch.zeros_like(recent))
    net=torch.linalg.vector_norm(p[:,4]-p[row,first],dim=-1)
    # At each time, look up the most recent earlier valid point.
    prior=torch.where(v,t,-1).cummax(-1).values[:,:4]
    deltas=torch.linalg.vector_norm(p[:,1:]-p.gather(1,prior.clamp(min=0)[...,None].expand(-1,-1,2)),dim=-1)
    path=(torch.where(v[:,1:]&(prior>=0),deltas,torch.zeros_like(deltas))).sum(-1)
    source,target=current_edges
    distance=torch.linalg.vector_norm(history[source,4]-history[target,4],dim=-1)
    near=valid[source,4]&valid[target,4]&(source!=target)&(distance<=50.0)
    source,target,distance=source[near],target[near],distance[near]
    counts=torch.zeros(n,device=p.device,dtype=p.dtype)
    counts.index_add_(0,target,torch.ones_like(distance))
    heterogeneous=torch.zeros_like(counts)
    heterogeneous.index_add_(0,target,(agent_type[source]!=agent_type[target]).to(p.dtype))
    minimum=torch.full_like(counts,50.0)
    minimum.scatter_reduce_(0,target,distance,reduce='amin',include_self=True)
    features=torch.cat((F.one_hot(agent_type[ids],3).to(p.dtype),
        torch.stack((recent.log1p(),net.log1p(),path.log1p(),counts[ids].log1p(),
                     minimum[ids]/50.0,heterogeneous[ids]/counts[ids].clamp(min=1)),dim=-1)),dim=-1)
    return features,ids


class NecessityGatedGlobalInteractor(TypeConditionedGlobalInteractor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.necessity_gate_mlp=nn.Sequential(nn.Linear(9,16),nn.ReLU(),nn.Linear(16,1))
        nn.init.zeros_(self.necessity_gate_mlp[-1].weight)
        nn.init.constant_(self.necessity_gate_mlp[-1].bias,math.log(0.1/0.9))
        self.record_necessity_gate=False
        self.necessity_gate_observation=None

    def necessity_gate(self,data,edge_index):
        features,ids=history_gate_features(data.positions[:,:5],
            data.padding_mask[:,:5],data.agent_type,edge_index)
        current=torch.sigmoid(self.necessity_gate_mlp(features)).squeeze(-1)
        all_nodes=features.new_zeros(data.num_nodes).index_copy(0,ids,current)
        return all_nodes,features,ids

    def forward(self,data,local_embed):
        edge_index,rel_pos,heading=self._edge_relations(data)
        rel_embed=self.rel_embed(rel_pos) if data['rotate_mat'] is None else self.rel_embed([rel_pos,heading])
        pair_ids,relation_features,bias=self._relation_features(data,edge_index,rel_pos,heading)
        necessity_gate,features,ids=self.necessity_gate(data,edge_index)
        # Same scalar for every incoming edge, every layer and every head.
        gated_bias=necessity_gate[edge_index[1],None,None]*bias
        if self.record_necessity_gate:
            self.necessity_gate_observation={'node_ids':ids.detach(),'gate':necessity_gate.detach(),
                'features':features.detach(),'edge_index':edge_index.detach()}
        if self.record_relation_bias:
            self.relation_bias_observation={'edge_index':edge_index.detach(),'pair_ids':pair_ids.detach(),
                'relation_features':relation_features.detach(),'bias':bias.detach(),
                'gated_bias':gated_bias.detach()}
        x=local_embed
        for layer_index,layer in enumerate(self.global_interactor_layers):
            x=layer(x,edge_index,rel_embed,gated_bias[:,layer_index,:])
        x=self.norm(x)
        x=self.multihead_proj(x).view(-1,self.num_modes,self.embed_dim)
        return x.transpose(0,1)
