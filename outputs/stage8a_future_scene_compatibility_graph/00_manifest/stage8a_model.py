"""Fixed feasibility prototype only: no fit loop, optimizer or checkpoint creation."""
import torch
from torch import nn

def mlp(a,b,c):return nn.Sequential(nn.Linear(a,b),nn.ReLU(),nn.Linear(b,c))
def masked_attention(scores,mask):
    # All-absent sets return exact zero without softmax(-inf,...,-inf) NaNs.
    return scores.masked_fill(~mask,-1e9).softmax(-1)*mask

class GraphReranker(nn.Module):
    def __init__(self,variant,seed=2022):
        super().__init__();assert variant in ('G1','G2','G3');self.variant=variant
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.node_encoder=mlp(15,64,64);self.norm=nn.LayerNorm(64);self.head=mlp(64,32,1)
            nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
            if variant in ('G1','G3'):
                torch.manual_seed(seed+101);self.interaction_encoder=mlp(47,64,64);self.interaction_attention=mlp(145,64,1)
            if variant in ('G2','G3'):
                torch.manual_seed(seed+102);self.map_encoder=mlp(45,64,64);self.map_attention=mlp(94,64,1)

    def forward(self,local_nodes,interaction_edge,neighbor_mask,map_node,map_edge,map_mask,original_logits):
        # local_nodes [B,9,6,15]; only the target's six nodes produce outputs.
        assert local_nodes.shape[1:]==(9,6,15)
        hidden=self.node_encoder(local_nodes);h=hidden[:,0];mi=torch.zeros_like(h);mm=torch.zeros_like(h)
        if self.variant in ('G1','G3'):
            a=local_nodes[:,0,:,None,None].expand(-1,6,8,6,-1)
            b=local_nodes[:,None,1:].expand(-1,6,-1,-1,-1)
            hi=h[:,:,None,None].expand(-1,6,8,6,-1);hj=hidden[:,None,1:].expand(-1,6,-1,-1,-1)
            message=self.interaction_encoder(torch.cat((a,b,interaction_edge),-1))
            score=self.interaction_attention(torch.cat((hi,hj,interaction_edge),-1)).squeeze(-1).reshape(-1,6,48)
            mask=neighbor_mask[:,None,:,None].expand(-1,6,8,6).reshape(-1,6,48)
            alpha=masked_attention(score,mask);mi=(alpha[...,None]*message.reshape(-1,6,48,64)).sum(-2)
        if self.variant in ('G2','G3'):
            target=local_nodes[:,0,:,None].expand(-1,6,8,-1)
            message=self.map_encoder(torch.cat((target,map_node,map_edge),-1))
            score=self.map_attention(torch.cat((h[:,:,None].expand(-1,6,8,-1),map_node,map_edge),-1)).squeeze(-1)
            beta=masked_attention(score,map_mask);mm=(beta[...,None]*message).sum(-2)
        delta=self.head(self.norm(h+mi+mm)).squeeze(-1);logits=original_logits+delta
        return {'delta_logits':delta,'mode_logits':logits,'mode_prob':logits.softmax(-1)}

def pack_targets(node,idx,keep,edge,selected,map_features,targets):
    lookup={int(v):i for i,v in enumerate(selected)};rows=torch.tensor([lookup[int(v)] for v in targets])
    own=node[targets];other=node[idx[targets]]
    local=torch.cat((own[:,None],other),dim=1)
    return (local,edge,keep[targets],torch.from_numpy(map_features['map_node'][rows]),
            torch.from_numpy(map_features['map_edge'][rows]),torch.from_numpy(map_features['map_mask'][rows]),own[:,:,3])
