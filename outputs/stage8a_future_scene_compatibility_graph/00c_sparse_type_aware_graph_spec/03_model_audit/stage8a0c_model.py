"""Frozen static G1/G2/G3 specification. No fitting, loss or optimizer API."""
import torch
from torch import nn

def mlp(a,b,c):return nn.Sequential(nn.Linear(a,b),nn.ReLU(),nn.Linear(b,c))
def interaction_attention(scores,mask):return scores.masked_fill(~mask,-1e9).softmax(-1)*mask

class MapAggregator(nn.Module):
    def __init__(self):
        super().__init__();self.encoder=mlp(44,64,64);self.attention=mlp(93,64,1)
    def forward(self,hidden,target_raw,map_node,map_edge,map_mask):
        assert hidden.shape[-1]==64 and target_raw.shape[-1]==15
        assert map_node.shape[-1]==18 and map_edge.shape[-1]==11
        if not bool(map_mask.any()):return torch.zeros_like(hidden)
        prefix=hidden.shape[:-1];slots=map_mask.shape[-1]
        h=hidden.reshape(-1,64);raw=target_raw.reshape(-1,15)
        node=map_node.reshape(-1,slots,18);edge=map_edge.reshape(-1,slots,11);mask=map_mask.reshape(-1,slots)
        row,slot=mask.nonzero(as_tuple=True)
        value=self.encoder(torch.cat((raw[row],node[row,slot],edge[row,slot]),-1))
        score=self.attention(torch.cat((h[row],node[row,slot],edge[row,slot]),-1)).squeeze(-1)
        scores=h.new_full((len(h),slots),float('-inf'));scores[row,slot]=score
        alpha=h.new_zeros((len(h),slots));active=mask.any(-1)
        # Softmax is called only on rows with actual map edges. Empty rows stay exact zero.
        alpha[active]=scores[active].softmax(-1)
        values=h.new_zeros((len(h),slots,64));values[row,slot]=value
        message=(alpha[...,None]*values).sum(-2)
        assert torch.equal(message[~active],torch.zeros_like(message[~active]))
        return message.reshape(*prefix,64)

class SparseGraphReranker(nn.Module):
    def __init__(self,variant,seed=2022):
        super().__init__();assert variant in ('G1','G2','G3');self.variant=variant
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.node_encoder=mlp(15,64,64);self.norm=nn.LayerNorm(64);self.head=mlp(64,32,1)
            nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
            if variant in ('G1','G3'):
                torch.manual_seed(seed+101);self.interaction_encoder=mlp(47,64,64);self.interaction_attention=mlp(145,64,1)
            if variant in ('G2','G3'):
                torch.manual_seed(seed+102);self.map_aggregator=MapAggregator()
    def forward(self,local_nodes,interaction_edge,neighbor_mask,map_node,map_edge,map_mask,original_logits):
        assert local_nodes.shape[1:]==(9,6,15)
        hidden=self.node_encoder(local_nodes);h=hidden[:,0];mi=torch.zeros_like(h);mm=torch.zeros_like(h)
        if self.variant in ('G1','G3'):
            a=local_nodes[:,0,:,None,None].expand(-1,6,8,6,-1)
            b=local_nodes[:,None,1:].expand(-1,6,-1,-1,-1)
            hi=h[:,:,None,None].expand(-1,6,8,6,-1);hj=hidden[:,None,1:].expand(-1,6,-1,-1,-1)
            message=self.interaction_encoder(torch.cat((a,b,interaction_edge),-1))
            score=self.interaction_attention(torch.cat((hi,hj,interaction_edge),-1)).squeeze(-1).reshape(-1,6,48)
            mask=neighbor_mask[:,None,:,None].expand(-1,6,8,6).reshape(-1,6,48)
            alpha=interaction_attention(score,mask);mi=(alpha[...,None]*message.reshape(-1,6,48,64)).sum(-2)
        if self.variant in ('G2','G3'):mm=self.map_aggregator(h,local_nodes[:,0],map_node,map_edge,map_mask)
        delta=self.head(self.norm(h+mi+mm)).squeeze(-1);logits=original_logits+delta
        return {'delta_logits':delta,'mode_logits':logits,'mode_prob':logits.softmax(-1),'map_message':mm}

def pack_targets(node,idx,keep,edge,selected,map_features,targets):
    lookup={int(v):i for i,v in enumerate(selected)};rows=torch.tensor([lookup[int(v)] for v in targets])
    own=node[targets];local=torch.cat((own[:,None],node[idx[targets]]),dim=1)
    # Convert index rows to NumPy explicitly: NumPy interprets a one-element Torch
    # index as a scalar in some cases, which would drop the target batch dimension.
    rr=rows.numpy()
    packed=(local,edge,keep[targets],torch.from_numpy(map_features['map_node'][rr]),
        torch.from_numpy(map_features['map_edge'][rr]),torch.from_numpy(map_features['map_mask'][rr]),own[:,:,3])
    assert [x.ndim for x in packed]==[4,5,2,4,4,3,2]
    return packed
