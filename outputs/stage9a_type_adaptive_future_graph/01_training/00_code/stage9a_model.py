"""One shared frozen-definition future graph, learned type adapters and gate."""
from stage9a_common import torch,SparseGraphReranker,interaction_attention
from torch import nn

class TypeAdaptiveGraph(SparseGraphReranker):
    def __init__(self,name):
        assert name in ('T1','T2','T3');super().__init__('G1');self.name=name
        if name!='T1':
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(2224)
                self.adapters=nn.ModuleList([nn.Sequential(nn.Linear(64,8),nn.ReLU(),nn.Linear(8,64)) for _ in range(3)])
                for a in self.adapters:nn.init.zeros_(a[-1].weight);nn.init.zeros_(a[-1].bias)
                torch.manual_seed(2225)
                self.gate_network=nn.Sequential(nn.Linear(8,16),nn.ReLU(),nn.Linear(16,1),nn.Sigmoid())
    def forward(self,local_nodes,interaction_edge,neighbor_mask,r2_logits,gate_input):
        assert local_nodes.shape[1:]==(9,6,15) and gate_input.shape==(len(local_nodes),8)
        hidden=self.node_encoder(local_nodes);h=hidden[:,0]
        a=local_nodes[:,0,:,None,None].expand(-1,6,8,6,-1);b=local_nodes[:,None,1:].expand(-1,6,-1,-1,-1)
        hi=h[:,:,None,None].expand(-1,6,8,6,-1);hj=hidden[:,None,1:].expand(-1,6,-1,-1,-1)
        message=self.interaction_encoder(torch.cat((a,b,interaction_edge),-1))
        score=self.interaction_attention(torch.cat((hi,hj,interaction_edge),-1)).squeeze(-1).reshape(-1,6,48)
        mask=neighbor_mask[:,None,:,None].expand(-1,6,8,6).reshape(-1,6,48)
        alpha=interaction_attention(score,mask);mi=(alpha[...,None]*message.reshape(-1,6,48,64)).sum(-2)
        h=self.norm(h+mi)
        gate=h.new_ones(len(h))
        if self.name!='T1':
            ty=gate_input[:,:3].argmax(-1);adapter=torch.zeros_like(h)
            for t,layer in enumerate(self.adapters):
                valid=ty==t
                if bool(valid.any()):adapter[valid]=layer(h[valid])
            h=h+adapter;gate=self.gate_network(gate_input).squeeze(-1)
        delta=self.head(h).squeeze(-1);correction=gate[:,None]*delta;logits=r2_logits+correction
        return {'delta_graph':delta,'delta_logits':correction,'gate':gate,'mode_logits':logits,'mode_prob':logits.softmax(-1),'interaction_alpha':alpha}
