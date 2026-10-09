"""Matched 641-parameter heads; only pedestrian mode scores can change."""
import torch
from torch import nn
class SemanticResidual(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(18,32),nn.ReLU(),nn.Linear(32,1))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
        assert sum(p.numel() for p in self.parameters())==641
    def forward(self,x,base):
        assert x.shape[-2:]==(6,18) and base.shape==x.shape[:-1]
        delta=2*torch.tanh(self.net(x).squeeze(-1));logits=base+delta
        return {'delta':delta,'logits':logits,'probabilities':logits.softmax(-1)}
def objective(out,fde,base):
    fde=fde.detach();cost=fde-fde.min(-1,keepdim=True).values;scale=cost.mean(-1).clamp(min=1.)
    logp=out['logits'].log_softmax(-1);p=out['probabilities'];teacher=base.detach().softmax(-1);teacherlog=base.detach().log_softmax(-1)
    risk=(p*(cost/scale[:,None])).sum(-1);anchor=(teacher*(teacherlog-logp)).sum(-1);penalty=out['delta'].square().mean(-1)
    total=risk+.1*anchor+.001*penalty
    return total,torch.stack((risk,anchor,penalty),-1)
