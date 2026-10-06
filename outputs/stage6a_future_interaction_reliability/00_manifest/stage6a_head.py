"""Identical 673-parameter residual ranking heads; geometry is never an output."""
import torch
from torch import nn

class ReliabilityHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(19,32),nn.ReLU(),nn.Linear(32,1))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
        assert sum(p.numel() for p in self.parameters())==673
    def forward(self,features,base_logits):
        assert features.shape[-2:]==(6,19) and base_logits.shape==features.shape[:-1]
        delta=self.net(features).squeeze(-1)
        logits=base_logits+delta
        return {'delta_logits':delta,'mode_logits':logits,'mode_prob':logits.softmax(-1)}

def ranking_loss(logits,fde_by_mode):
    q=(-fde_by_mode.detach()/1.).softmax(-1)
    return -(q*logits.log_softmax(-1)).sum(-1).mean()
