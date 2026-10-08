"""Independent G1 experts; original-logit residual; Bicycle routes to frozen R2."""
from stage10a_common import torch, SparseGraphReranker
from torch import nn

class Expert(SparseGraphReranker):
    def __init__(self, seed):
        super().__init__('G1',seed=seed)
    def forward(self, local_nodes, interaction_edge, neighbor_mask, original_logits):
        return super().forward(local_nodes,interaction_edge,neighbor_mask,None,None,None,original_logits)

class DualExpert(nn.Module):
    def __init__(self, vehicle, pedestrian, r2):
        super().__init__()
        self.vehicle=vehicle; self.pedestrian=pedestrian; self.r2=r2.eval().requires_grad_(False)
        assert not {p.data_ptr() for p in vehicle.parameters()} & {p.data_ptr() for p in pedestrian.parameters()}
    def forward(self, local_nodes, interaction_edge, neighbor_mask, original_logits, r2_features, candidates):
        assert candidates.shape==(len(original_logits),6,12,2) and not candidates.requires_grad
        types=local_nodes[:,0,0,:3].argmax(-1)
        with torch.no_grad(): base=self.r2(r2_features,original_logits)
        logits=base['mode_logits'].clone(); probability=base['mode_prob'].clone()
        for t,expert in enumerate((self.vehicle,self.pedestrian)):
            selected=types==t
            if bool(selected.any()):
                out=expert(local_nodes[selected],interaction_edge[selected],neighbor_mask[selected],original_logits[selected])
                logits[selected]=out['mode_logits']; probability[selected]=out['mode_prob']
        # Raw candidate tensor is returned directly, with identical storage and values.
        return {'mode_logits':logits,'mode_prob':probability,'candidates':candidates}
