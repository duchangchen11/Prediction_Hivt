"""One preregistered active-capacity NoGraph control for Stage14B.

Original G1 target node encoder, LayerNorm, scoring head and original-logit
residual are reused exactly. An active own-feature adapter adds 16,576 trainable
parameters; the complete model has 24,001, compared with G1's 24,066. It never
computes a neighbor interaction message, and no unused parameter padding exists.
"""
from pathlib import Path
import sys

import torch
from torch import nn


PROJECT = Path(__file__).resolve().parents[3]
ORIGINAL_DIRECTORY = PROJECT / "outputs/stage8a_future_scene_compatibility_graph/00c_sparse_type_aware_graph_spec/03_model_audit"
if str(ORIGINAL_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(ORIGINAL_DIRECTORY))
from stage8a0c_model import SparseGraphReranker


class MatchedNoGraphReranker(nn.Module):
    """G1's own-node score plus one fixed 64→128→64 own-node residual adapter."""

    def __init__(self, seed=2022):
        super().__init__()
        source = SparseGraphReranker("G1", seed=seed)
        self.node_encoder = source.node_encoder
        self.norm = source.norm
        self.head = source.head
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed) + 101)
            self.own_adapter = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 64))
        self.variant = "CapacityMatchedNoGraph"
        self.seed = int(seed)
        assert sum(parameter.numel() for parameter in self.parameters()) == 24001

    def forward(self, local_nodes, interaction_edge, neighbor_mask,
                map_node, map_edge, map_mask, original_logits):
        assert local_nodes.shape[1:] == (9, 6, 15)
        assert original_logits.shape == (local_nodes.shape[0], 6)
        own_hidden = self.node_encoder(local_nodes[:, 0])
        # Every adapter parameter contributes to a nonlinear target-only path.
        # Full cached graph tensors remain unchanged and available to the common
        # pipeline, but no neighbor feature enters this structural control.
        own_message = self.own_adapter(own_hidden)
        interaction_message = torch.zeros_like(own_hidden)
        map_message = torch.zeros_like(own_hidden)
        delta = self.head(self.norm(own_hidden + own_message + interaction_message + map_message)).squeeze(-1)
        logits = original_logits + delta
        return {"delta_logits": delta, "mode_logits": logits,
                "mode_prob": logits.softmax(-1), "map_message": map_message}


def parameter_counts(model):
    return {"TotalParameters": sum(parameter.numel() for parameter in model.parameters()),
            "TrainableParameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
            "OwnAdapterParameters": sum(parameter.numel() for parameter in model.own_adapter.parameters())}
