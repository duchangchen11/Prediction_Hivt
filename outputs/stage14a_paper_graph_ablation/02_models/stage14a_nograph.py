"""Stage14A target-only ablation of the frozen Stage8 G1 architecture.

The seven forward arguments remain the original G1 arguments. Complete graph
inputs are supplied by the shared training/evaluation pipeline; no actor, edge,
or candidate is removed to construct this ablation. Only the learned interaction
message calculation is omitted. The target encoder, normalization, score head,
and original-logit residual are unchanged.
"""
from pathlib import Path
import sys

import torch
from torch import nn


_PROJECT = Path(__file__).resolve().parents[3]
_ORIGINAL_MODEL_DIRECTORY = (
    _PROJECT / "outputs/stage8a_future_scene_compatibility_graph/"
    "00c_sparse_type_aware_graph_spec/03_model_audit"
)
if str(_ORIGINAL_MODEL_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(_ORIGINAL_MODEL_DIRECTORY))
from stage8a0c_model import SparseGraphReranker


class NoGraphReranker(nn.Module):
    """G1 with an exact zero interaction message and no interaction parameters."""

    def __init__(self, seed=2022):
        super().__init__()
        # Reuse the original constructor's initialization and its exact modules.
        # The temporary source model is discarded after extracting shared parts;
        # its unused interaction modules are never registered on this model.
        source = SparseGraphReranker("G1", seed=seed)
        self.node_encoder = source.node_encoder
        self.norm = source.norm
        self.head = source.head
        self.variant = "NoGraph"
        self.seed = int(seed)

    def forward(self, local_nodes, interaction_edge, neighbor_mask,
                map_node, map_edge, map_mask, original_logits):
        assert local_nodes.shape[1:] == (9, 6, 15)
        assert original_logits.shape == (local_nodes.shape[0], 6)
        # Graph input remains intact. Encoding only the target is numerically the
        # same target operation as original G1; neighbor encodings are unnecessary
        # when the interaction message is fixed to zero.
        h = self.node_encoder(local_nodes[:, 0])
        interaction_message = torch.zeros_like(h)
        map_message = torch.zeros_like(h)
        delta = self.head(self.norm(h + interaction_message + map_message)).squeeze(-1)
        logits = original_logits + delta
        return {
            "delta_logits": delta,
            "mode_logits": logits,
            "mode_prob": logits.softmax(-1),
            "map_message": map_message,
        }


def parameter_counts(model):
    """Report all registered and actually trainable parameters separately."""
    return {
        "TotalParameters": sum(p.numel() for p in model.parameters()),
        "TrainableParameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
