"""Fresh canonical Stage3B with one zero-initialized static lane residual."""
from pathlib import Path
import sys
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / 'stage3_multitype_hivt/00_manifest'))
from stage3b_common import model_new as canonical_stage3b_new
from stage7a_local_encoder import SemanticALEncoder, SemanticLocalEncoder

def model_new(device='cuda', audit=False):
    # Canonical constructor completes ALL Stage3B initialization first.
    model = canonical_stage3b_new(device='cpu', audit=False)
    cpu_rng = torch.get_rng_state().clone()
    old = model.local_encoder
    al = SemanticALEncoder(node_dim=2, edge_dim=2, embed_dim=64,
                          num_heads=8, dropout=0.1)
    missing, unexpected = al.load_state_dict(old.al_encoder.state_dict(), strict=False)
    assert missing == ['semantic_mlp.0.weight', 'semantic_mlp.0.bias',
                       'semantic_mlp.2.weight', 'semantic_mlp.2.bias'] and not unexpected
    local = SemanticLocalEncoder.__new__(SemanticLocalEncoder)
    nn.Module.__init__(local)
    local.historical_steps, local.parallel = old.historical_steps, old.parallel
    local.drop_edge, local.aa_encoder = old.drop_edge, old.aa_encoder
    local.temporal_encoder, local.al_encoder = old.temporal_encoder, al
    model.local_encoder = local
    # Match Stage3B training RNG policy; the new branch has its own deterministic init.
    torch.set_rng_state(cpu_rng)
    assert sum(p.numel() for p in model.parameters()) == 648433
    return model.to(device)
