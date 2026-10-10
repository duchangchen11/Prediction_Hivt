"""New predictor candidates → frozen graph/R2 feature definitions, no old weights."""
import ast
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "00_manifest"))
from stage15b_common import PROJECT, model_input, atomic_json, state_digest
import numpy as np
import torch

S8 = PROJECT / "outputs/stage8a_future_scene_compatibility_graph"
S6 = PROJECT / "outputs/stage6a_future_interaction_reliability"
sys.path[:0] = [str(S8 / "00_manifest"), str(S8 / "00c_sparse_type_aware_graph_spec/03_model_audit"),
               str(S6 / "00_manifest"), str(PROJECT / "outputs/stage14a_paper_graph_ablation/02_models"),
               str(PROJECT / "outputs/stage14b_paper_validation/02_models")]
from stage8a_graph import observable_window, node_features, neighbors, interaction_edges
from stage8a0c_model import SparseGraphReranker
from stage6a_features import observable_features, normalize as normalize_r2
from stage6a_head import ReliabilityHead, ranking_loss
from stage14a_nograph import NoGraphReranker
from stage14b_matched_nograph import MatchedNoGraphReranker

# Reuse the actual Stage11B function bodies without importing its write hooks,
# cached arrays, frozen all700 predictor, or historical head checkpoints.
source = PROJECT / "outputs/stage11b_error_aware_ranking/00_manifest/stage11b_common.py"
tree = ast.parse(source.read_text())
body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("graph_normalize", "objective")]
scope = {"torch": torch}
exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), str(source), "exec"), scope)
graph_normalize, objective = scope["graph_normalize"], scope["objective"]


@torch.no_grad()
def candidate_windows(model, batch, graphs, context, role):
    context.check_batch(batch, role, "candidate_interface")
    model.eval()
    data = batch.cuda()
    observed = model_input(data)
    out = model(observed)
    pred = model.ego_predictions(out, observed)
    ptr = batch.ptr.tolist()
    result = []
    for j, g in enumerate(graphs):
        lo, hi = ptr[j:j + 2]
        w = {"history": g.positions[:, :5].clone(), "history_padding": g.padding_mask[:, :5].clone(),
             "agent_type": g.agent_type.clone(), "ego_prediction": pred[lo:hi].cpu(),
             "raw_prediction": out["raw_prediction"][:, lo:hi].cpu(),
             "mode_logits": out["mode_logits"][lo:hi].cpu(), "mode_prob": out["mode_prob"][lo:hi].cpu(),
             "rotation": out["rotation"][lo:hi].cpu(), "scene_token": g.scene_token,
             "sample_token": g.sample_token, "instance_tokens": tuple(g.instance_tokens),
             "map_location": g.map_location, "origin": g.origin.cpu().numpy(), "yaw": float(g.ego_yaw),
             "history_times": g.history_times.cpu(), "future_sample_times_metadata": g.future_times.cpu(),
             "source_fold": context.fold, "source_seed": context.seed, "source_role": role,
             "predictor_state_sha256": state_digest(model.state_dict()), "candidate_modes": tuple(range(6)),
             "candidate_mask": torch.ones((g.num_nodes, 6), dtype=torch.bool),
             "coordinate_frame": "t0 ego x-forward y-left meters; 6sec 12samples"}
        assert not set(w) & {"GT", "y", "future_mask", "target_mask"}
        explicit = torch.matmul(w["raw_prediction"][..., :2].permute(1, 0, 2, 3),
                                w["rotation"].transpose(-1, -2)[:, None]) + w["history"][:, 4, None, None]
        assert torch.allclose(explicit, w["ego_prediction"], atol=1e-5, rtol=1e-6)
        assert w["ego_prediction"].shape == (g.num_nodes, 6, 12, 2)
        assert torch.allclose(w["mode_logits"].softmax(-1), w["mode_prob"], atol=1e-7, rtol=1e-6)
        assert len(w["instance_tokens"]) == g.num_nodes
        result.append(w)
    return result


@torch.no_grad()
def pack_window(w, targets=None):
    obs = observable_window(w, w["map_location"], w["origin"], w["yaw"])
    node = node_features(obs)
    idx, keep = neighbors(obs)
    # Default selection uses past eligibility, never future horizon visibility.
    targets = torch.where((~w["history_padding"]).sum(-1) >= 2)[0] if targets is None else targets
    edge = interaction_edges(obs, idx, keep, targets)
    local = torch.cat((node[targets, None], node[idx[targets]]), 1)
    base = w["mode_logits"][targets].clone()
    args = (local, edge, keep[targets], None, None, None, base)
    r2, flags, _, _ = observable_features(w["history"], w["history_padding"], w["agent_type"],
                                        w["ego_prediction"], w["mode_logits"], w["mode_prob"])
    assert args[0].shape == (len(targets), 9, 6, 15)
    assert args[1].shape == (len(targets), 6, 8, 6, 17)
    assert torch.equal(args[0][:, 0, :, 3], base)
    return {"args": args, "r2": r2[targets], "flags": flags[targets], "targets": targets,
            "scene_token": w["scene_token"], "source_role": w["source_role"],
            "sample_token": w["sample_token"], "identities": [w["instance_tokens"][int(t)] for t in targets]}


def moments(values):
    v = values.double()
    assert v.numel() > 0 and torch.isfinite(v).all()
    return {"count": v.numel(), "mean": float(v.mean()), "std": float(v.std(unbiased=False))}


def fit_normalization(entries, context):
    assert entries and all(e["source_role"] == "InnerTrain" and e["scene_token"] in context.parts["InnerTrain"] for e in entries)
    node = torch.cat([e["args"][0] for e in entries])
    edge = torch.cat([e["args"][1] for e in entries])
    mask = torch.cat([e["args"][2] for e in entries])
    nv = torch.cat((torch.ones((len(mask), 1), dtype=torch.bool), mask), 1)[..., None].expand(-1, 9, 6)
    ev = mask[:, None, :, None].expand(-1, 6, 8, 6)
    stats = {"0": {}, "1": {}}
    for k, x, valid, cols in ((0, node, nv, range(3, 15)), (1, edge, ev, range(11))):
        for c in cols:
            stats[str(k)][str(c)] = moments(x[..., c][valid])
    raw_r2 = torch.cat([e["r2"] for e in entries])
    flags = torch.cat([e["flags"] for e in entries])
    rstats = [moments(raw_r2[flags[:, 0], :, c] if 12 <= c < 15 else raw_r2[:, :, c]) for c in range(19)]
    norm = {"Fold": context.fold, "fit_partition": "InnerTrain", "preflight_subset_only": True,
            "formal_eligible": False, "fit_scene_tokens": sorted({e["scene_token"] for e in entries}),
            "fit_actor_windows": len(mask), "graph": stats,
            "R2": {"mean": [s["mean"] for s in rstats], "std": [s["std"] for s in rstats]},
            "epsilon": 1e-6, "std_ddof": 0, "InnerDev_or_Outer_or_HeadDev_used": False}
    return norm


def normalized(entry, norm):
    node, edge, mask = graph_normalize(*entry["args"][:3], norm["graph"])
    return (node, edge, mask, None, None, None, entry["args"][6]), normalize_r2(entry["r2"], entry["flags"], norm["R2"])


def fresh_heads(seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return {"R2": ReliabilityHead(), "NG-A": NoGraphReranker(seed), "NG-C": NoGraphReranker(seed),
                "G-A": SparseGraphReranker("G1", seed), "G-C": SparseGraphReranker("G1", seed),
                "Matched-NG-C": MatchedNoGraphReranker(seed)}


def head_forward(head, name, args, r2):
    return head(r2, args[6]) if name == "R2" else head(*args)
