"""Standalone CPU audit of the one fixed active-capacity NoGraph architecture.

No dataset, checkpoint, optimizer, or evaluation-array reads occur here. Real
cached-input and historical loss checks are a separate preregistered preflight.
"""
from pathlib import Path
import ast
import hashlib
import json
import sys
import time

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
sys.path.insert(0, str(ROOT / "02_models"))
from stage14b_matched_nograph import MatchedNoGraphReranker, SparseGraphReranker, parameter_counts


def tensor_sha(value):
    value = value.detach().cpu().contiguous()
    h = hashlib.sha256(str(value.dtype).encode() + str(tuple(value.shape)).encode())
    h.update(value.numpy().tobytes())
    return h.hexdigest()


def fixture(batch=16, seed=2022):
    generator = torch.Generator().manual_seed(seed)
    return (torch.randn(batch, 9, 6, 15, generator=generator),
            torch.randn(batch, 6, 8, 6, 17, generator=generator),
            torch.ones(batch, 8, dtype=torch.bool),
            torch.randn(batch, 6, 16, 18, generator=generator),
            torch.randn(batch, 6, 16, 11, generator=generator),
            torch.ones(batch, 6, 16, dtype=torch.bool),
            torch.randn(batch, 6, generator=generator))


def historical_objective():
    path = PROJECT / "outputs/stage11b_error_aware_ranking/00_manifest/stage11b_common.py"
    tree = ast.parse(path.read_text())
    body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "objective"]
    assert len(body) == 1
    scope = {"torch": torch}
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), str(path), "exec"), scope)
    return scope["objective"], hashlib.sha256(path.read_bytes()).hexdigest()


def seed_audit(seed, objective):
    model = MatchedNoGraphReranker(seed).eval()
    repeated = MatchedNoGraphReranker(seed).eval()
    g1 = SparseGraphReranker("G1", seed=seed).eval()
    assert all(torch.equal(value, repeated.state_dict()[name]) for name, value in model.state_dict().items())
    shared = {name: tensor_sha(value) == tensor_sha(g1.state_dict()[name])
              for name, value in model.state_dict().items() if not name.startswith("own_adapter.")}
    assert all(shared.values())
    assert parameter_counts(model) == {"TotalParameters": 24001, "TrainableParameters": 24001, "OwnAdapterParameters": 16576}
    assert all(not hasattr(model, key) for key in ("interaction_encoder", "interaction_attention", "map_aggregator"))
    inputs = fixture(seed=seed)
    input_sha = [tensor_sha(value) for value in inputs]
    with torch.inference_mode():
        assert torch.equal(model(*inputs)["mode_logits"], inputs[-1])
    # Audit-only nonzero head probe: no training and no checkpoint. This avoids
    # a trivial invariance result from the original zero-initialized final head.
    generator = torch.Generator().manual_seed(seed + 201)
    with torch.no_grad():
        model.head[-1].weight.copy_(.02 * torch.randn(model.head[-1].weight.shape, generator=generator))
        model.head[-1].bias.copy_(.005 * torch.randn(model.head[-1].bias.shape, generator=generator))
    with torch.inference_mode():
        before = model(*inputs)
        perturbed = list(fixture(seed=seed + 301))
        perturbed[0][:, 0] = inputs[0][:, 0]
        perturbed[-1] = inputs[-1]
        after = model(*perturbed)
        assert all(torch.equal(before[name], after[name]) for name in before)
        own_changed = [value.clone() for value in inputs]
        own_changed[0][:, 0] += .25 * fixture(seed=seed + 401)[0][:, 0]
        own_effect = float((model(*own_changed)["mode_logits"] - before["mode_logits"]).abs().max())
        assert own_effect > 1e-6
        hidden = model.node_encoder(inputs[0][:, 0])
        own_adapter = model.own_adapter(hidden)
        assert own_adapter.var(dim=1).mean() > 0
        without_adapter = inputs[-1] + model.head(model.norm(hidden)).squeeze(-1)
        adapter_effect = float((without_adapter - before["mode_logits"]).abs().max())
        assert adapter_effect > 1e-6
        changed_original = list(inputs)
        changed_original[-1] = inputs[-1] + .125
        residual_error = float((model(*changed_original)["mode_logits"] - before["mode_logits"] - .125).abs().max())
        assert residual_error < 1e-6
    errors = torch.rand(16, 6, generator=generator).requires_grad_(True) * 5
    output = model(*inputs)
    loss = objective(output["mode_logits"], errors, "C")
    assert torch.autograd.grad(loss.sum(), errors, allow_unused=True, retain_graph=True)[0] is None
    loss.mean().backward()
    real_c_gradients = {}
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        norm = float(parameter.grad.norm())
        if name.startswith("own_adapter."):
            assert norm > 1e-9, name
        real_c_gradients[name] = {"Finite": True, "Norm": norm, "NonzeroElements": int(torch.count_nonzero(parameter.grad))}
    with torch.inference_mode():
        ordinary = model(*inputs)
        poisoned_labels = errors.detach().flip(-1)
        repeated_forward = model(*inputs)
        assert all(torch.equal(ordinary[name], repeated_forward[name]) for name in ordinary)
        loss_change = float((objective(ordinary["mode_logits"], errors, "C") - objective(ordinary["mode_logits"], poisoned_labels, "C")).abs().max())
        assert loss_change > 1e-6
    # The common last scalar score bias cancels out of softmax. A separate logit
    # MSE backward proves every registered tensor affects forward computation;
    # this is an audit, not a new optimization objective or a fitted model.
    model.zero_grad(set_to_none=True)
    model(*inputs)["mode_logits"].square().mean().backward()
    all_forward_gradient = {name: float(parameter.grad.norm()) for name, parameter in model.named_parameters()}
    assert all(norm > 0 for norm in all_forward_gradient.values())
    assert input_sha == [tensor_sha(value) for value in inputs]
    return {"Seed": seed, "Status": "PASS", "SharedInitializationBitwise": shared,
            "AdapterInitializationRepeatable": True, "AdapterSeed": seed + 101,
            "InitialLogitsEqualOriginal": True, "NoNeighborOrMapParameters": True,
            "NonzeroHeadNeighborAndEdgeMaxDiff": 0., "OwnNodeChangeMaxLogitDiff": own_effect,
            "OwnAdapterContributionMaxLogitDiff": adapter_effect, "OriginalLogitResidualMaxDiff": residual_error,
            "InputUnmodified": True, "InputSHA256": input_sha,
            "FutureLabelsNotInForward": True, "LabelPoisonForwardMaxDiff": 0.,
            "LabelPoisonChangesSeparateLoss": loss_change, "LabelsDetached": True,
            "NormalizedCLossGradientChecks": real_c_gradients,
            "AllAdapterTensorsHaveSubstantialCGradient": True,
            "LogitMSEForwardParticipationProbeGradientNorms": all_forward_gradient,
            "OptimizerOrTrainingPerformed": False}


def compute_audit():
    inputs = fixture(batch=128)
    report = {}
    for name, model in (("MatchedNoGraph", MatchedNoGraphReranker()), ("G1", SparseGraphReranker("G1"))):
        model.eval()
        multiply_adds = []
        hooks = []
        def linear(module, args, output):
            multiply_adds.append((args[0].numel() // module.in_features) * module.in_features * module.out_features)
        for module in model.modules():
            if isinstance(module, nn.Linear): hooks.append(module.register_forward_hook(linear))
        with torch.inference_mode(): model(*inputs)
        for hook in hooks: hook.remove()
        times = []
        with torch.inference_mode():
            for _ in range(5): model(*inputs)
            for _ in range(20):
                start = time.perf_counter_ns(); model(*inputs)
                times.append((time.perf_counter_ns() - start) / 1e6)
        report[name] = {"TotalParameters": sum(parameter.numel() for parameter in model.parameters()),
                        "TrainableParameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
                        "LinearFLOPsPer128Targets": 2 * sum(multiply_adds),
                        "CPU128TargetForwardMedianMS": float(np.median(times)), "Warmups": 5, "Repeats": 20}
    report["Scope"] = "FP32 single-thread CPU head forward only; excludes packing/predictor/CUDA transfer/nonlinear FLOPs. Matching active parameter capacity does not match graph computational complexity."
    return report


def main():
    torch.set_num_threads(1)
    objective, source_sha = historical_objective()
    report = {"Stage": "Stage14B", "Status": "PASS", "Architecture": "fixed original own node64 + residual own adapter64->128->64 + original norm/head/logit residual",
              "ArchitectureSearchPerformed": False, "CandidateOrOriginalLogitModification": False,
              "TotalParameters": 24001, "TrainableParameters": 24001, "OwnAdapterParameters": 16576,
              "G1Parameters": 24066, "CapacityDifferenceParameters": 65, "CapacityDifferenceFraction": 65 / 24066,
              "UnusedCapacityPadding": False, "InteractionMessage": "exact zero; neighbors are not consumed",
              "FoldSeeds": [2022, 2122, 2222], "SeedAudits": [seed_audit(seed, objective) for seed in (2022, 2122, 2222)],
              "InferenceCompute": compute_audit(), "HistoricalObjectiveSourceSHA256": source_sha,
              "SharedScoreBiasNote": "The original final scalar head bias adds a common six-mode offset and is unidentifiable under both SoftCE and normalized C. This same1 shared dimension exists in G1; it is retained for architectural identity and is not added capacity padding.",
              "InputSource": "Synthetic complete full-neighbor/full-candidate tensors only; real raw-GT isolation is checked separately",
              "OuterTestOrVALOrTestRead": False, "TrainingPerformed": False, "OptimizerConstructed": False,
              "FormalTrainingGate": "Requires frozen registration, real input/loss preflight and original300updateC tiny PASS; this structural audit alone grants no training permission"}
    path = ROOT / "01_preflight/stage14b_model_integrity.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print("STAGE14B_MATCHED_NOGRAPH_STRUCTURE_PASS", json.dumps(report["InferenceCompute"]), flush=True)


if __name__ == "__main__": main()
