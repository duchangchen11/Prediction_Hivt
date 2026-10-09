"""CPU-only structural/numeric NoGraph checks, before any formal training.

This audit uses generated, complete tensors with all eight neighbors and all six
candidates intact. It never reads InnerDev, OuterTest, VAL, test, or checkpoints;
real-data leakage/loss/tiny-overfit checks are separate preflight gates.
"""
from pathlib import Path
import hashlib
import json
import sys
import time

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "02_models"))
from stage14a_nograph import NoGraphReranker, SparseGraphReranker, parameter_counts


def tensor_sha(value):
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(tuple(value.shape)).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def fixture(batch=16, seed=14):
    """Full-shape synthetic graph, not an experimental data transformation."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return (
        torch.randn(batch, 9, 6, 15, generator=generator),
        torch.randn(batch, 6, 8, 6, 17, generator=generator),
        torch.ones(batch, 8, dtype=torch.bool),
        torch.randn(batch, 6, 16, 18, generator=generator),
        torch.randn(batch, 6, 16, 11, generator=generator),
        torch.ones(batch, 6, 16, dtype=torch.bool),
        torch.randn(batch, 6, generator=generator),
    )


def reference_zero_message(g1, inputs):
    """Original G1 target computation with mi=mm=0, using original modules."""
    hidden = g1.node_encoder(inputs[0])
    h = hidden[:, 0]
    mi = torch.zeros_like(h)
    mm = torch.zeros_like(h)
    delta = g1.head(g1.norm(h + mi + mm)).squeeze(-1)
    logits = inputs[-1] + delta
    return {
        "delta_logits": delta,
        "mode_logits": logits,
        "mode_prob": logits.softmax(-1),
        "map_message": mm,
    }


def maximum_output_difference(a, b):
    assert set(a) == set(b)
    return {key: float((a[key] - b[key]).abs().max()) for key in a}


def seed_audit(seed):
    ng = NoGraphReranker(seed=seed).eval()
    g1 = SparseGraphReranker("G1", seed=seed).eval()
    shared = ("node_encoder.", "norm.", "head.")
    ng_state = ng.state_dict()
    g1_shared = {key: value for key, value in g1.state_dict().items() if key.startswith(shared)}
    assert set(ng_state) == set(g1_shared)
    bitwise = {key: tensor_sha(value) == tensor_sha(g1_shared[key]) for key, value in ng_state.items()}
    assert all(bitwise.values())
    assert all(key.startswith(shared) for key, _ in ng.named_parameters())
    assert not hasattr(ng, "interaction_encoder")
    assert not hasattr(ng, "interaction_attention")
    assert not hasattr(ng, "map_aggregator")
    assert parameter_counts(ng) == {"TotalParameters": 7425, "TrainableParameters": 7425}
    assert parameter_counts(g1) == {"TotalParameters": 24066, "TrainableParameters": 24066}
    for name in ("node_encoder", "norm", "head"):
        assert repr(getattr(ng, name)) == repr(getattr(g1, name))
    assert torch.equal(ng.head[-1].weight, torch.zeros_like(ng.head[-1].weight))
    assert torch.equal(ng.head[-1].bias, torch.zeros_like(ng.head[-1].bias))
    inputs = fixture(seed=seed + 14)
    input_before = [tensor_sha(value) for value in inputs]
    with torch.inference_mode():
        initial = ng(*inputs)
        reference = reference_zero_message(g1, inputs)
        initial_diffs = maximum_output_difference(initial, reference)
        assert max(initial_diffs.values()) < 1e-6
        assert torch.equal(initial["mode_logits"], inputs[-1])
    # Initial head weights are zero. A deterministic nonzero audit-only head
    # makes the numerical and neighbor-independence checks nontrivial. No fit or
    # optimizer occurs, and this temporary probe model is never checkpointed.
    generator = torch.Generator(device="cpu").manual_seed(seed + 1414)
    with torch.no_grad():
        probe_weight = .02 * torch.randn(ng.head[-1].weight.shape, generator=generator)
        probe_bias = .005 * torch.randn(ng.head[-1].bias.shape, generator=generator)
        for model in (ng, g1):
            model.head[-1].weight.copy_(probe_weight)
            model.head[-1].bias.copy_(probe_bias)
    with torch.inference_mode():
        output = ng(*inputs)
        reference = reference_zero_message(g1, inputs)
        probe_diffs = maximum_output_difference(output, reference)
        assert max(probe_diffs.values()) < 1e-6
        altered = [value.clone() for value in inputs]
        # Separate numeric sensitivity probe only. Baseline/training graph data
        # are never altered to construct NoGraph; complete shapes/counts persist.
        altered[0][:, 1:] = fixture(seed=seed + 28)[0][:, 1:]
        altered[1] = fixture(seed=seed + 29)[1]
        altered[3] = fixture(seed=seed + 30)[3]
        altered[4] = fixture(seed=seed + 31)[4]
        altered_output = ng(*altered)
        independent = maximum_output_difference(output, altered_output)
        assert all(value == 0. for value in independent.values())
        target_changed = [value.clone() for value in inputs]
        target_changed[0][:, 0] += .25 * fixture(seed=seed + 32)[0][:, 0]
        target_difference = float((ng(*target_changed)["mode_logits"] - output["mode_logits"]).abs().max())
        assert target_difference > 0.
        residual_inputs = list(inputs)
        residual_inputs[-1] = inputs[-1] + .125
        residual_output = ng(*residual_inputs)
        residual_max_diff = float((residual_output["mode_logits"] - output["mode_logits"] - .125).abs().max())
        assert residual_max_diff < 1e-6
    assert input_before == [tensor_sha(value) for value in inputs]
    ng.train()
    output = ng(*inputs)
    loss = output["mode_logits"].square().mean()
    loss.backward()
    gradients = {}
    for name, parameter in ng.named_parameters():
        assert name.startswith(shared)
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        gradients[name] = {
            "Finite": True,
            "Elements": parameter.numel(),
            "NonzeroElements": int(torch.count_nonzero(parameter.grad)),
        }
    assert any(value["NonzeroElements"] > 0 for name, value in gradients.items() if name.startswith("node_encoder."))
    assert all(parameter.grad is None for parameter in g1.parameters())
    return {
        "Seed": seed,
        "Status": "PASS",
        "SharedInitialStateBitwise": bitwise,
        "SharedModuleDefinitionsIdentical": True,
        "UnusedInteractionParametersRegistered": False,
        "InitialFinalHeadExactlyZero": True,
        "InitialOutputEqualsOriginalLogits": True,
        "InitialZeroMessageOutputMaxDiff": initial_diffs,
        "NonzeroHeadProbeZeroMessageOutputMaxDiff": probe_diffs,
        "NonzeroHeadProbeNeighborAndEdgeChangesMaxDiff": independent,
        "TargetFeatureChangeMaxLogitDiff": target_difference,
        "OriginalLogitResidualMaxDiff": residual_max_diff,
        "CompleteInputUnmodified": True,
        "CompleteInputSHA256": input_before,
        "GradientOnlySharedRankingModules": True,
        "GradientChecks": gradients,
        "OptimizerConstructed": False,
        "TrainingPerformed": False,
    }


def compute_audit():
    inputs = fixture(batch=128, seed=1414)
    result = {}
    for name, model in (("NoGraph", NoGraphReranker(2022)), ("G1", SparseGraphReranker("G1", 2022))):
        model.eval()
        linear_counts = {"Calls": 0, "MultiplyAdds": 0, "OutputElements": 0}
        handles = []
        def count_linear(module, args, output):
            rows = args[0].numel() // module.in_features
            linear_counts["Calls"] += 1
            linear_counts["MultiplyAdds"] += rows * module.in_features * module.out_features
            linear_counts["OutputElements"] += output.numel()
        for module in model.modules():
            if isinstance(module, nn.Linear):
                handles.append(module.register_forward_hook(count_linear))
        with torch.inference_mode():
            model(*inputs)
        for handle in handles:
            handle.remove()
        times = []
        with torch.inference_mode():
            for _ in range(5):
                model(*inputs)
            for _ in range(20):
                start = time.perf_counter_ns()
                model(*inputs)
                times.append((time.perf_counter_ns() - start) / 1e6)
        result[name] = {
            **parameter_counts(model),
            "LinearMultiplyAddsPer128Targets": linear_counts["MultiplyAdds"],
            "LinearFLOPsPer128Targets": 2 * linear_counts["MultiplyAdds"],
            "LinearCalls": linear_counts["Calls"],
            "LinearOutputElements": linear_counts["OutputElements"],
            "CPUHeadForwardMedianMilliseconds": float(np.median(times)),
            "CPUHeadForwardP10Milliseconds": float(np.quantile(times, .1)),
            "CPUHeadForwardP90Milliseconds": float(np.quantile(times, .9)),
            "Measurements": len(times),
            "Warmups": 5,
        }
    result["NoGraphToG1LinearFLOPsRatio"] = result["NoGraph"]["LinearFLOPsPer128Targets"] / result["G1"]["LinearFLOPsPer128Targets"]
    result["Scope"] = "CPU single-thread, FP32, 128 targets; complete 8-neighbor / 6-mode inputs. Head forward only. Excludes packing, transfer, frozen candidate generator, and non-linear operation FLOPs. No GPU throughput claim."
    return result


def main():
    torch.set_num_threads(1)
    seeds = [2022, 2122, 2222]
    audits = [seed_audit(seed) for seed in seeds]
    compute = compute_audit()
    source = Path(sys.modules["stage8a0c_model"].__file__).resolve()
    report = {
        "Stage": "Stage14A",
        "Status": "PASS",
        "NoGraphImplementationAudit": "PASS",
        "API": "NoGraphReranker(seed); forward(original seven G1 arguments)",
        "SharedModules": ["node_encoder", "norm", "head"],
        "NodeEmbeddingDimension": 64,
        "InteractionMessage": "exact zero, no interaction network registered or executed",
        "GraphInputsDestroyedOrRemovedForAblation": False,
        "OriginalG1Source": str(source.relative_to(ROOT.parents[1])),
        "OriginalG1SourceSHA256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "FoldSeeds": seeds,
        "SeedAudits": audits,
        "InferenceCompute": compute,
        "DataSource": "Synthetic full-shape, full-neighbor fixtures only; no dataset read",
        "OuterTestRead": False,
        "VALOrTestRead": False,
        "CheckpointReadOrWritten": False,
        "TrainingPerformed": False,
        "AdditionalRequiredPreflight": ["Protocol/checkpoint/split freeze", "Real-data observation-only and loss integrity", "Tiny overfit", "Candidate identity", "Gradients only new ranking network"],
    }
    output = ROOT / "01_preflight/stage14a_model_integrity.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"Status": "PASS", "NoGraphParameters": 7425, "G1Parameters": 24066,
                      "Output": str(output), "InferenceCompute": compute}, ensure_ascii=False))


if __name__ == "__main__":
    main()
