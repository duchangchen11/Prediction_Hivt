"""Explicit fold/seed/list entry. Stage15A only permits bounded preflight updates."""
import argparse
import copy
import hashlib
import math
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "00_manifest"))
from stage15a_common import (PROTOCOL, FoldContext, FoldDataset, atomic_json, model_input,
    model_new, next_training_batch, read_json, sha256, state_digest, evaluate_dev)
import numpy as np
import torch


def config_for(context, scope="PREFLIGHT_ONLY_NOT_ELIGIBLE_FOR_FORMAL_TRAINING"):
    return {"protocol_sha256": sha256(PROTOCOL), "fold": context.fold, "seed": context.seed,
            "lists": {k: context.record["files"][k] for k in ("InnerTrain", "InnerDev")},
            "output_dir": str(context.output), "precision": "FP32", "scope": scope,
            "fitting_source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in (
                ROOT / "00_manifest/stage15a_common.py", ROOT / "02_training/stage15a_train.py")}}


def checkpoint_save(path, model, optimizer, context, state, iterator, scope="PREFLIGHT_ONLY_NOT_ELIGIBLE_FOR_FORMAL_TRAINING"):
    path = Path(path)
    assert path.resolve().is_relative_to(context.output)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save({"state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                "config": config_for(context, scope), "phase_state": copy.deepcopy(state), "iterator": dict(iterator),
                "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
                "python_rng": random.getstate(), "numpy_rng": np.random.get_state()}, temp)
    os.replace(temp, path)


def restore(path, context, scope="PREFLIGHT_ONLY_NOT_ELIGIBLE_FOR_FORMAL_TRAINING"):
    saved = context.load(path, "preflight_checkpoint")
    assert saved["config"] == config_for(context, scope), "Wrong fold/seed/list/source/scope checkpoint"
    model = model_new(context.seed)
    model.load_state_dict(saved["state_dict"], strict=True)
    lr = saved["optimizer_state_dict"]["param_groups"][0]["lr"]
    optimizer = model.optimizer(lr, context.protocol["weight_decay"])
    optimizer.load_state_dict(saved["optimizer_state_dict"])
    assert state_digest(model.state_dict()) == state_digest(saved["state_dict"])
    torch.set_rng_state(saved["torch_rng"])
    torch.cuda.set_rng_state_all(saved["cuda_rng"])
    random.setstate(saved["python_rng"])
    np.random.set_state(saved["numpy_rng"])
    return model, optimizer, saved


def update_selection(state, fde, protocol_step_index, phase, budget):
    assert math.isfinite(fde)
    improved = state["best_FDE"] is None or fde < state["best_FDE"]
    state["bad_validations"] = 0 if improved else state["bad_validations"] + 1
    if improved:
        state.update(best_FDE=fde, best_step=protocol_step_index)
    state["completed"] = state["phase_step"] >= budget or (phase == "original_nll" and state["bad_validations"] >= 5)
    return improved


def new_phase_state():
    return {"phase_step": 0, "best_FDE": None, "best_step": None, "bad_validations": 0, "completed": False}


def optimize_step(model, optimizer, dataset, iterator, phase):
    assert phase in ("fixed_scale", "original_nll") and dataset.role == "InnerTrain"
    batch, ids = next_training_batch(dataset, iterator, phase)
    data = batch.cuda()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    out = model(model_input(data))
    values = model.recovery_loss(out, data, phase)
    assert all(torch.isfinite(v) for v in values.values()), "Nonfinite loss"
    values["loss"].backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None), "Nonfinite gradient"
    grad = float(model.type_embedding.weight.grad.norm())
    assert math.isfinite(grad) and grad > 0
    adapter_grads = {n: float(p.grad.norm()) if p.grad is not None else 0. for n, p in model.named_parameters()
                     if n.startswith(("decoder.experts.", "decoder.router."))}
    optimizer.step()
    iterator["next_batch"] += 1
    return {"phase": phase, "loss": float(values["loss"].detach()),
            "regression_loss": float(values["regression_loss"].detach()),
            "classification_loss": float(values["classification_loss"].detach()),
            "NLL": float(values["NLL"].detach()), "type_gradient": grad,
            "adapter_gradients": adapter_grads, "dataset_indices": ids,
            "scene_tokens": list(batch.scene_token), "sample_tokens": list(batch.sample_token)}


def transition_from_own_warmup(path, context, scope="PREFLIGHT_ONLY_NOT_ELIGIBLE_FOR_FORMAL_TRAINING"):
    model, optimizer, saved = restore(path, context, scope)
    assert saved["phase_state"]["phase"] == "fixed_scale"
    if scope == "FORMAL_SCENE_ISOLATED":
        assert saved["phase_state"]["phase_step"] <= 5000 and saved["phase_state"].get("preflight_probe") is not True
    before = copy.deepcopy(optimizer.state_dict())
    for group in optimizer.param_groups:
        group["lr"] = .0001
    after = optimizer.state_dict()
    assert state_digest({str(i) + "/" + k: v for i, row in before["state"].items() for k, v in row.items() if torch.is_tensor(v)}) == state_digest(
        {str(i) + "/" + k: v for i, row in after["state"].items() for k, v in row.items() if torch.is_tensor(v)})
    a, b = copy.deepcopy(before["param_groups"]), copy.deepcopy(after["param_groups"])
    for g in a + b:
        g.pop("lr")
    assert a == b
    return model, optimizer, {"epoch": 0, "next_batch": 0}, {
        "Status": "PASS", "OnlyLRChanged": True, "OptimizerMomentsPreserved": True,
        "AllRNGRestored": True, "SamplerCursorReset": True, "NLLEpochOffset": 100000,
        "OwnWarmupCheckpoint": str(Path(path).relative_to(ROOT)), "SHA256": sha256(path),
        "WarmupActualSteps": saved["phase_state"]["phase_step"], "FormalWarmupWouldRequire": 5000}


def train_registered_phase(context, phase):
    """Complete frozen 5000→16000 loop; authorization gate is closed in Stage15A.

    This function is present for review and consistency checks. No CLI or
    preflight call in this stage can activate it. A later authorized stage must
    register its formal sources, open its own gate, and start from fresh state.
    """
    if not context.protocol["FullTrainingAuthorized"]:
        raise RuntimeError("Formal fitting is disabled by the Stage15A registration")
    assert phase in ("fixed_scale", "original_nll")
    scope = "FORMAL_SCENE_ISOLATED"
    warm = phase == "fixed_scale"
    label = "warmup" if warm else "nll"
    budget, offset, lr = (5000, 0, .001) if warm else (16000, 5000, .0001)
    last = context.output / f"stage15a_{label}_last_checkpoint.pt"
    best = context.output / f"stage15a_{label}_best_overall_minfde.pt"
    warm_best = context.output / "stage15a_warmup_best_overall_minfde.pt"
    state = new_phase_state()
    state["phase"] = phase
    cursor = {"epoch": 0, "next_batch": 0}
    if last.exists():
        model, optimizer, saved = restore(last, context, scope)
        state, cursor = saved["phase_state"], saved["iterator"]
    elif warm:
        model = model_new(context.seed)
        optimizer = model.optimizer(lr, .0001)
    else:
        summary = read_json(context.output / "stage15a_warmup_summary.json")
        assert summary["phase_steps"] == 5000 and summary["status"] == "COMPLETE"
        model, optimizer, cursor, transition = transition_from_own_warmup(warm_best, context, scope)
        atomic_json(context.output / "stage15a_formal_transition.json", transition)
    assert all(g["lr"] == lr for g in optimizer.param_groups)
    train, dev = FoldDataset(context, "InnerTrain"), FoldDataset(context, "InnerDev")
    started = time.monotonic()
    while state["phase_step"] < budget and not state["completed"]:
        optimize_step(model, optimizer, train, cursor, phase)
        state["phase_step"] += 1
        global_step = offset + state["phase_step"]
        assert global_step <= 21000
        if state["phase_step"] % 500 == 0:
            measured = evaluate_dev(model, dev)
            assert measured["subset_only"] is False
            improved = update_selection(state, measured["full_horizon_Overall_minFDE6"], global_step, phase, budget)
            state["last_development"] = measured
            # Checkpoint is authoritative. A crash after checkpoint but before
            # a sidecar write is repaired from its state on deterministic resume.
            if improved:
                checkpoint_save(best, model, optimizer, context, state, cursor, scope)
            checkpoint_save(last, model, optimizer, context, state, cursor, scope)
            atomic_json(context.output / f"stage15a_dev_step_{global_step:05d}.json", {
                "phase": phase, "global_step": global_step, "measured": measured,
                "improved": improved, "best_step": state["best_step"], "bad_validations": state["bad_validations"]})
    summary = {"status": "COMPLETE", "phase_steps": state["phase_step"], "phase": phase,
        "global_executed_step": offset + state["phase_step"], "best_global_step": state["best_step"],
        "best_overall_FDE": state["best_FDE"], "consecutive_nonimprovements": state["bad_validations"],
        "stop_reason": "warmup_fixed5000" if warm else ("patience_5" if state["bad_validations"] >= 5 else "global21000_budget"),
        "selected_checkpoint": str(best), "last_checkpoint_sha256": sha256(last),
        "elapsed_seconds_this_invocation": time.monotonic() - started}
    atomic_json(context.output / f"stage15a_{label}_summary.json", summary)
    del model, optimizer
    train.clear()
    dev.clear()
    torch.cuda.empty_cache()
    return summary


def run_preflight_phases(context):
    """Six primary updates; diagnostic dev checkpoints are never formal models."""
    assert not (context.output / "stage15a_preflight_warm_best.pt").exists(), "Never overwrite an existing preflight run"
    train, dev = FoldDataset(context, "InnerTrain"), FoldDataset(context, "InnerDev")
    model = model_new(context.seed)
    initial = state_digest(model.state_dict())
    optimizer = model.optimizer(.001, .0001)
    iterator = {"epoch": 0, "next_batch": 0}
    warm_state = new_phase_state()
    warm_state["phase"] = "fixed_scale"
    rows = []
    peak = {}
    adapter_max = {}
    warm_path = context.output / "stage15a_preflight_warm_best.pt"
    last_path = context.output / "stage15a_preflight_last.pt"
    for j in range(4):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        row = optimize_step(model, optimizer, train, iterator, "fixed_scale")
        torch.cuda.synchronize()
        row["wall_seconds"] = time.perf_counter() - started
        row["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        row["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        warm_state["phase_step"] += 1
        row["phase_step"] = warm_state["phase_step"]
        for n, v in row["adapter_gradients"].items():
            adapter_max[n] = max(adapter_max.get(n, 0.), v)
        rows.append(row)
    assert all(v > 0 and math.isfinite(v) for v in adapter_max.values()), "Disconnected adapter after bounded checks"
    assert state_digest(model.state_dict()) != initial
    dev_ids = list(range(4))
    selection = evaluate_dev(model, dev, dev_ids)
    # Keep the registered 5000/16000 stopping budgets; this early diagnostic
    # probe is not a validation event from the formal every500 schedule.
    update_selection(warm_state, selection["full_horizon_Overall_minFDE6"], 4, "fixed_scale", 5000)
    assert not warm_state["completed"]
    warm_state["preflight_probe"] = True
    checkpoint_save(warm_path, model, optimizer, context, warm_state, iterator)
    checkpoint_save(last_path, model, optimizer, context, warm_state, iterator)
    del model, optimizer
    torch.cuda.empty_cache()
    model, optimizer, iterator, transition = transition_from_own_warmup(warm_path, context)
    nll_state = new_phase_state()
    nll_state["phase"] = "original_nll"
    for j in range(2):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        row = optimize_step(model, optimizer, train, iterator, "original_nll")
        torch.cuda.synchronize()
        row["wall_seconds"] = time.perf_counter() - started
        row["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        row["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        nll_state["phase_step"] += 1
        row["phase_step"] = nll_state["phase_step"]
        rows.append(row)
    nll_selection = evaluate_dev(model, dev, dev_ids)
    update_selection(nll_state, nll_selection["full_horizon_Overall_minFDE6"], 5002, "original_nll", 16000)
    assert not nll_state["completed"]
    nll_state["preflight_probe"] = True
    nll_path = context.output / "stage15a_preflight_nll_best.pt"
    checkpoint_save(nll_path, model, optimizer, context, nll_state, iterator)
    checkpoint_save(last_path, model, optimizer, context, nll_state, iterator)
    result = {"Status": "PASS", "Fold": context.fold, "Seed": context.seed, "InitialStateSHA256": initial,
        "ActualPrimaryOptimizerUpdates": 6, "WarmUpdates": 4, "NLLUpdates": 2,
        "FormalWarmupExecuted": False, "FormalTrainingExecuted": False, "DevelopmentSubsetWindows": 4,
        "WarmDevDiagnostic": selection, "NLLDevDiagnostic": nll_selection,
        "AdapterMaximumGradients": adapter_max, "Transition": transition,
        "Checkpoints": {p.name: {"sha256": sha256(p), "bytes": p.stat().st_size} for p in (warm_path, nll_path, last_path)},
        "Rows": rows, "SourceConfig": config_for(context)}
    atomic_json(context.output / "stage15a_training_probe.json", result)
    train.clear()
    dev.clear()
    del model, optimizer
    torch.cuda.empty_cache()
    return result


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fold", type=int, choices=(1, 2, 3), required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--training-scenes", type=Path, required=True)
    p.add_argument("--development-scenes", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--mode", choices=("describe", "preflight", "formal"), default="describe")
    return p.parse_args()


def main():
    args = parse_args()
    if args.mode == "formal":
        raise RuntimeError("Stage15A does not authorize full predictor fitting; STOP for brainAI review")
    context = FoldContext(args.fold, args.seed, args.training_scenes, args.development_scenes, args.output_dir)
    if args.mode == "describe":
        atomic_json(context.output / "stage15a_entry_description.json", {
            "Status": "PASS", "Fold": context.fold, "Seed": context.seed, "Config": config_for(context),
            "Protocol": {k: context.protocol[k] for k in ("batch_size", "optimizer", "weight_decay", "warmup", "nll", "maximum_global_steps", "validation_interval", "selection")},
            "TrainWindows": len(FoldDataset(context, "InnerTrain")), "DevWindows": len(FoldDataset(context, "InnerDev")),
            "GraphPayloadsLoaded": 0, "FullTrainingAuthorized": False})
        print("STAGE15A_ENTRY_DESCRIPTION_PASS", args.fold)
    else:
        run_preflight_phases(context)
        print("STAGE15A_BOUNDED_TRAINING_CHECK_PASS", args.fold)


if __name__ == "__main__":
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    main()
