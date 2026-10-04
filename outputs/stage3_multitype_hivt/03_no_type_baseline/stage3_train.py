"""Bounded GPU benchmark and resumable optimizer-step-controlled Protocol 1."""
import argparse
import csv
import gc
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "00_manifest"))
from stage3_common import (CONFIG, GROUPS, SceneShardDataset, SceneShuffleSampler, atomic_json, config, evaluate,
                           git, model_new, model_input, loss_diagnostics, CLASSES, read_json, seed_all, sha256, update_manifest, verify_previous, write_csv)
import torch
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def finite_gradients(model):
    norms = torch.stack([p.grad.norm() for p in model.parameters() if p.grad is not None])
    assert torch.isfinite(norms).all(), "Nonfinite training gradient"


def benchmark_trial(graphs, batch_size):
    model = optimizer = data = output = values = None
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    try:
        model = model_new(); model.train(); optimizer = model.optimizer(.0001, config()["weight_decay"])
        data = Batch.from_data_list(graphs[:batch_size]).cuda(); times = []
        for i in range(3):
            optimizer.zero_grad(set_to_none=True); torch.cuda.synchronize(); start = time.monotonic()
            output = model(model_input(data)); values = loss_diagnostics(model, output, data, "original_nll")
            assert torch.isfinite(values["loss"]); values["loss"].backward(); finite_gradients(model); optimizer.step()
            torch.cuda.synchronize(); times.append(time.monotonic()-start)
        total = torch.cuda.get_device_properties(0).total_memory
        allocated = torch.cuda.max_memory_allocated(); reserved = torch.cuda.max_memory_reserved()
        row = {"batch_size": batch_size, "status": "PASS", "optimizer_steps": 3, "peak_allocated_bytes": allocated,
               "peak_reserved_bytes": reserved, "total_device_bytes": total, "peak_reserved_fraction": reserved/total,
               "iteration_seconds": times, "mean_steady_iteration_seconds": sum(times[1:])/2,
               "stable_below_90_percent": reserved/total < .90, "NaN": 0, "Inf": 0}
    except torch.cuda.OutOfMemoryError as error:
        row = {"batch_size": batch_size, "status": "OOM", "error": str(error), "elapsed_seconds": time.monotonic()-started}
    finally:
        del model, optimizer, data, output, values; gc.collect(); torch.cuda.empty_cache()
    return row


def benchmark():
    manifest = read_json(ROOT / "02_preprocessed/stage3_preprocess_manifest.json")
    assert manifest["status"] == "COMPLETE" and manifest["scope"] == "full official trainval"
    ds = SceneShardDataset("train")
    # Stress-test the largest-context windows instead of choosing only easy scenes.
    order = sorted(range(len(ds)), key=lambda i: int(ds.rows[i]["actor_count"]), reverse=True)[:16]
    graphs = [ds[i] for i in order]; rows = []
    for batch_size in (16, 8, 4):
        row = benchmark_trial(graphs, batch_size); rows.append(row)
        print("BATCH_BENCHMARK", row, flush=True)
    stable = [r["batch_size"] for r in rows if r["status"] == "PASS" and r["stable_below_90_percent"]]
    assert stable, "No authorized batch size stable below 90%; inspect benchmark before training"
    selected = max(stable); c = config(); c["batch_size"] = selected
    CONFIG.write_text(yaml.safe_dump(c, sort_keys=False))
    atomic_json(ROOT / "00_manifest/stage3_batch_benchmark.json", {"trials": rows, "selected_batch_size": selected,
                "selection": "largest stable tested batch below 90% torch reserved VRAM in high-context train windows",
                "stress_vehicle_counts": [g.num_nodes for g in graphs], "model_changed": False,
                "baseline_gpu_memory_note": "CUDA display/process allocations outside PyTorch remain separate"})
    atomic_json(ROOT / "00_manifest/stage3_training_plan.json", {"train_windows": len(ds), "batch_size": selected,
                "steps_per_epoch": math.ceil(len(ds)/selected), "config_sha256": sha256(CONFIG), "protocol": 1,
                "warmup_minimum_steps": c["warmup"]["minimum_steps"], "warmup_maximum_steps": c["warmup"]["maximum_steps"],
                "nll_maximum_steps": c["nll"]["maximum_steps"], "validation_interval": c["validation_interval_steps"],
                "sampling": c["data_loader"]["shuffle"], "primary_checkpoint_selection": c["checkpoint_selection"]})
    ds.clear(); verify_previous(); update_manifest()


def checkpoint_save(path, model, optimizer, metadata, iterator, state):
    temp = path.with_name(path.name + ".tmp")
    torch.save({"state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "metadata": metadata,
                "iterator": iterator, "phase_state": state, "config": config(), "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all()}, temp); os.replace(temp, path)
    manifest_path = ROOT / "00_manifest/stage3_checkpoint_manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {"checkpoints": {}}
    manifest["checkpoints"][str(path.relative_to(ROOT))] = {**metadata, "sha256": sha256(path), "file_size": path.stat().st_size, "git_tracked": False}
    atomic_json(manifest_path, manifest)


def checkpoint_restore(path, lr):
    saved = torch.load(path, weights_only=False, map_location="cpu")
    assert saved["metadata"]["config_sha256"] == sha256(CONFIG), "Frozen training config changed"
    model = model_new(); model.load_state_dict(saved["state_dict"])
    optimizer = model.optimizer(lr, config()["weight_decay"]); optimizer.load_state_dict(saved["optimizer_state_dict"])
    for group in optimizer.param_groups: group["lr"] = lr
    torch.set_rng_state(saved["torch_rng"]); torch.cuda.set_rng_state_all(saved["cuda_rng"])
    return model, optimizer, saved


def plot_curve(rows, path):
    if not rows: return
    previous = ROOT / "03_no_type_baseline/stage3_warmup_curve.csv"
    if "nll" in path.name and previous.exists():
        with open(previous) as handle:
            warm=[{k:float(v) if v else None for k,v in r.items()} for r in csv.DictReader(handle)]
        rows=warm+rows
    write_csv(ROOT / "03_no_type_baseline/stage3_no_type_train_curve.csv",rows)
    with open(ROOT / "03_no_type_baseline/stage3_no_type_train_curve.csv") as handle:
        rows=[{k:float(v) if v else None for k,v in r.items()} for r in csv.DictReader(handle)]
    for category,keys in (("loss",("train_loss","regression_loss","classification_loss")),
                          ("val_fde",("VAL_overall_FDE","VAL_vehicle_FDE","VAL_pedestrian_FDE","VAL_bicycle_FDE"))):
        fig,ax=plt.subplots(figsize=(9,4),layout="constrained")
        for key in keys:ax.plot([r['global_step'] for r in rows],[r[key] for r in rows],marker='o',ms=3,label=key)
        ax.set_xlabel('Cumulative executed optimizer step');ax.set_ylabel('Loss' if category=='loss' else 'Full-horizon VAL FDE (m)')
        ax.grid(alpha=.15);ax.legend(fontsize=9)
        fig.savefig(ROOT / f"03_no_type_baseline/stage3_no_type_{category}_curve.png",dpi=200);plt.close(fig)


def train_phase(phase, model=None, optimizer=None, warmup_steps=0, warmup_source_step=None):
    c = config(); train = SceneShardDataset("train"); val = SceneShardDataset("val")
    settings = c["warmup" if phase == "fixed_scale" else "nll"]; label = "warmup" if phase == "fixed_scale" else "nll"
    plan = read_json(ROOT / "00_manifest/stage3_training_plan.json"); steps_per_epoch = plan["steps_per_epoch"]
    phase_config = ROOT / f"03_no_type_baseline/stage3_{label}_config.yaml"
    if not phase_config.exists(): phase_config.write_text(yaml.safe_dump({**c, "active_phase": phase, "phase_settings": settings,
                                "steps_per_epoch": steps_per_epoch, "data_manifest_sha256": sha256(ROOT / "02_preprocessed/stage3_preprocess_manifest.json")}, sort_keys=False))
    resume_path = ROOT / f"07_checkpoints/stage3_{label}_last_checkpoint.pt"
    state = {"phase_step": 0, "best_overall": float("inf"), "best_moving": float("inf"), "bad_validations": 0,
             "warmup_executed_steps": warmup_steps, "warmup_source_step": warmup_source_step, "completed": False}
    iterator = {"epoch": 0, "next_batch": 0}; rows = []
    curve_path = ROOT / f"03_no_type_baseline/stage3_{label}_curve.csv"
    if resume_path.exists():
        model, optimizer, saved = checkpoint_restore(resume_path, settings["lr"])
        state = saved["phase_state"]; iterator = saved["iterator"]
        if curve_path.exists():
            with open(curve_path) as f:
                rows = [{k: float(v) if v else None for k,v in r.items()} for r in csv.DictReader(f)
                        if int(float(r["phase_step"])) <= state["phase_step"]]
        print("RESUME_TRAIN", label, state["phase_step"], iterator, flush=True)
    elif model is None:
        model = model_new(); optimizer = model.optimizer(settings["lr"], c["weight_decay"])
    summary_path = ROOT / f"03_no_type_baseline/stage3_{label}_summary.json"
    if state["completed"] and summary_path.exists():
        return read_json(summary_path)
    lr = settings["lr"]
    for g in optimizer.param_groups: g["lr"] = lr
    sampler = SceneShuffleSampler(train, c["seed"]); running = {k: 0. for k in ("loss", "regression_loss", "classification_loss", "NLL")}; class_sums={c:0. for c in CLASSES}; class_batches={c:0 for c in CLASSES}; running_count=0
    started = time.monotonic(); stop = bool(state["completed"])
    last_validation = evaluate(val, model, phase) if stop else None
    while state["phase_step"] < settings["maximum_steps"] and not stop:
        sampler.epoch = iterator["epoch"] + (100000 if label == "nll" else 0)
        loader = DataLoader(train, batch_size=c["batch_size"], sampler=sampler, num_workers=0)
        for batch_index, batch in enumerate(loader):
            if batch_index < iterator["next_batch"]: continue
            model.train(); data = batch.cuda(); optimizer.zero_grad(set_to_none=True)
            output = model(model_input(data)); values = loss_diagnostics(model, output, data, phase)
            assert all(torch.isfinite(v) for v in values.values())
            values["loss"].backward(); finite_gradients(model); optimizer.step()
            state["phase_step"] += 1; iterator["next_batch"] = batch_index + 1
            for k in running: running[k] += float(values[k].detach())
            for name in CLASSES:
                if name+"_regression_loss" in values:
                    class_sums[name]+=float(values[name+"_regression_loss"]); class_batches[name]+=1
            running_count += 1; global_step = state["warmup_executed_steps"] + state["phase_step"]
            if state["phase_step"] % 100 == 0: print("TRAIN_STEP", label, global_step, "phase", state["phase_step"], flush=True)
            if state["phase_step"] % c["validation_interval_steps"] == 0 or state["phase_step"] == settings["maximum_steps"]:
                last_validation = evaluate(val, model, phase)
                last_validation = {k: v for k, v in last_validation.items() if k != "scenes"}
                full = last_validation["metrics"]["full_horizon"]; overall = full["overall"]; moving = full["vehicle.moving"]
                row = {"global_step": global_step, "phase_step": state["phase_step"], "epoch": iterator["epoch"]+iterator["next_batch"]/steps_per_epoch,
                       "learning_rate": lr, "train_loss": running["loss"]/running_count,
                       "regression_loss": running["regression_loss"]/running_count, "classification_loss": running["classification_loss"]/running_count, "NLL":running["NLL"]/running_count}
                for name in CLASSES: row[name+"_regression_loss"] = class_sums[name]/class_batches[name] if class_batches[name] else None
                for group in GROUPS:
                    tag = group.split(".")[-1] if group != "vehicle.moving" else "moving"
                    for field, suffix in (("minADE6", "ADE"), ("minFDE6", "FDE"), ("MR6", "MR")):
                        row[f"VAL_{tag}_{suffix}"] = full[group][field]
                rows.append(row); write_csv(curve_path, rows); plot_curve(rows, ROOT / f"03_no_type_baseline/stage3_{label}_curve.png")
                print("VALIDATION", label, row, flush=True)
                atomic_json(ROOT / f"04_evaluation/stage3_{label}_val_step_{state['phase_step']:05d}.json", last_validation)
                improved = overall["minFDE6"] < state["best_overall"]
                improved_moving = moving["minFDE6"] < state["best_moving"]
                state["bad_validations"] = 0 if improved else state["bad_validations"]+1
                if improved: state["best_overall"] = overall["minFDE6"]
                if improved_moving: state["best_moving"] = moving["minFDE6"]
                minimum = settings.get("minimum_steps", 0)
                stop = (state["phase_step"] >= minimum and state["bad_validations"] >= settings["plateau_patience"]) or state["phase_step"] >= settings["maximum_steps"]
                state["completed"] = stop
                metadata = {"epoch": row["epoch"], "global_step": global_step, "phase_step": state["phase_step"], "phase": phase,
                            "LR": lr, "validation_ADE": overall["minADE6"], "validation_FDE": overall["minFDE6"], "validation_MR": overall["MR6"],
                            "moving_ADE": moving["minADE6"], "moving_FDE": moving["minFDE6"], "git_commit_SHA": git("rev-parse", "HEAD"),
                            "config_sha256": sha256(CONFIG), "phase_config_sha256": sha256(phase_config),
                            "selection_split": "official val", "selection_horizon": "full_horizon", "optimizer_state_preserved": label=="nll",
                            "warmup_source_step": state["warmup_source_step"], "warmup_executed_steps": state["warmup_executed_steps"]}
                metadata["per_class_full_horizon_metrics"] = {name: full[name] for name in CLASSES}
                if improved:
                    filename = "stage3_warmup_best_overall.pt" if label=="warmup" else "stage3_best_overall_minfde.pt"
                    checkpoint_save(ROOT / "07_checkpoints" / filename, model, optimizer, {**metadata, "selection_metric": "overall minFDE6"}, iterator, state)
                if improved_moving:
                    filename = "stage3_warmup_best_moving.pt" if label=="warmup" else "stage3_best_moving_minfde.pt"
                    checkpoint_save(ROOT / "07_checkpoints" / filename, model, optimizer, {**metadata, "selection_metric": "moving minFDE6; diagnostic only"}, iterator, state)
                checkpoint_save(resume_path, model, optimizer, {**metadata, "selection_metric": "last completed validation"}, iterator, state)
                checkpoint_save(ROOT / "07_checkpoints/stage3_last_checkpoint.pt", model, optimizer, {**metadata, "selection_metric": "last completed validation"}, iterator, state)
                running = {k: 0. for k in running}; class_sums={c:0. for c in CLASSES}; class_batches={c:0 for c in CLASSES}; running_count = 0; update_manifest()
                if stop: break
        if not stop: iterator["epoch"] += 1; iterator["next_batch"] = 0
    summary = {"phase": phase, "phase_steps": state["phase_step"], "warmup_executed_steps": state["warmup_executed_steps"],
               "warmup_source_step": state["warmup_source_step"], "best_overall_FDE": state["best_overall"], "best_moving_FDE": state["best_moving"],
               "early_stop": state["phase_step"] < settings["maximum_steps"], "bad_validations": state["bad_validations"],
               "elapsed_seconds_this_invocation": time.monotonic()-started, "status": "COMPLETE", "NaN": 0, "Inf": 0,
               "validation_windows": len(val), "training_windows": len(train), "test_used": False, "last_validation": last_validation}
    atomic_json(ROOT / f"03_no_type_baseline/stage3_{label}_summary.json", summary)
    verify_previous(); update_manifest(); print("PHASE_COMPLETE", label, summary["phase_steps"], flush=True)
    return summary


def train(stage):
    assert read_json(ROOT / "03_no_type_baseline/stage3_tiny_overfit.json")["status"] == "PASS"
    assert read_json(ROOT / "02_preprocessed/stage3_preprocess_manifest.json")["scope"] == "full official trainval"
    assert read_json(ROOT / "00_manifest/stage3_batch_benchmark.json")["selected_batch_size"] == config()["batch_size"]
    if stage == "warmup": train_phase("fixed_scale")
    else:
        warm = read_json(ROOT / "03_no_type_baseline/stage3_warmup_summary.json"); assert warm["status"] == "COMPLETE"
        model, optimizer, saved = checkpoint_restore(ROOT / "07_checkpoints/stage3_warmup_best_overall.pt", config()["nll"]["lr"])
        train_phase("original_nll", model, optimizer, warmup_steps=warm["phase_steps"], warmup_source_step=saved["metadata"]["global_step"])


if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--stage", choices=("benchmark", "warmup", "nll"), required=True); a=p.parse_args()
    torch.set_num_threads(4)
    if a.stage == "benchmark": benchmark()
    else: train(a.stage)
