"""Read-only numerical diagnosis after the tiny-overfit gate fails; no training."""
import json
from pathlib import Path

import numpy as np
import torch

from metrics.hivt_forecasting import aggregate, metric_records, multimodal_errors
from models.constant_velocity import constant_velocity
from models.hivt_nuscenes import HiVTNuScenesVehicle
from preprocessing.common import PROJECT_ROOT, write_json
from scripts.train_hivt_stage2 import MODEL_KEYS, save_cases


@torch.no_grad()
def main():
    torch.set_num_threads(4)
    directory = PROJECT_ROOT/"outputs/stage2/tiny_overfit"
    summary = json.loads((directory/"metrics.json").read_text())
    assert not summary["OVERFIT"], "This diagnostic is intended for a failed gate"
    checkpoint = torch.load(directory/"checkpoint.pt", weights_only=False, map_location="cpu")
    config = checkpoint["config"]
    corpus = torch.load(PROJECT_ROOT/"outputs/stage2/mini/processed/graphs.pt", weights_only=False)
    indices = np.linspace(0, len(corpus["train"])-1, config["windows"], dtype=int)
    graphs = [corpus["train"][i] for i in indices]
    model = HiVTNuScenesVehicle(**{key: config[key] for key in MODEL_KEYS}).cuda().eval()
    model.load_state_dict(checkpoint["state_dict"])
    records, cv_records, per_actor = [], [], []
    invariance_errors, selected_losses, target_sizes, predicted_sizes, scales = [], [], [], [], []
    for graph in graphs:
        data = graph.clone().to("cuda")
        output = model(data)
        pred = model.ego_predictions(output, data)
        errors = multimodal_errors(pred, data.positions[:, 5:], data.future_mask, data.target_mask)
        records.extend(metric_records(errors))
        cv, valid, _ = constant_velocity(data.positions[:, :5], data.history_mask, data.history_times, data.future_times)
        cv_errors = multimodal_errors(cv[:, None], data.positions[:, 5:], data.future_mask, data.target_mask & valid)
        cv_records.extend(metric_records(cv_errors))
        local_target = torch.bmm(data.y, output["rotation"])
        raw = output["raw_prediction"][..., :2].permute(1, 0, 2, 3)
        local_error = torch.linalg.vector_norm(raw-local_target[:, None], dim=-1)
        ego_error = torch.linalg.vector_norm(pred-data.positions[:, None, 5:], dim=-1)
        active = data.future_mask[:, None].expand_as(local_error)
        invariance_errors.append(float((local_error[active]-ego_error[active]).abs().max()))
        target_sizes.extend(torch.linalg.vector_norm(data.y[data.future_mask], dim=-1).tolist())
        predicted_sizes.extend(torch.linalg.vector_norm(raw[active], dim=-1).tolist())
        raw_scale = output["raw_prediction"][..., 2:].permute(1, 0, 2, 3)
        scales.extend(raw_scale[active].flatten().tolist())
        values = model.loss(output, data)
        selected_losses.append({k: float(v) for k, v in values.items()})
        for node in torch.where(errors["full_horizon"])[0].tolist():
            last = torch.linalg.vector_norm(data.positions[node, -1]-data.positions[node, 4]).item()
            per_actor.append({"scene_token": graph.scene_token, "sample_token": graph.sample_token,
                              "instance_token": graph.instance_tokens[node], "gt_endpoint_displacement_m": last,
                              "minADE_K": float(errors["minADE_K"][node]), "minFDE_K": float(errors["minFDE_K"][node]),
                              "cv_ADE": float(cv_errors["minADE_K"][node]), "cv_FDE": float(cv_errors["minFDE_K"][node])})
    moving = [r for r in per_actor if r["gt_endpoint_displacement_m"] > 2]
    stationary = [r for r in per_actor if r["gt_endpoint_displacement_m"] <= 2]
    result = {"status": "OVERFIT_FAIL", "subsequent_training_started": False,
              "local_vs_ego_displacement_error_max_difference_m": max(invariance_errors),
              "target_displacement_max_m": max(target_sizes), "prediction_displacement_max_m": max(predicted_sizes),
              "laplace_scale_min": min(scales), "laplace_scale_max": max(scales),
              "same_tiny_windows_HiVT": aggregate(records), "same_tiny_windows_CV": aggregate(cv_records),
              "moving_full_horizon": {"count": len(moving), "mean_minADE_K": float(np.mean([r["minADE_K"] for r in moving])),
                                      "mean_minFDE_K": float(np.mean([r["minFDE_K"] for r in moving]))},
              "stationary_full_horizon": {"count": len(stationary), "mean_minADE_K": float(np.mean([r["minADE_K"] for r in stationary])),
                                          "mean_minFDE_K": float(np.mean([r["minFDE_K"] for r in stationary]))},
              "findings": ["Loss falls substantially but trajectory errors fail the preregistered overfit gate.",
                           "Moving-target endpoint errors remain high; loss decline alone is insufficient to claim memorization.",
                           "Local-agent and t0-ego error norms agree; no inverse-rotation mismatch identified.",
                           "Mask/batch/source tests passed, upstream source hashes verified, losses unchanged.",
                           "Learning schedule reaches near-zero LR at 240 epochs; insufficient optimization is a hypothesis, not a proven sole cause."],
              "next_checks": ["Inspect moving-actor decoder location and scale gradients, learning-rate schedule, and multimodal selection before another tiny-only run.",
                              "Do not run mini training or Stage3 until a new tiny overfit run passes."],
              "MR_definition": "best of six final displacements >2m, upstream HiVT; full/partial horizons separated"}
    assert result["local_vs_ego_displacement_error_max_difference_m"] < 1e-4
    write_json(PROJECT_ROOT/"outputs/reports/hivt_overfit_diagnosis.json", result)
    write_json(PROJECT_ROOT/"outputs/reports/stage2/tiny_actor_errors.json", per_actor)
    save_cases(model, graphs, PROJECT_ROOT/"outputs/figures/stage2/tiny_failure", "failure", count=5)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
