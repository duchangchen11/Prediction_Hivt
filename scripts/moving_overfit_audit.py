"""Conditional, controlled moving-actor diagnostics; never run full tiny/mini."""
import argparse
import copy
import csv
import hashlib
import json
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.loader import DataLoader
import yaml

from metrics.hivt_forecasting import aggregate, metric_records, multimodal_errors
from models.hivt_nuscenes import HiVTNuScenesVehicle
from preprocessing.common import PROJECT_ROOT, load_config, load_nuscenes, scene_samples, write_json
from preprocessing.coordinates import global_to_ego
from scripts.audit_vehicle_motion import AUDIT
from scripts.plot_hivt_predictions import plot_case
from scripts.train_hivt_stage2 import MODEL_KEYS


def selected_graphs(corpus, targets):
    """Change only supervision mask; keep all original nodes/edges/lanes."""
    grouped = {}
    for target in targets:
        grouped.setdefault(target["graph_index"], []).append(target)
    graphs = []
    for index, rows in grouped.items():
        original = corpus[index]
        graph = original.clone()
        graph.target_mask = torch.zeros_like(original.target_mask)
        for row in rows:
            node = row["node"]
            assert graph.instance_tokens[node] == row["instance_token"]
            assert graph.history_mask[node].all() and graph.future_mask[node].all()
            graph.target_mask[node] = True
        for key in original.keys():
            if key == "target_mask": continue
            if torch.is_tensor(original[key]):
                assert torch.equal(graph[key], original[key]), f"Context changed: {key}"
            else:
                assert graph[key] == original[key], f"Metadata changed: {key}"
        graphs.append(graph)
    assert sum(int(g.target_mask.sum()) for g in graphs) == len(targets)
    return graphs


def diagnostic_loss(model, output, data, fixed_scale):
    if not fixed_scale:
        return model.loss(output, data)
    assert model.num_modes == 1
    target = torch.bmm(data.y, output["rotation"])
    mask = data.future_mask & data.target_mask[:, None]
    location = output["raw_prediction"][0, ..., :2]
    # Exact unit-scale Laplace loss, same coordinate-wise mean reduction.
    # Scale/pi heads remain architecturally present, but do not affect this loss.
    regression = torch.abs(location[mask]-target[mask]).mean() + np.log(2.)
    return {"loss": regression, "regression_loss": regression,
            "classification_loss": regression.new_zeros(())}


@torch.no_grad()
def evaluate(model, graphs, fixed_scale=False, batch_size=4):
    model.eval()
    records, loss_sums, count, actors = [], {"loss": 0., "regression_loss": 0., "classification_loss": 0.}, 0, []
    for batch in DataLoader(graphs, batch_size=batch_size, shuffle=False):
        data = batch.to("cuda")
        output = model(data)
        values = diagnostic_loss(model, output, data, fixed_scale)
        assert all(torch.isfinite(v) for v in values.values())
        n = int(data.target_mask.sum()); count += n
        for key in loss_sums: loss_sums[key] += float(values[key])*n
        prediction = model.ego_predictions(output, data)
        errors = multimodal_errors(prediction, data.positions[:, 5:], data.future_mask, data.target_mask)
        records.extend(metric_records(errors))
        target = torch.bmm(data.y, output["rotation"])
        raw = output["raw_prediction"]
        distance = torch.linalg.vector_norm(raw[..., :2]-target[None], dim=-1)
        best_l2 = distance.sum(dim=-1).argmin(dim=0)
        for node in torch.where(data.target_mask)[0].tolist():
            best = int(errors["best_mode"][node])
            graph_number = int(data.batch[node])
            local_node = node-int(data.ptr[graph_number])
            scale = raw[:, node, :, 2:]
            effective_scale = torch.ones_like(scale) if fixed_scale else scale
            gt_norm = torch.linalg.vector_norm(target[node], dim=-1)
            pred_norm = torch.linalg.vector_norm(raw[:, node, :, :2], dim=-1)
            actors.append({"node_in_batch": node, "node_in_graph":local_node,
                           "scene_token":data.scene_token[graph_number], "sample_token":data.sample_token[graph_number],
                           "instance_token":data.instance_tokens[graph_number][local_node],
                           "minADE_K": float(errors["minADE_K"][node]),
                           "minFDE_K": float(errors["minFDE_K"][node]), "best_FDE_mode": best,
                           "best_training_L2_mode": int(best_l2[node]), "mode_probabilities": output["mode_prob"][node].tolist(),
                           "gt_endpoint_displacement_m": float(gt_norm[-1]), "gt_max_displacement_m": float(gt_norm.max()),
                           "pred_max_displacement_m": float(pred_norm.max()), "best_mode_endpoint_displacement_m": float(pred_norm[best, -1]),
                           "scale_mean": float(effective_scale.mean()), "scale_max": float(effective_scale.max()),
                           "unused_scale_head_mean": float(scale.mean()) if fixed_scale else None,
                           "output_units": [{"future_step": step, "future_time_s": float(data.future_times.reshape(-1,12)[int(data.batch[node]), step-1]),
                                             "GT_local_xy_m": target[node, step-1].tolist(), "prediction_local_xy_m": raw[best, node, step-1, :2].tolist(),
                                             "prediction_to_GT_norm_ratio": float(pred_norm[best, step-1]/gt_norm[step-1]) if gt_norm[step-1]>.001 else None}
                                            for step in (1,3,6,12)]})
    metrics = aggregate(records)["full_horizon"]
    assert metrics["count"] == count
    return {**{k:v/count for k,v in loss_sums.items()}, "minADE": metrics["minADE_K"], "minFDE": metrics["minFDE_K"],
            "independent_minADE": metrics["independent_minADE_K"], "MR": metrics["MR_K"], "target_count": count,
            "pred_max_displacement_m": max(a["pred_max_displacement_m"] for a in actors),
            "GT_endpoint_displacement_mean_m": float(np.mean([a["gt_endpoint_displacement_m"] for a in actors])),
            "GT_max_displacement_m": max(a["gt_max_displacement_m"] for a in actors),
            "scale_mean": float(np.mean([a["scale_mean"] for a in actors])), "scale_max": max(a["scale_max"] for a in actors),
            "actors": actors}


def target_sanity(graph, target, model):
    data = graph.clone().to("cuda")
    node = target["node"]
    model.eval()
    with torch.no_grad():
        output = model(data)
        rotation = output["rotation"][node]
        y = data.y[node]
        rotated = y@rotation
        reconstructed = rotated@rotation.T+data.positions[node, 4]
        error = float((reconstructed-data.positions[node, 5:]).abs().max())
        assert error < 1e-4
        assert torch.allclose(y, data.positions[node, 5:]-data.positions[node, 4], atol=1e-5)
        # CPU probe avoids the small run-to-run GPU scatter reduction variation.
        probe = copy.deepcopy(model).cpu().eval()
        unchanged = probe(graph.clone())
        changed = graph.clone()
        changed.history_times *= 10
        changed.future_times *= .5
        changed_output = probe(changed)
        time_error = float((unchanged["raw_prediction"]-changed_output["raw_prediction"]).abs().max())
        assert time_error == 0
    nusc = load_nuscenes(load_config())
    chain = scene_samples(nusc, nusc.get("scene", target["scene_token"]))
    t0 = next(i for i,s in enumerate(chain) if s["token"] == target["sample_token"])
    index = {(a["sample_token"],a["instance_token"]):a for a in nusc.sample_annotation}
    annotations = [index[(s["token"],target["instance_token"])] for s in chain[t0-4:t0+13]]
    assert all(a["next"] == b["token"] and b["prev"] == a["token"] for a,b in zip(annotations, annotations[1:]))
    global_xy = np.array([a["translation"][:2] for a in annotations])
    source_ego = global_to_ego(global_xy, graph.origin.numpy(), float(graph.ego_yaw))
    source_error = float(np.max(np.abs(source_ego-graph.positions[node].numpy())))
    assert source_error < 1e-4
    audit = {"scene_token": target["scene_token"], "sample_token": target["sample_token"], "instance_token": target["instance_token"],
             "category": target["category_name"], "attribute_names": target["attribute_names"], "node": node,
             "context_agents": graph.num_nodes, "context_lane_segments": len(graph.lane_vectors), "supervised_targets": 1,
             "history_ego_coordinates_m": data.positions[node, :5].tolist(), "current_position_ego_m": data.positions[node,4].tolist(),
             "future_ego_coordinates_m": data.positions[node, 5:].tolist(), "y_future_minus_current_m": y.tolist(),
             "rotation_angle_rad": float(data.rotate_angles[node]), "rotation_matrix": rotation.tolist(), "rotated_y_m": rotated.tolist(),
             "inverse_rotation_plus_current_m": reconstructed.tolist(), "roundtrip_max_abs_error_m": error,
             "history_times_s": data.history_times.tolist(), "future_times_s": data.future_times.tolist(),
             "GT_endpoint_displacement_m": target["gt_endpoint_displacement_m"],
             "source_annotation_continuity": "all 17 source annotations consecutive in instance prev/next chain; checked in selection",
             "source_annotation_tokens": [a["token"] for a in annotations], "source_global_coordinates_m": global_xy.tolist(),
             "source_to_cached_ego_max_abs_error_m": source_error,
             "timestep_probe": {"history_time_multiplier": 10, "future_time_multiplier": .5, "max_output_difference": time_error,
                                "device": "CPU with same model weights, to avoid GPU scatter run-to-run roundoff",
                                "finding": "forward uses geometric displacements and learned step embeddings, not timestamp fields; no hidden timestep multiplier observed"},
             "units": "positions, x/y, lane geometry and decoder locations are meters; no rescaling applied", "TARGET_SANITY": "PASS"}
    write_json(PROJECT_ROOT/"outputs/reports/single_moving_target_audit.json", audit)
    print("TARGET_SANITY", error, "m", flush=True)


def gradient_norm(module):
    squared = sum(float(p.grad.detach().double().square().sum()) for p in module.parameters() if p.grad is not None)
    return squared**.5


def save_curve(rows, destination):
    fig, axes = plt.subplots(1,3,figsize=(15,4),layout="constrained")
    e = [r["epoch"] for r in rows]
    for key in ("loss", "regression_loss", "classification_loss"): axes[0].plot(e,[r[key] for r in rows],label=key)
    for key in ("minADE", "minFDE"): axes[1].plot(e,[r[key] for r in rows],label=key)
    for key in ("pred_max_displacement_m", "GT_max_displacement_m", "scale_mean", "scale_max"): axes[2].plot(e,[r[key] for r in rows],label=key)
    for ax in axes:
        ax.set_xlabel("Epoch"); ax.grid(alpha=.2); ax.legend(fontsize=7)
    axes[0].set_ylabel("Deterministic evaluation loss")
    axes[1].set_ylabel("Full-horizon error (m)")
    axes[2].set_ylabel("Displacement / scale (m)")
    fig.savefig(destination,dpi=130); plt.close(fig)


def run_experiment(name, targets, corpus, learning_rate=5e-4, num_modes=6, fixed_scale=False,
                   max_epochs=500, gate=(.5,1.0)):
    destination = AUDIT/name
    if destination.exists():
        raise RuntimeError(f"Experiment {name} already exists; do not overwrite")
    destination.mkdir()
    seed = 2022
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    graphs = selected_graphs(corpus, targets)
    original = yaml.safe_load((PROJECT_ROOT/"configs/stage2_tiny.yaml").read_text())
    config = {k:original[k] for k in MODEL_KEYS}
    config["num_modes"] = num_modes
    config.update(seed=seed, learning_rate=learning_rate, weight_decay=original["weight_decay"], scheduler="none",
                  max_epochs=max_epochs, evaluation_interval=10, batch_size=4, target_count=len(targets), scene_windows=len(graphs),
                  context="all nodes, actor edges and lane context unchanged; only target_mask limited",
                  loss="fixed unit-scale Laplace, location only, K=1 (diagnostic)" if fixed_scale else "unchanged official LaplaceNLL + detached soft-target mode CE",
                  diagnostic_only=True, gate_minADE_m=gate[0], gate_minFDE_m=gate[1],
                  early_stop="at evaluation epoch>=100 only if both strict thresholds pass; no loosened threshold")
    (destination/"config.yaml").write_text(yaml.safe_dump(config,sort_keys=False))
    write_json(destination/"targets.json", targets)
    model = HiVTNuScenesVehicle(**{k:config[k] for k in MODEL_KEYS}).cuda()
    initial_hash = hashlib.sha256(b"".join(p.detach().cpu().numpy().tobytes() for p in model.state_dict().values())).hexdigest()
    if name == "single_moving_actor_overfit": target_sanity(graphs[0], targets[0], model)
    optimizer = model.optimizer(learning_rate, original["weight_decay"])
    initial = evaluate(model, graphs, fixed_scale)
    write_json(destination/"initial_metrics.json", initial)
    print("INITIAL", name, json.dumps({k:v for k,v in initial.items() if k!="actors"}),flush=True)
    loader = DataLoader(graphs, batch_size=4, shuffle=True, num_workers=0)
    rows = [{"epoch":0, **{k:v for k,v in initial.items() if k!="actors"}}]
    units = [{"epoch":0, "actors":initial["actors"]}]
    gradients = []
    started = time.monotonic()
    passed = False
    for epoch in range(1,max_epochs+1):
        model.train()
        epoch_gradients = []
        for batch in loader:
            data = batch.to("cuda")
            optimizer.zero_grad(set_to_none=True)
            output = model(data)
            output["raw_prediction"].retain_grad()
            values = diagnostic_loss(model,output,data,fixed_scale)
            assert torch.isfinite(values["loss"])
            values["loss"].backward()
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            raw, target = output["raw_prediction"], torch.bmm(data.y,output["rotation"])
            best = torch.linalg.vector_norm(raw[..., :2]-target[None],dim=-1).sum(dim=-1).argmin(dim=0)
            nodes = torch.where(data.target_mask)[0]
            selected_raw_grad = raw.grad[best[nodes],nodes]
            unselected = ~data.target_mask
            assert not unselected.any() or torch.count_nonzero(raw.grad[:,unselected])==0, "Unselected context entered prediction loss"
            epoch_gradients.append({"loc_grad_norm":gradient_norm(model.decoder.loc),
                                    "scale_grad_norm":gradient_norm(model.decoder.scale),
                                    "pi_grad_norm":gradient_norm(model.decoder.pi),
                                    "chosen_location_output_grad_mean_abs":float(selected_raw_grad[..., :2].abs().mean()),
                                    "chosen_scale_output_grad_mean_abs":float(selected_raw_grad[..., 2:].abs().mean()),
                                    "target_count":len(nodes)})
            optimizer.step()
        grad = {"experiment":name, "epoch":epoch, "learning_rate":learning_rate, "gradient_timing":"train-mode backward before optimizer step",
                **{key: float(np.sqrt(np.mean([g[key]**2 for g in epoch_gradients]))) for key in epoch_gradients[0] if key!="target_count"}}
        gradients.append(grad)
        if epoch%10==0 or epoch in (1,50,100) or epoch==max_epochs:
            final = evaluate(model,graphs,fixed_scale)
            row = {"epoch":epoch, **{k:v for k,v in final.items() if k!="actors"}}
            rows.append(row); units.append({"epoch":epoch,"actors":final["actors"]})
            for key in ("pred_max_displacement_m", "scale_mean", "scale_max", "GT_max_displacement_m"): grad[key] = final[key]
            print("EVAL",name,json.dumps(row),flush=True)
            passed = gate[0] is not None and final["minADE"]<gate[0] and final["minFDE"]<gate[1]
            with open(destination/"training_curve.csv","w",newline="") as f:
                writer = csv.DictWriter(f,list(rows[0]),lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
            if passed and epoch>=100: break
    final = evaluate(model,graphs,fixed_scale)
    assert gradients[-1].get("pred_max_displacement_m") is not None
    gradient_fields = list(gradients[0])
    gradient_fields += [k for k in gradients[-1] if k not in gradient_fields]
    with open(destination/"gradient_audit.csv","w",newline="") as f:
        writer=csv.DictWriter(f,gradient_fields,lineterminator="\n"); writer.writeheader(); writer.writerows(gradients)
    write_json(destination/"output_units.json", units)
    torch.save({"state_dict":model.state_dict(),"config":config,"epoch":epoch},destination/"checkpoint.pt")
    result = {"experiment":name,"epochs":epoch,"elapsed_seconds":time.monotonic()-started,"initial_state_sha256":initial_hash,
              "configuration":config,"initial":initial,"final":final,
              "status":"PASS" if passed else "FAIL" if gate[0] is not None else "TREND_ONLY",
              "loss_supervision_verified":"all non-target raw prediction gradients exactly zero; context retained",
              "gradient_epoch_1":gradients[0],"gradient_final":gradients[-1]}
    write_json(destination/"metrics.json",result)
    save_curve(rows,destination/"training_curve.png")
    model.eval()
    for index,g in enumerate(graphs[:2]):
        with torch.no_grad():
            data=g.clone().cuda(); output=model(data); predictions=model.ego_predictions(output,data).cpu()
        node=int(torch.where(g.target_mask)[0][0])
        plot_case(g,predictions,output["mode_prob"].cpu(),node,destination/f"target_prediction_{index+1:02d}.png",f"{name}: {result['status']}, epoch {epoch}")
    print("FINAL",name,result["status"],final["minADE"],final["minFDE"],flush=True)
    return result


def verify_freeze():
    manifest = json.loads((AUDIT/"frozen_failure_manifest.json").read_text())
    errors = [p for p,info in manifest["files"].items() if hashlib.sha256((PROJECT_ROOT/p).read_bytes()).hexdigest()!=info["sha256"]]
    assert not errors, f"Frozen source changed: {errors}"
    write_json(AUDIT/"freeze_verification.json", {"original_commit":manifest["commit"],"verified_files":len(manifest["files"]),"unchanged":True})


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("single", "followups", "budget-check", "scaleups"),required=True)
    args=parser.parse_args()
    torch.set_num_threads(4)
    corpus=torch.load(PROJECT_ROOT/"outputs/stage2/mini/processed/graphs.pt",weights_only=False)["train"]
    selection=json.loads((AUDIT/"target_selection.json").read_text())
    if args.phase=="single":
        run_experiment("single_moving_actor_overfit",[selection["single"]],corpus)
    elif args.phase=="followups":
        primary=json.loads((AUDIT/"single_moving_actor_overfit/metrics.json").read_text())
        results=[primary]
        if primary["status"]!="PASS":
            primary=run_experiment("single_moving_actor_overfit_lr1e3",[selection["single"]],corpus,learning_rate=1e-3)
            results.append(primary)
        if primary["status"]!="PASS":
            k1=run_experiment("single_moving_actor_k1",[selection["single"]],corpus,learning_rate=1e-3,num_modes=1)
            results.append(k1)
            if k1["status"]!="PASS":
                results.append(run_experiment("single_moving_actor_fixed_scale",[selection["single"]],corpus,learning_rate=1e-3,num_modes=1,fixed_scale=True))
        if primary["status"]=="PASS":
            lr=primary["configuration"]["learning_rate"]
            for n,gate in [(4,(.75,1.5)),(8,(1.,2.)),(16,(None,None))]:
                results.append(run_experiment(f"moving_only_{n}",selection["moving"][:n],corpus,learning_rate=lr,gate=gate))
            # A separate balanced diagnostic is meaningful after moving-only 4/8 pass.
            if all(r["status"]=="PASS" for r in results if r["experiment"] in ("moving_only_4","moving_only_8")):
                results.append(run_experiment("balanced_8_moving_8_stationary",selection["moving"][:8]+selection["stationary"],corpus,learning_rate=lr,gate=(1.,2.)))
        write_json(AUDIT/"experiment_summary.json",[{k:r[k] for k in ("experiment","epochs","initial_state_sha256","configuration","initial","final","status","gradient_epoch_1","gradient_final")} for r in results])
    elif args.phase=="budget-check":
        results=json.loads((AUDIT/"experiment_summary.json").read_text())
        by_name={r["experiment"]:r for r in results}
        assert by_name["single_moving_actor_k1"]["status"]=="FAIL"
        assert by_name["single_moving_actor_fixed_scale"]["status"]=="FAIL"
        # Equal diagnostic budgets distinguish incomplete convergence from an
        # output/target defect; official K=6 experiments stay capped at 500.
        plan={"reason":"both K=1 diagnostics fail at 500, but fixed scale improves location substantially; check convergence with equal budgets",
              "scope":"K=1 diagnostic only; no K=6 extension, no full tiny/mini", "max_epochs":1000,
              "initialization":"fresh seed2022, same as respective 500-epoch runs", "learning_rate":.001,
              "only_change_vs_each_500_epoch_run":"maximum epoch budget", "matched_comparison":"compare official K=1 at the exact first fixed-scale passing epoch"}
        write_json(AUDIT/"budget_check_plan.json",plan)
        for name,fixed in [("single_moving_actor_k1_budget1000",False),("single_moving_actor_fixed_scale_budget1000",True)]:
            result=run_experiment(name,[selection["single"]],corpus,learning_rate=1e-3,num_modes=1,fixed_scale=fixed,max_epochs=1000)
            results.append({k:result[k] for k in ("experiment","epochs","initial_state_sha256","configuration","initial","final","status","gradient_epoch_1","gradient_final")})
        write_json(AUDIT/"experiment_summary.json",results)
    else:
        results=json.loads((AUDIT/"experiment_summary.json").read_text())
        passing=[r for r in results if r["configuration"]["target_count"]==1 and r["status"]=="PASS"]
        assert passing, "Single-moving must pass before scaling diagnostic target count"
        source=passing[-1]
        lr=source["configuration"]["learning_rate"]
        modes=source["configuration"]["num_modes"]
        fixed="fixed unit-scale" in source["configuration"]["loss"]
        budget=source["configuration"]["max_epochs"]
        write_json(AUDIT/"scaleup_plan.json",{"passing_single_experiment":source["experiment"],"loss":source["configuration"]["loss"],
                   "num_modes":modes,"learning_rate":lr,"max_epochs":budget,"diagnostic_only":True,
                   "gate_4":[.75,1.5],"gate_8":[1.,2.],"gate_16":"trend only; no mandatory threshold",
                   "balanced_condition":"4 and 8 moving actor diagnostic gates pass; 16 is trend-only",
                   "balanced_gate":[1.,2.],"context":"all original agents and lanes preserved in each selected window"})
        for n,gate in [(4,(.75,1.5)),(8,(1.,2.)),(16,(None,None))]:
            result=run_experiment(f"moving_only_{n}",selection["moving"][:n],corpus,learning_rate=lr,num_modes=modes,
                                  fixed_scale=fixed,max_epochs=budget,gate=gate)
            results.append({k:result[k] for k in ("experiment","epochs","initial_state_sha256","configuration","initial","final","status","gradient_epoch_1","gradient_final")})
        if all(r["status"]=="PASS" for r in results if r["experiment"] in ("moving_only_4","moving_only_8")):
            result=run_experiment("balanced_8_moving_8_stationary",selection["moving"][:8]+selection["stationary"],corpus,
                                  learning_rate=lr,num_modes=modes,fixed_scale=fixed,max_epochs=budget,gate=(1.,2.))
            results.append({k:result[k] for k in ("experiment","epochs","initial_state_sha256","configuration","initial","final","status","gradient_epoch_1","gradient_final")})
        write_json(AUDIT/"experiment_summary.json",results)
    verify_freeze()


if __name__ == "__main__":
    main()
