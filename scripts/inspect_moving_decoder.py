"""Post-training backward/parameter inspection; no optimizer steps or new training."""
import csv
import copy
import hashlib
import json

import torch

from models.hivt_nuscenes import HiVTNuScenesVehicle
from preprocessing.common import PROJECT_ROOT, write_json
from scripts.audit_vehicle_motion import AUDIT
from scripts.moving_overfit_audit import diagnostic_loss, evaluate, gradient_norm, selected_graphs, verify_freeze
from scripts.train_hivt_stage2 import MODEL_KEYS


def inspect(checkpoint, corpus, targets):
    config=checkpoint["config"]
    torch.manual_seed(config["seed"])
    model=HiVTNuScenesVehicle(**{k:config[k] for k in MODEL_KEYS}).eval()
    initial_state={k:v.clone() for k,v in model.state_dict().items()}
    model.load_state_dict(checkpoint["state_dict"])
    graph=selected_graphs(corpus,targets)[0]
    assert int(graph.target_mask.sum())==1
    node=int(torch.where(graph.target_mask)[0][0])
    fixed="fixed unit-scale" in config["loss"]
    captured={}
    handle=model.decoder.loc[-1].register_forward_pre_hook(lambda module,inputs:captured.update(activation=inputs[0].detach()))
    output=model(graph); handle.remove()
    raw=output["raw_prediction"]; raw.retain_grad()
    local_target=torch.bmm(graph.y,output["rotation"])
    best=int(torch.linalg.vector_norm(raw[:,node,:,:2]-local_target[node],dim=-1).sum(dim=-1).argmin())
    values=diagnostic_loss(model,output,graph,fixed)
    values["loss"].backward()
    chosen=raw[best,node].detach()
    grad=raw.grad[best,node]
    mu=chosen[:,:2]; scale=chosen[:,2:]
    residual=(mu-local_target[node]).abs()
    effective=torch.ones_like(scale) if fixed else scale
    theoretical_mu=(mu-local_target[node]).sign()/(24*effective)
    derivative_error=float((theoretical_mu-grad[:,:2]).abs().max())
    assert derivative_error<1e-5
    activation=captured["activation"][best,node]
    last=model.decoder.loc[-1]
    linear=(activation@last.weight.T+last.bias).reshape(12,2)
    linear_error=float((linear-mu).abs().max())
    assert linear_error<1e-5
    optimizer=model.optimizer(config["learning_rate"],config["weight_decay"])
    params=[id(p) for group in optimizer.param_groups for p in group["params"]]
    assert len(params)==len(set(params))==len(list(model.parameters()))
    assert set(params)=={id(p) for p in model.parameters()}
    ego=model.ego_predictions(output,graph)[node,best]
    norm_error=float((torch.linalg.vector_norm(ego-graph.positions[node,4],dim=-1)-torch.linalg.vector_norm(mu,dim=-1)).abs().max())
    assert norm_error<1e-4
    delta={}
    for name in ("decoder.loc.3.weight","decoder.loc.3.bias","decoder.scale.3.weight","decoder.scale.3.bias","decoder.pi.6.weight"):
        value=model.state_dict()[name]; before=initial_state[name]
        delta[name]={"initial_norm":float(before.norm()),"final_norm":float(value.norm()),"change_norm":float((value-before).norm())}
    # Algebraic range/unit probe in a disposable model copy. Uses GT deliberately,
    # so it is never an overfit result or a forecasting metric.
    oracle=copy.deepcopy(model).eval()
    with torch.no_grad():
        oracle.decoder.loc[-1].weight.zero_()
        oracle.decoder.loc[-1].bias.copy_(local_target[node].reshape(-1))
        oracle_output=oracle(graph)
        oracle_ego=oracle.ego_predictions(oracle_output,graph)[node]
        oracle_error=float((oracle_ego-graph.positions[node,None,5:]).abs().max())
    assert oracle_error<1e-4
    return {"epoch":checkpoint["epoch"],"device":"CPU eval-mode backward; no optimizer.step",
            "best_sum_L2_mode":best,"losses":{k:float(v) for k,v in values.items()},
            "head_gradient_norms":{name:gradient_norm(getattr(model.decoder,name)) for name in ("loc","scale","pi")},
            "encoder_gradient_norms":{"local":gradient_norm(model.local_encoder),"global":gradient_norm(model.global_interactor)},
            "location_output_gradient_mean_abs_xy":grad[:,:2].abs().mean(dim=0).tolist(),
            "scale_output_gradient_mean_abs_xy":grad[:,2:].abs().mean(dim=0).tolist(),
            "location_last_weight_gradient_norm_xy":[float(last.weight.grad[i::2].norm()) for i in (0,1)],
            "residual_mean_abs_xy_m":residual.mean(dim=0).tolist(),"chosen_scale_mean_xy_m":effective.mean(dim=0).tolist(),
            "NLL_mean_xy":(torch.log(2*effective)+residual/effective).mean(dim=0).tolist(),
            "chosen_scale_max_xy_m":effective.max(dim=0).values.tolist(),
            "chosen_scale_min_xy_m":effective.min(dim=0).values.tolist(),
            "chosen_raw_scale_mean_xy_m":scale.mean(dim=0).tolist(),
            "chosen_prediction_local_xy_m":mu.tolist(),"GT_local_xy_m":local_target[node].tolist(),
            "stepwise_location_scale_gradient": [{"future_step":i+1,"residual_abs_xy_m":residual[i].tolist(),
                                                  "effective_scale_xy_m":effective[i].tolist(),
                                                  "location_output_gradient_abs_xy":grad[i,:2].abs().tolist()}
                                                 for i in range(12)],
            "decoder_final_linear_reconstruction_max_error_m":linear_error,
            "ego_vs_local_displacement_norm_max_difference_m":norm_error,
            "analytic_dNLL_dmu_vs_autograd_max_difference":derivative_error,
            "analytic_derivative":"sign(mu-y)/(valid_steps*2*b); fixed diagnostic b=1",
            "optimizer_includes_every_parameter_once":True,"all_gradients_finite":all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None),
            "final_loc_activation_norm":float(activation.norm()),"initial_to_final_parameter_changes":delta,
            "algebraic_output_range_probe":{"target_supplied_to_disposable_last_linear_bias":True,
                                                  "GT_reconstruction_max_error_m":oracle_error,
                                                  "not_a_training_result":True,
                                                  "finding":"unchanged decoder can represent the 49.53m target; no tanh bound, output scaling or inverse-frame range restriction"}}


def main():
    torch.set_num_threads(4)
    corpus=torch.load(PROJECT_ROOT/"outputs/stage2/mini/processed/graphs.pt",weights_only=False)["train"]
    summary=json.loads((AUDIT/"experiment_summary.json").read_text())
    selections=json.loads((AUDIT/"target_selection.json").read_text())
    result={"experiments":{},"training_started":False,"optimizer_steps":0}
    rows=[]
    for r in summary:
        d=AUDIT/r["experiment"]
        rows.extend(csv.DictReader((d/"gradient_audit.csv").open()))
        if r["configuration"]["target_count"]!=1: continue
        checkpoint=torch.load(d/"checkpoint.pt",weights_only=False,map_location="cpu")
        result["experiments"][r["experiment"]]=inspect(checkpoint,corpus,[selections["single"]])
    # Explicit per-actor identities and subgroup metrics for all final checkpoints.
    final_actor_metrics={}
    for r in summary:
        checkpoint=torch.load(AUDIT/r["experiment"]/"checkpoint.pt",weights_only=False,map_location="cpu")
        model=HiVTNuScenesVehicle(**{k:r["configuration"][k] for k in MODEL_KEYS}).cuda().eval()
        model.load_state_dict(checkpoint["state_dict"])
        targets=json.loads((AUDIT/r["experiment"]/"targets.json").read_text())
        graphs=selected_graphs(corpus,targets)
        measured=evaluate(model,graphs,"fixed unit-scale" in r["configuration"]["loss"])
        assert abs(measured["minADE"]-r["final"]["minADE"])<1e-3
        assert abs(measured["minFDE"]-r["final"]["minFDE"])<1e-3
        identities={(t["sample_token"],t["instance_token"]):t for t in targets}
        for a in measured["actors"]:
            a["attribute_state"]=identities[(a["sample_token"],a["instance_token"])]["attribute_state"]
        measured["by_real_attribute"]={}
        for state in sorted({a["attribute_state"] for a in measured["actors"]}):
            actors=[a for a in measured["actors"] if a["attribute_state"]==state]
            measured["by_real_attribute"][state]={"count":len(actors),"minADE":sum(a["minADE_K"] for a in actors)/len(actors),
                                                   "minFDE":sum(a["minFDE_K"] for a in actors)/len(actors)}
        final_actor_metrics[r["experiment"]]=measured
    write_json(PROJECT_ROOT/"outputs/reports/moving_final_actor_metrics.json",final_actor_metrics)
    with open(PROJECT_ROOT/"outputs/reports/moving_gradient_audit.csv","w",newline="") as f:
        fields=list(rows[0])
        writer=csv.DictWriter(f,fields,lineterminator="\n");writer.writeheader();writer.writerows(rows)
    source=PROJECT_ROOT/"baselines/hivt_official"
    manifest=json.loads((source/"SOURCE.json").read_text())
    assert all(hashlib.sha256((source/path).read_bytes()).hexdigest()==sha for path,sha in manifest["sha256"].items())
    result["official_source_hashes_verified"]=len(manifest["sha256"])
    write_json(PROJECT_ROOT/"outputs/reports/moving_decoder_audit.json",result)
    verify_freeze()
    print(json.dumps(result,indent=2),flush=True)


if __name__ == "__main__":
    main()
