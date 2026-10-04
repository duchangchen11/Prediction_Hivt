"""Stage 2B gated K=6 recovery experiments; existing experiments are read-only."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import shutil
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.loader import DataLoader
import yaml

from metrics.hivt_forecasting import multimodal_errors
from models.hivt_loss_recovery import HiVTLossRecovery
from preprocessing.common import PROJECT_ROOT, load_config, load_nuscenes, write_json
from scripts.moving_overfit_audit import gradient_norm, selected_graphs
from scripts.plot_hivt_predictions import plot_case
from scripts.train_hivt_stage2 import MODEL_KEYS

ROOT = PROJECT_ROOT / "outputs/stage2/k6_loss_recovery"
GROUPS = ("overall", "vehicle.moving", "vehicle.stopped", "vehicle.parked", "unknown")


def read_config():
    return yaml.safe_load((PROJECT_ROOT/"configs/stage2b_recovery.yaml").read_text())


def seed_all(seed=2022):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)


def state_hash(model):
    return hashlib.sha256(b"".join(p.detach().cpu().numpy().tobytes() for p in model.state_dict().values())).hexdigest()


def new_model(config):
    seed_all(config["seed"])
    return HiVTLossRecovery(**{k:config[k] for k in MODEL_KEYS}).cuda()


def attribute_index():
    nusc=load_nuscenes(load_config())
    names={a["token"]:a["name"] for a in nusc.attribute}
    print("ACTUAL_ATTRIBUTE_NAMES",json.dumps(sorted(names.values())),flush=True)
    return {(a["sample_token"],a["instance_token"]):[names[t] for t in a["attribute_tokens"]] for a in nusc.sample_annotation}


def summarize_records(records):
    result={}
    for horizon in ("full_horizon", "partial_future"):
        result[horizon]={}
        for group in GROUPS:
            rows=[r for r in records if r["horizon"]==horizon and (group=="overall" or r["attribute_group"]==group)]
            result[horizon][group]={"count":len(rows),**{k:float(np.mean([r[k] for r in rows])) if rows else None
                                                        for k in ("minADE", "minFDE", "MR", "independent_minADE")}}
    return result


@torch.no_grad()
def evaluate(model, graphs, attributes, phase="original_nll", b_max=None, batch_size=4):
    model.eval()
    records=[]; regression_sum=classification_sum=location_sum=nll_sum=0.; steps=targets=0
    scale_sum=raw_scale_sum=preactivation_sum=0.; scale_count=0
    scale_max=raw_scale_max=preactivation_max=-float("inf"); preactivation_min=float("inf")
    captured={}
    handle=model.decoder.scale[-1].register_forward_hook(lambda module,inputs,value:captured.update(value=value.detach().clone()))
    try:
        for batch in DataLoader(graphs,batch_size=batch_size,shuffle=False):
            data=batch.to(next(model.parameters()).device); output=model(data)
            values=model.recovery_loss(output,data,phase,b_max)
            assert all(torch.isfinite(v) for v in values.values())
            mask=data.future_mask & data.target_mask[:,None]
            s=int(mask.sum()); n=int(data.target_mask.sum()); steps+=s; targets+=n
            regression_sum+=float(values["regression_loss"])*s
            classification_sum+=float(values["classification_loss"])*n
            location_sum+=float(values["location_error"])*s; nll_sum+=float(values["NLL"])*s
            raw=output["raw_prediction"]
            active=mask[None,:,:,None].expand_as(raw[...,2:])
            raw_b=raw[...,2:][active]
            effective=torch.ones_like(raw_b) if phase=="fixed_scale" else raw_b.clamp(min=1e-3,max=b_max) if phase=="bounded_scale" else raw_b
            preactivation=captured["value"].reshape(*raw.shape[:-1],2)[active]
            scale_sum+=float(effective.sum()); raw_scale_sum+=float(raw_b.sum()); preactivation_sum+=float(preactivation.sum())
            scale_count+=effective.numel()
            scale_max=max(scale_max,float(effective.max())); raw_scale_max=max(raw_scale_max,float(raw_b.max()))
            preactivation_max=max(preactivation_max,float(preactivation.max())); preactivation_min=min(preactivation_min,float(preactivation.min()))
            prediction=model.ego_predictions(output,data)
            errors=multimodal_errors(prediction,data.positions[:,5:],data.future_mask,data.target_mask)
            rotated=torch.bmm(data.y,output["rotation"])
            best_training=(torch.linalg.vector_norm(raw[...,:2]-rotated[None],dim=-1)*mask[None]).sum(-1).argmin(0)
            for horizon in ("full_horizon", "partial_future"):
                for node in torch.where(errors[horizon])[0].tolist():
                    graph=int(data.batch[node]); local=node-int(data.ptr[graph])
                    sample=data.sample_token[graph]; instance=data.instance_tokens[graph][local]
                    actual=attributes[(sample,instance)]
                    group=next((a for a in actual if a in GROUPS),"unknown")
                    last=int(errors["last_valid_index"][node])
                    gt_displacement=float(torch.linalg.vector_norm(data.y[node,last]))
                    records.append({"scene_token":data.scene_token[graph],"sample_token":sample,"instance_token":instance,
                                    "node_in_graph":local,"horizon":horizon,"attribute_names":actual,"attribute_group":group,
                                    "valid_future_steps":int(errors["valid_steps"][node]),"GT_endpoint_displacement_m":gt_displacement,
                                    "minADE":float(errors["minADE_K"][node]),"minFDE":float(errors["minFDE_K"][node]),
                                    "MR":float(errors["MR_K"][node]),"independent_minADE":float(errors["independent_minADE_K"][node]),
                                    "best_training_mode":int(best_training[node]),"best_FDE_mode":int(errors["best_mode"][node]),
                                    "mode_probabilities":output["mode_prob"][node].tolist()})
    finally:
        handle.remove()
    assert steps and targets and scale_count
    metrics=summarize_records(records)
    full=metrics["full_horizon"]["overall"]
    return {"metrics":metrics,"actors":records,"ADE":full["minADE"],"FDE":full["minFDE"],"MR":full["MR"],
            "loss":regression_sum/steps+classification_sum/targets,"regression_loss":regression_sum/steps,
            "classification_loss":classification_sum/targets,"location_error":location_sum/steps,"NLL":nll_sum/steps,
            "scale_mean":scale_sum/scale_count,"scale_max":scale_max,
            "raw_processed_scale_mean":raw_scale_sum/scale_count,"raw_processed_scale_max":raw_scale_max,
            "raw_scale_head_mean":preactivation_sum/scale_count,"raw_scale_head_min":preactivation_min,"raw_scale_head_max":preactivation_max}


def scalar_row(epoch,evaluation,lr):
    result={"epoch":epoch,"learning_rate":lr,**{k:evaluation[k] for k in ("ADE","FDE","MR","loss","regression_loss","classification_loss","location_error","NLL","scale_mean","scale_max","raw_processed_scale_mean","raw_processed_scale_max","raw_scale_head_mean","raw_scale_head_min","raw_scale_head_max")}}
    for group in GROUPS[1:]:
        values=evaluation["metrics"]["full_horizon"][group]
        result[group+"_ADE"]=values["minADE"];result[group+"_FDE"]=values["minFDE"]
    return result


def write_csv(path,rows):
    with open(path,"w",newline="") as f:
        writer=csv.DictWriter(f,list(rows[0]),lineterminator="\n");writer.writeheader();writer.writerows(rows)


def plot_curve(rows,path):
    fig,axes=plt.subplots(1,3,figsize=(15,4),layout="constrained")
    epochs=[r["epoch"] for r in rows]
    for key in ("loss","regression_loss","classification_loss"):axes[0].plot(epochs,[r[key] for r in rows],label=key)
    for key in ("ADE","FDE","vehicle.moving_ADE","vehicle.moving_FDE"):axes[1].plot(epochs,[r[key] for r in rows],label=key)
    for key in ("scale_mean","scale_max","raw_processed_scale_mean"):axes[2].plot(epochs,[r[key] for r in rows],label=key)
    for ax in axes:ax.set_xlabel("Epoch");ax.grid(alpha=.2);ax.legend(fontsize=7)
    axes[1].set_ylabel("Full-horizon error (m)");axes[2].set_ylabel("Scale (m)")
    fig.savefig(path,dpi=140);plt.close(fig)


def save_checkpoint(path,model,optimizer,config,epoch):
    torch.save({"state_dict":model.state_dict(),"optimizer_state_dict":optimizer.state_dict(),"config":config,"epoch":epoch,
                "torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all()},path)


def train_phase(name,graphs,attributes,settings,model=None,optimizer=None,stop_gate=None,final_gate=None):
    config=read_config()
    directory=ROOT/name
    if directory.exists():raise RuntimeError(f"Refusing to overwrite {directory}")
    directory.mkdir(parents=True)
    phase=settings["phase"];lr=settings["lr"];b_max=settings.get("b_max")
    fresh=model is None
    if fresh:model=new_model(config)
    if optimizer is None:optimizer=model.optimizer(lr,config["weight_decay"])
    for group in optimizer.param_groups:group["lr"]=lr
    config={**config,**settings,"experiment":name,"fresh_initialization":fresh,"scene_windows":len(graphs),
            "prediction_targets":sum(int(g.target_mask.sum()) for g in graphs),"optimizer_state_preserved":not fresh,
            "metric":"best-FDE mode ADE; full and partial horizons reported separately"}
    (directory/"config.yaml").write_text(yaml.safe_dump(config,sort_keys=False))
    write_json(directory/"windows.json",[{"scene_token":g.scene_token,"sample_token":g.sample_token,"targets":int(g.target_mask.sum())} for g in graphs])
    initial_hash=state_hash(model);initial=evaluate(model,graphs,attributes,phase,b_max)
    write_json(directory/"initial_metrics.json",initial)
    print("INITIAL",name,json.dumps(scalar_row(0,initial,lr)),flush=True)
    rows=[scalar_row(0,initial,lr)];gradients=[];series=[{"epoch":0,"actors":initial["actors"]}]
    loader=DataLoader(graphs,batch_size=config["batch_size"],shuffle=True,num_workers=0)
    started=time.monotonic();passed=False
    for epoch in range(1,settings["max_epochs"]+1):
        model.train();batch_grad=[]
        for batch in loader:
            data=batch.to("cuda");optimizer.zero_grad(set_to_none=True)
            output=model(data);output["raw_prediction"].retain_grad()
            values=model.recovery_loss(output,data,phase,b_max)
            assert torch.isfinite(values["loss"])
            values["loss"].backward()
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            assert not (~data.target_mask).any() or torch.count_nonzero(output["raw_prediction"].grad[:,~data.target_mask])==0
            if phase=="fixed_scale":assert torch.count_nonzero(output["raw_prediction"].grad[...,2:])==0
            batch_grad.append({k:gradient_norm(getattr(model.decoder,k)) for k in ("loc","scale","pi")})
            optimizer.step()
        gradients.append({"epoch":epoch,"lr":lr,**{k+"_grad_norm":float(np.sqrt(np.mean([g[k]**2 for g in batch_grad]))) for k in batch_grad[0]}})
        if epoch%config["evaluation_interval"]==0 or epoch in (1,settings["max_epochs"]):
            final=evaluate(model,graphs,attributes,phase,b_max)
            rows.append(scalar_row(epoch,final,lr));series.append({"epoch":epoch,"actors":final["actors"]})
            write_csv(directory/"training_curve.csv",rows)
            print("EVAL",name,json.dumps(rows[-1]),flush=True)
            if stop_gate is not None and stop_gate(final):passed=True;break
    final=evaluate(model,graphs,attributes,phase,b_max)
    if final_gate is not None:passed=final_gate(final)
    elif stop_gate is not None:passed=stop_gate(final)
    save_checkpoint(directory/"checkpoint.pt",model,optimizer,config,epoch)
    write_csv(directory/"gradient_logs.csv",gradients)
    write_json(directory/"actor_series.json",series)
    result={"experiment":name,"epochs":epoch,"optimizer_steps":epoch*len(loader),"phase":phase,"status":"PASS" if passed else "FAIL",
            "initial_state_sha256":initial_hash,"final_state_sha256":state_hash(model),"elapsed_seconds":time.monotonic()-started,
            "config":config,"initial":initial,"final":final,"gradient_first":gradients[0],"gradient_last":gradients[-1]}
    write_json(directory/"metrics.json",result);plot_curve(rows,directory/"training_curve.png")
    if len(graphs)==1:
        g=graphs[0];node=int(torch.where(g.target_mask)[0][0])
        with torch.no_grad():
            model.eval();data=g.clone().cuda();output=model(data)
        plot_case(g,model.ego_predictions(output,data).detach().cpu(),output["mode_prob"].cpu(),node,directory/"predictions.png",f"{name}: {result['status']}, epoch {epoch}")
    print("FINAL",name,result["status"],final["ADE"],final["FDE"],flush=True)
    verify_previous()
    return result,model,optimizer


def single_graph():
    corpus=torch.load(PROJECT_ROOT/"outputs/stage2/mini/processed/graphs.pt",weights_only=False)["train"]
    target=json.loads((PROJECT_ROOT/"outputs/stage2/moving_overfit_audit/target_selection.json").read_text())["single"]
    assert target["scene_token"]=="bebf5f5b2a674631ab5c88fd1aa9e87a"
    assert target["sample_token"]=="baaa60749cd04db7952fd8f4ef8ac837"
    assert target["instance_token"]=="c56ebf9c16dc44b8b9cd34fb79f40bc6"
    graph=selected_graphs(corpus,[target])[0]
    assert graph.num_nodes==51 and len(graph.lane_vectors)==666 and int(graph.target_mask.sum())==1
    write_json(ROOT/"single_target.json",target)
    return [graph]


def simple_gate(settings):
    return lambda e:e["ADE"]<settings["max_ADE_m"] and e["FDE"]<settings["max_FDE_m"]


@torch.no_grad()
def mode_audit(model,graph,experiment):
    model.eval();data=graph.clone().cuda();output=model(data)
    node=int(torch.where(data.target_mask)[0][0]);raw=output["raw_prediction"][:,node,:,:2]
    target=torch.bmm(data.y,output["rotation"])[node]
    distance=torch.linalg.vector_norm(raw-target,dim=-1)
    ego=model.ego_predictions(output,data)[node]
    pairwise=torch.linalg.vector_norm(raw[:,None]-raw[None],dim=-1)
    tolerance=read_config()["mode_collapse_max_pairwise_distance_m"]
    result={"experiment":experiment,"frame":"agent-local displacement and t0-ego positions, meters",
            "best_training_mode":int(distance.sum(-1).argmin()),"best_FDE_mode":int(distance[:,-1].argmin()),
            "best_modes_agree":bool(distance.sum(-1).argmin()==distance[:,-1].argmin()),
            "mode_probabilities":output["mode_prob"][node].tolist(),"max_pairwise_trajectory_distance_m":float(pairwise.max()),
            "collapse_tolerance_m":tolerance,"MODE_COLLAPSE":"YES" if pairwise.max()<=tolerance else "NO",
            "modes":[{"mode":k,"endpoint_local_xy_m":raw[k,-1].tolist(),"endpoint_ego_xy_m":ego[k,-1].tolist(),
                      "endpoint_displacement_m":float(torch.linalg.vector_norm(raw[k,-1])),"trajectory_local_m":raw[k].tolist(),
                      "trajectory_ego_m":ego[k].tolist(),"probability":float(output["mode_prob"][node,k]),
                      "regression_sum_L2_m":float(distance[k].sum()),"final_FDE_m":float(distance[k,-1])} for k in range(6)]}
    write_json(PROJECT_ROOT/"outputs/reports/k6_mode_audit.json",result)


def verify_previous():
    manifest=json.loads((ROOT/"frozen_previous_manifest.json").read_text())
    assert all(hashlib.sha256((PROJECT_ROOT/p).read_bytes()).hexdigest()==i["sha256"] for p,i in manifest["files"].items())
    write_json(ROOT/"freeze_verification.json",{"original_commit":manifest["commit"],"unchanged_files":len(manifest["files"]),"status":"PASS"})


def restore_checkpoint(path,config,lr):
    saved=torch.load(path,weights_only=False,map_location="cpu")
    model=new_model(config);model.load_state_dict(saved["state_dict"])
    optimizer=model.optimizer(lr,config["weight_decay"])
    optimizer.load_state_dict(saved["optimizer_state_dict"])
    torch.set_rng_state(saved["torch_rng"]);torch.cuda.set_rng_state_all(saved["cuda_rng"])
    for group in optimizer.param_groups:group["lr"]=lr
    return model,optimizer


def select_protocol(protocol,reason,b_max=None):
    model_name="HiVT-NuScenes-Vehicle-Baseline"
    if protocol==2:model_name+=" with bounded uncertainty scale"
    if protocol==3:model_name="HiVT-NuScenes adapted regression loss"
    selected={"protocol":protocol,"model_name":model_name,"reason":reason,"b_max":b_max,
              "num_modes":6,"historical_steps":5,"future_steps":12,"architecture_unchanged":True,
              "phase":"original_nll" if protocol==1 else "bounded_scale" if protocol==2 else "fixed_scale"}
    write_json(ROOT/"selected_protocol.json",selected)
    return selected


def run_B(graphs,attributes,config):
    a=json.loads((ROOT/"A_fixed_scale/metrics.json").read_text())
    assert a["status"]=="PASS", "A FAIL prohibits B"
    warm,_,_=train_phase("B_warmup",graphs,attributes,config["experiment_B_warmup"],stop_gate=simple_gate(config["experiment_B_warmup"]))
    assert warm["initial_state_sha256"]==a["initial_state_sha256"], "B must start from fresh same seed, not A final"
    checkpoint=ROOT/"warmup_checkpoint.pt"
    shutil.copy2(ROOT/"B_warmup/checkpoint.pt",checkpoint)
    if warm["status"]!="PASS":
        result={"status":"FAIL","reason":"Warm-up did not reach ADE<1/FDE<2 by 700; original NLL not started"}
        write_json(ROOT/"B_result.json",result);return result
    settings=config["experiment_B_nll"]
    model,optimizer=restore_checkpoint(checkpoint,config,settings["lr"])
    assert state_hash(model)==warm["final_state_sha256"]
    gate=lambda e:(simple_gate(settings)(e) and e["ADE"]<=warm["final"]["ADE"]+settings["max_ADE_degradation_m"]
                   and e["FDE"]<=warm["final"]["FDE"]+settings["max_FDE_degradation_m"])
    restored,_,_=train_phase("B_original_nll",graphs,attributes,settings,model=model,optimizer=optimizer,final_gate=gate)
    assert restored["initial_state_sha256"]==warm["final_state_sha256"]
    result={"status":restored["status"],"warmup_epochs":warm["epochs"],"nll_epochs":restored["epochs"],
            "warmup_ADE":warm["final"]["ADE"],"warmup_FDE":warm["final"]["FDE"],
            "restored_ADE":restored["final"]["ADE"],"restored_FDE":restored["final"]["FDE"],
            "ADE_change":restored["final"]["ADE"]-warm["final"]["ADE"],
            "FDE_change":restored["final"]["FDE"]-warm["final"]["FDE"]}
    write_json(ROOT/"B_result.json",result)
    if restored["status"]=="PASS":
        select_protocol(1,"K=6 fresh fixed-scale warm-up→原 Laplace NLL 严格通过，保留原 probabilistic loss；仅 optimization warm-up 为 nuScenes 约6s任务适配。")
    return result


def run_C(graphs,attributes,config):
    assert json.loads((ROOT/"A_fixed_scale/metrics.json").read_text())["status"]=="PASS"
    assert json.loads((ROOT/"B_result.json").read_text())["status"]=="FAIL", "C requires B FAIL"
    results=[]
    for b_max in config["experiment_C"]["b_max_values"]:
        settings={**config["experiment_C"],"b_max":b_max}
        result,_,_=train_phase(f"C_bmax{int(b_max)}",graphs,attributes,settings,final_gate=simple_gate(settings))
        results.append(result)
    passing=[r for r in results if r["status"]=="PASS"]
    if passing:
        selected=max(passing,key=lambda r:r["config"]["b_max"])
        select_protocol(2,"B FAIL；在仅2/4/8三组预注册限制中选择严格通过的最大 b_max，保持最弱 uncertainty 限制。",selected["config"]["b_max"])
    else:
        # Explicit user-authorized third-priority fallback; never call it the original loss.
        select_protocol(3,"Warm-up→原 NLL 与三组 bounded scale 均未通过；A 的 permanent fixed-scale K=6 严格通过，按用户 Protocol 3 使用 adapted regression loss。")
    write_json(ROOT/"C_result.json",{str(int(r["config"]["b_max"])):r["status"] for r in results})


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--experiment",choices=("A","B","C"),required=True);args=parser.parse_args()
    torch.set_num_threads(4)
    config=read_config();graphs=single_graph();attributes=attribute_index()
    if args.experiment=="A":
        result,model,_=train_phase("A_fixed_scale",graphs,attributes,config["experiment_A"],stop_gate=simple_gate(config["experiment_A"]))
        if result["status"]=="PASS":mode_audit(model,graphs[0],result["experiment"])
        else:print("HARD_STOP: K6_FIXED_SCALE FAIL; no further experiments authorized by gate",flush=True)
    elif args.experiment=="B":run_B(graphs,attributes,config)
    else:run_C(graphs,attributes,config)


if __name__=="__main__":main()
