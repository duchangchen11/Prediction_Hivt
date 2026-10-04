"""Original-window full tiny recovery and gated train/val mini baseline."""
import argparse
import json

import torch
import yaml

from metrics.hivt_forecasting import multimodal_errors
from models.constant_velocity import constant_velocity
from models.hivt_loss_recovery import HiVTLossRecovery
from preprocessing.common import PROJECT_ROOT, write_json
from scripts.k6_loss_recovery import (ROOT, GROUPS, attribute_index, evaluate, read_config,
                                     summarize_records, verify_previous, restore_checkpoint, train_phase)
from scripts.train_hivt_stage2 import MODEL_KEYS
from scripts.plot_hivt_predictions import plot_case


def baseline_config():
    return yaml.safe_load((PROJECT_ROOT/"configs/stage2b_baseline.yaml").read_text())


def original_tiny_graphs():
    train=torch.load(PROJECT_ROOT/"outputs/stage2/mini/processed/graphs.pt",weights_only=False,map_location="cpu")["train"]
    index={(g.scene_token,g.sample_token):g for g in train}
    windows=json.loads((PROJECT_ROOT/"outputs/stage2/tiny_overfit/training_windows.json").read_text())
    assert len(windows)==16
    graphs=[index[(w["scene_token"],w["sample_token"])] for w in windows]
    assert sum(int(g.target_mask.sum()) for g in graphs)==382
    assert sum(int((g.target_mask & g.future_mask.all(-1)).sum()) for g in graphs)==236
    assert all(g.x.shape[1]==5 and g.y.shape[1]==12 for g in graphs)
    return graphs


@torch.no_grad()
def evaluate_cv(graphs,attributes):
    records=[]
    for g in graphs:
        prediction,valid,_=constant_velocity(g.positions[:,:5],g.history_mask,g.history_times,g.future_times)
        assert torch.equal(g.target_mask & valid,g.target_mask)
        errors=multimodal_errors(prediction[:,None],g.positions[:,5:],g.future_mask,g.target_mask)
        for horizon in ("full_horizon","partial_future"):
            for node in torch.where(errors[horizon])[0].tolist():
                actual=attributes[(g.sample_token,g.instance_tokens[node])]
                group=next((a for a in actual if a in GROUPS),"unknown")
                records.append({"scene_token":g.scene_token,"sample_token":g.sample_token,"instance_token":g.instance_tokens[node],
                                "horizon":horizon,"attribute_names":actual,"attribute_group":group,
                                "minADE":float(errors["minADE_K"][node]),"minFDE":float(errors["minFDE_K"][node]),
                                "MR":float(errors["MR_K"][node]),"independent_minADE":float(errors["independent_minADE_K"][node])})
    return {"metrics":summarize_records(records),"actors":records,"K":1,"timestamps":"actual per-window timestamps, past-only last-two observed positions"}


def reference():
    graphs=original_tiny_graphs();attributes=attribute_index()
    config=read_config()
    model=HiVTLossRecovery(**{k:config[k] for k in MODEL_KEYS}).eval()
    checkpoint=torch.load(PROJECT_ROOT/"outputs/stage2/tiny_overfit/checkpoint.pt",weights_only=False,map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    original=evaluate(model,graphs,attributes)
    saved=json.loads((PROJECT_ROOT/"outputs/stage2/tiny_overfit/metrics.json").read_text())["final"]["metrics"]["full_horizon"]
    actual=original["metrics"]["full_horizon"]["overall"]
    assert actual["count"]==saved["count"]==236
    assert abs(actual["minADE"]-saved["minADE_K"])<1e-3
    assert abs(actual["minFDE"]-saved["minFDE_K"])<1e-3
    comparison={"CV":evaluate_cv(graphs,attributes),"Original HiVT free-scale failed run":original}
    write_json(ROOT/"tiny_reference_comparison.json",comparison)
    print("TINY_REFERENCE",json.dumps({k:v["metrics"]["full_horizon"] for k,v in comparison.items()}),flush=True)
    verify_previous()


def tiny_gate(evaluation,reference_result,gate):
    current=evaluation["metrics"]["full_horizon"]
    old=reference_result["Original HiVT free-scale failed run"]["metrics"]["full_horizon"]
    a,b=current["overall"],current["vehicle.moving"]
    return (a["count"]==old["overall"]["count"] and b["count"]==old["vehicle.moving"]["count"] and b["count"]>0
            and a["minADE"]<old["overall"]["minADE"]*gate["max_overall_to_failed_ADE_ratio"]
            and a["minFDE"]<old["overall"]["minFDE"]*gate["max_overall_to_failed_FDE_ratio"]
            and b["minADE"]<old["vehicle.moving"]["minADE"]*gate["max_moving_to_failed_ADE_ratio"]
            and b["minFDE"]<old["vehicle.moving"]["minFDE"]*gate["max_moving_to_failed_FDE_ratio"]
            and b["minADE"]<gate["max_moving_ADE_m"] and b["minFDE"]<gate["max_moving_FDE_m"])


def full_tiny():
    selected=json.loads((ROOT/"selected_protocol.json").read_text())
    assert json.loads((ROOT/"A_fixed_scale/metrics.json").read_text())["status"]=="PASS"
    if selected["protocol"]==1:
        assert json.loads((ROOT/"B_result.json").read_text())["status"]=="PASS"
    graphs=original_tiny_graphs();attributes=attribute_index()
    reference_result=json.loads((ROOT/"tiny_reference_comparison.json").read_text())
    settings=baseline_config()["tiny"];config=read_config()
    gate=lambda e:tiny_gate(e,reference_result,settings["gate"])
    if selected["protocol"]==1:
        cv=reference_result["CV"]["metrics"]["full_horizon"]["vehicle.moving"]
        def preferred(e):
            overall=e["metrics"]["full_horizon"]["overall"];moving=e["metrics"]["full_horizon"]["vehicle.moving"]
            ratio=settings["preferred_warmup_max_moving_to_CV_ratio"]
            return (overall["minADE"]<settings["preferred_warmup_max_overall_ADE_m"]
                    and overall["minFDE"]<settings["preferred_warmup_max_overall_FDE_m"]
                    and moving["minADE"]<=cv["minADE"]*ratio and moving["minFDE"]<=cv["minFDE"]*ratio)
        warm,_,_=train_phase("full_tiny_warmup",graphs,attributes,
                            {"phase":"fixed_scale","lr":settings["fixed_lr"],"max_epochs":settings["warmup_max_epochs"],
                             "gate_scope":"preferred early transition; reaching budget also permits NLL phase"},stop_gate=preferred)
        model,optimizer=restore_checkpoint(ROOT/"full_tiny_warmup/checkpoint.pt",config,settings["nll_lr"])
        recovered,_,_=train_phase("full_tiny_original_nll",graphs,attributes,
                                 {"phase":"original_nll","lr":settings["nll_lr"],"max_epochs":settings["original_nll_epochs"],
                                  "full_tiny_gate":settings["gate"]},model=model,optimizer=optimizer,final_gate=gate)
    else:
        recovered,_,_=train_phase("full_tiny_final",graphs,attributes,
                                 {"phase":selected["phase"],"b_max":selected["b_max"],"lr":settings["fixed_lr"],
                                  "max_epochs":settings["warmup_max_epochs"]+settings["original_nll_epochs"],"full_tiny_gate":settings["gate"]},stop_gate=gate)
    result={"status":recovered["status"],"protocol":selected,"experiment":recovered["experiment"],
            "epochs_final_phase":recovered["epochs"],"final":recovered["final"],"gate":settings["gate"],
            "source_windows":"outputs/stage2/tiny_overfit/training_windows.json","windows":16,
            "source_targets":382,"full_horizon_count":236,"partial_future_count":146}
    write_json(ROOT/"full_tiny_result.json",result)
    comparison={**reference_result,"Recovered HiVT":recovered["final"]}
    for method,measured in comparison.items():
        assert measured["metrics"]["full_horizon"]["overall"]["count"]==236
        assert measured["metrics"]["partial_future"]["overall"]["count"]==146
    write_json(ROOT/"tiny_comparison.json",comparison)
    # Compact, actor-mean table for the required original/CV/recovered contrast.
    with open(ROOT/"tiny_comparison.csv","w",newline="") as f:
        import csv
        writer=csv.DictWriter(f,["method","horizon","attribute","count","minADE","minFDE","MR"],lineterminator="\n")
        writer.writeheader()
        for method,measured in comparison.items():
            for horizon,groups in measured["metrics"].items():
                for attribute,values in groups.items():writer.writerow({"method":method,"horizon":horizon,"attribute":attribute,**{k:values[k] for k in ("count","minADE","minFDE","MR")}})
    verify_previous()
    print("FULL_TINY_RESULT",json.dumps({"status":result["status"],"metrics":result["final"]["metrics"]}),flush=True)
    if result["status"]!="PASS":print("HARD_STOP: Full Tiny FAIL; Mini prohibited",flush=True)


@torch.no_grad()
def mini_figures(model,graphs,evaluation):
    """Real t0 attributes, full-horizon cases; success means FDE <= 2m."""
    full=[r for r in evaluation["actors"] if r["horizon"]=="full_horizon"]
    moving=[r for r in full if r["attribute_group"]=="vehicle.moving" and r["GT_endpoint_displacement_m"]>=5.]
    pools={"moving_success":sorted([r for r in moving if r["minFDE"]<=2.],key=lambda r:r["minFDE"]),
           "moving_failure":sorted([r for r in moving if r["minFDE"]>2.],key=lambda r:-r["minFDE"]),
           "stopped":sorted([r for r in full if r["attribute_group"]=="vehicle.stopped"],key=lambda r:r["minFDE"]),
           "parked":sorted([r for r in full if r["attribute_group"]=="vehicle.parked"],key=lambda r:r["minFDE"])}
    required={"moving_success":5,"moving_failure":5,"stopped":2,"parked":2}
    index={(g.scene_token,g.sample_token):g for g in graphs}
    cases=[];coverage={}
    for kind,pool in pools.items():
        selected=[];instances=set();identities=set()
        for require_unique_instance in (True,False):
            for row in pool:
                identity=(row["sample_token"],row["instance_token"])
                if identity in identities or (require_unique_instance and row["instance_token"] in instances):continue
                selected.append(row);instances.add(row["instance_token"]);identities.add(identity)
                if len(selected)==required[kind]:break
            if len(selected)==required[kind]:break
        coverage[kind]={"requested":required[kind],"available_actor_windows":len(pool),"produced":len(selected),
                        "distinct_instances":len(instances)}
        for i,row in enumerate(selected,1):
            g=index[(row["scene_token"],row["sample_token"])];node=row["node_in_graph"]
            model.eval();data=g.clone().cuda();output=model(data)
            prediction=model.ego_predictions(output,data).detach().cpu();prob=output["mode_prob"].cpu()
            path=ROOT/"mini_figures"/f"{kind}_{i:02d}.png"
            plot_case(g,prediction,prob,node,path,
                      f"{kind}: ADE={row['minADE']:.2f}m, FDE={row['minFDE']:.2f}m; GT travel={row['GT_endpoint_displacement_m']:.1f}m",
                      fit_all_modes=True)
            cases.append({**row,"case_group":kind,"path":str(path.relative_to(PROJECT_ROOT)),
                          "trajectory_ego_m":prediction[node].tolist(),"mode_probabilities":prob[node].tolist()})
    result={"source":"validation only; true t0 attribute names",
            "moving_visual_GT_min_displacement_m":5.,"success_max_FDE_m":2.,"coverage":coverage,"cases":cases,
            "complete":all(v["produced"]>=v["requested"] for v in coverage.values())}
    write_json(ROOT/"mini_visualization_audit.json",result)
    return result


def mini():
    tiny=json.loads((ROOT/"full_tiny_result.json").read_text())
    assert tiny["status"]=="PASS","Full Tiny FAIL prohibits Mini"
    selected=json.loads((ROOT/"selected_protocol.json").read_text())
    assert selected["protocol"]==1,"This mini configuration is predeclared for Protocol 1"
    corpus=torch.load(PROJECT_ROOT/"outputs/stage2/mini/processed/graphs.pt",weights_only=False,map_location="cpu")
    graphs=corpus["train"];validation=corpus["val"]
    train_scenes={g.scene_token for g in graphs};val_scenes={g.scene_token for g in validation}
    assert len(graphs)==146 and len(validation)==48 and len(train_scenes)==6 and len(val_scenes)==2
    assert not train_scenes & val_scenes
    assert all(g.x.shape[1]==5 and g.y.shape[1]==12 for g in graphs+validation)
    attributes=attribute_index();settings=baseline_config()["mini"];config=read_config()
    # Completion is checked here; no accuracy threshold or test-dependent search was authorized for mini.
    finite=lambda e:all(torch.isfinite(torch.tensor(e[k])) for k in ("ADE","FDE","loss","NLL"))
    warm,_,_=train_phase("mini_warmup",graphs,attributes,
                        {"phase":"fixed_scale","lr":settings["fixed_lr"],"max_epochs":settings["warmup_epochs"],
                         "gate_scope":"finite completion of predeclared mini warm-up budget"},final_gate=finite)
    model,optimizer=restore_checkpoint(ROOT/"mini_warmup/checkpoint.pt",config,settings["nll_lr"])
    nll,_,_=train_phase("mini_original_nll",graphs,attributes,
                       {"phase":"original_nll","lr":settings["nll_lr"],"max_epochs":settings["original_nll_epochs"],
                        "gate_scope":"finite completion; select final-phase checkpoint using validation FDE only",
                        "checkpoint_selection":settings["checkpoint_selection"]},model=model,optimizer=optimizer,
                       final_gate=finite,validation_graphs=validation)
    checkpoint=ROOT/"mini_original_nll/best_validation_checkpoint.pt"
    model,_=restore_checkpoint(checkpoint,config,settings["nll_lr"])
    measured=evaluate(model,validation,attributes)
    assert abs(measured["FDE"]-nll["best_validation"]["FDE"])<1e-5
    cv=evaluate_cv(validation,attributes)
    for horizon in ("full_horizon","partial_future"):
        for group in GROUPS:
            assert measured["metrics"][horizon][group]["count"]==cv["metrics"][horizon][group]["count"]
    figures=mini_figures(model,validation,measured)
    result={"status":"COMPLETE" if figures["complete"] else "INCOMPLETE_VISUALS","protocol":selected,
            "fresh_initialization":True,"train_windows":len(graphs),"validation_windows":len(validation),
            "train_scene_tokens":sorted(train_scenes),"validation_scene_tokens":sorted(val_scenes),
            "split_note":"existing Stage 1 custom mini 6/2/2 scene split; not official benchmark",
            "test_used_for_tuning":False,"test_evaluated":False,"warmup_epochs":warm["epochs"],"NLL_epochs":nll["epochs"],
            "checkpoint_selection":settings["checkpoint_selection"],"selected_epoch":nll["best_validation"]["epoch"],
            "checkpoint_path":str(checkpoint.relative_to(PROJECT_ROOT)),"CV":cv,"HiVT":measured,
            "visualization_coverage":figures["coverage"],"visualizations_complete":figures["complete"],
            "completion_note":"engineering baseline completed; this status does not assert generalization superiority to CV"}
    write_json(ROOT/"mini_result.json",result)
    verify_previous()
    print("MINI_RESULT",json.dumps({"status":result["status"],"selected_epoch":result["selected_epoch"],
                                    "CV":cv["metrics"],"HiVT":measured["metrics"],"visuals":figures["coverage"]}),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--stage",choices=("reference","tiny","mini"),required=True);args=parser.parse_args()
    torch.set_num_threads(4)
    if args.stage=="reference":reference()
    elif args.stage=="tiny":full_tiny()
    else:mini()


if __name__=="__main__":main()
