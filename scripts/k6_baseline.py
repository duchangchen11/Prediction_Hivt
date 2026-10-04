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
                                     summarize_records, verify_previous)
from scripts.train_hivt_stage2 import MODEL_KEYS


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


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--stage",choices=("reference",),required=True);args=parser.parse_args()
    torch.set_num_threads(4);reference()


if __name__=="__main__":main()
