"""Train-only, complete-context three-class sanity; abort formal training on failure."""
import argparse
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "00_manifest"))
from stage3_common import (CLASSES, SceneDataset, atomic_json, config, evaluate_graphs,
                          git, model_new, model_input, loss_diagnostics, read_json,
                          sha256, verify_frozen, update_manifest, write_csv)
import torch
from torch_geometric.data import Batch
import yaml


def finite_gradients(model):
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert any(float(g.abs().max()) > 0 for g in gradients)


def choose_tiny():
    ds = SceneDataset("train")
    candidates = [i for i,r in enumerate(ds.rows) if all(int(r["full_"+c]) > 0 for c in CLASSES)]
    candidates.sort(key=lambda i: (int(ds.rows[i]["actor_count"]), -int(ds.rows[i]["full_bicycle"]), ds.rows[i]["sample_token"]))
    assert candidates, "No train windows contain eligible full-horizon examples of all three classes"
    graphs = []; totals = [0,0,0]; records = []
    for i in candidates:
        g = ds[i]
        counts = [int((g.target_mask & (g.agent_type==t)).sum()) for t in range(3)]
        graphs.append(g); totals = [a+b for a,b in zip(totals,counts)]
        records.append({"scene_name":g.scene_name,"scene_token":g.scene_token,"sample_token":g.sample_token,
                        "actor_count":g.num_nodes,"targets_per_class":counts,"index":i})
        if all(n>=minimum for n,minimum in zip(totals,config()["tiny"]["minimum_targets"])):break
    assert all(n>=minimum for n,minimum in zip(totals,config()["tiny"]["minimum_targets"]))
    ds.clear()
    atomic_json(ROOT / "00_manifest/stage3_tiny_selection.json",
                {"selection":"train-only three-class windows, minimum context size for bounded sanity; all actors retained",
                 "target_counts":dict(zip(CLASSES,totals)),"windows":records,"full_scene_context_preserved":True,"val_used":False})
    return graphs


def io_checks(model, graphs):
    model.eval(); data=Batch.from_data_list(graphs[:2]).cuda()
    working=model_input(data); assert "agent_type" not in working
    with torch.no_grad():
        original=model(working); prediction=model.ego_predictions(original,data)
        targets=data.target_mask
        assert prediction[targets].shape==(int(targets.sum()),6,12,2)
        assert original["mode_prob"][targets].shape==(int(targets.sum()),6)
        perturbed=data.clone();perturbed.agent_type=(data.agent_type+1)%3
        perturbed.positions[:,5:]+=1234;perturbed.y-=777
        perturbed.future_mask=~perturbed.future_mask;perturbed.padding_mask[:,5:]=~perturbed.padding_mask[:,5:]
        changed=model(model_input(perturbed))
        torch.testing.assert_close(original["raw_prediction"],changed["raw_prediction"],rtol=0,atol=0)
        torch.testing.assert_close(original["mode_prob"],changed["mode_prob"],rtol=0,atol=0)
    per_class={}
    for t,name in enumerate(CLASSES):
        model.zero_grad(set_to_none=True);sample=data.clone();sample.target_mask=data.target_mask&(data.agent_type==t)
        assert sample.target_mask.any()
        out=model(model_input(sample));values=model.recovery_loss(out,sample,"original_nll")
        assert all(torch.isfinite(v) for v in values.values())
        values["loss"].backward();finite_gradients(model)
        per_class[name]={"targets":int(sample.target_mask.sum()),"loss":float(values["loss"].detach()),"finite_nonzero_gradient":True}
    model.zero_grad(set_to_none=True)
    atomic_json(ROOT / "00_manifest/stage3_input_output_checks.json",
                {"status":"PASS","target_prediction_shape":list(prediction[targets].shape),"target_mode_probability_shape":list(original['mode_prob'][targets].shape),
                 "per_class":per_class,"agent_type_removed_from_model_input":True,"type_and_future_perturbation_invariance":True,
                 "model_class":type(model).__name__,"architecture_identical_to_Stage2C":True,"parameter_count":sum(p.numel() for p in model.parameters()),
                 "smoke_gradients_discarded":True,"validation_optimizer_steps":0})


def adequate(initial,current):
    ratio=1-config()["tiny"]["minimum_relative_improvement"]
    return all(current[c][k] < ratio*initial[c][k] for c in CLASSES for k in ("ADE","FDE","fixed_scale_regression_loss"))


def run():
    assert read_json(ROOT/"02_preprocessed/stage3_preprocess_manifest.json")["status"]=="COMPLETE"
    assert read_json(ROOT/"01_data_audit/stage3_coordinate_checks.json")["status"]=="PASS"
    graphs=choose_tiny();model=model_new();io_checks(model,graphs)
    # Fresh deterministic initialization after all I/O and gradient checks.
    model=model_new();optimizer=model.optimizer(config()["tiny"]["warmup_lr"],config()["weight_decay"])
    initial=evaluate_graphs(model,graphs);rows=[];started=time.monotonic();global_step=0
    tiny_config=ROOT/"03_no_type_baseline/stage3_tiny_config.yaml"
    tiny_config.write_text(yaml.safe_dump(config(),sort_keys=False))
    best_state=None;best_optimizer=None;best_fde=float("inf");warm_steps=0;warm_source=0
    for phase,max_key,lr_key in (("fixed_scale","warmup_max_steps","warmup_lr"),("original_nll","nll_max_steps","nll_lr")):
        if phase=="original_nll":
            assert best_state is not None
            model.load_state_dict(best_state);optimizer.load_state_dict(best_optimizer)
        for group in optimizer.param_groups:group["lr"]=config()["tiny"][lr_key]
        for step in range(1,config()["tiny"][max_key]+1):
            start=((step-1)*2)%len(graphs)
            chosen=[graphs[(start+j)%len(graphs)] for j in range(min(2,len(graphs)))]
            data=Batch.from_data_list(chosen).cuda();model.train();optimizer.zero_grad(set_to_none=True)
            output=model(model_input(data));values=loss_diagnostics(model,output,data,phase)
            assert all(torch.isfinite(v) for v in values.values())
            values["loss"].backward();finite_gradients(model);optimizer.step();global_step+=1
            if step%config()["tiny"]["check_every"]==0 or step==config()["tiny"][max_key]:
                current=evaluate_graphs(model,graphs)
                row={"global_step":global_step,"phase":phase,"phase_step":step,"train_loss":float(values["loss"].detach()),
                     "regression_loss":float(values["regression_loss"].detach()),"classification_loss":float(values["classification_loss"].detach())}
                for name in CLASSES:
                    for field,value in current[name].items():row[name+"_"+field]=value
                rows.append(row);write_csv(ROOT/"03_no_type_baseline/stage3_tiny_curve.csv",rows)
                print("TINY",phase,step,{c:(current[c]['ADE'],current[c]['FDE']) for c in CLASSES},flush=True)
                if phase=="fixed_scale" and current["overall"]["FDE"]<best_fde:
                    best_fde=current["overall"]["FDE"];warm_source=step
                    best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                    import copy
                    best_optimizer=copy.deepcopy(optimizer.state_dict())
                if phase=="fixed_scale" and step>=config()["tiny"]["warmup_min_steps"] and adequate(initial,current):break
        if phase=="fixed_scale":warm_steps=step
    final=evaluate_graphs(model,graphs);status="PASS" if adequate(initial,final) else "FAIL"
    result={"status":status,"initial":initial,"final":final,"criterion":"all three classes ADE, FDE and fixed-scale diagnostic regression decrease at least50%; all losses/gradients finite",
            "warmup_steps":warm_steps,"warmup_best_source_step":warm_source,"nll_steps":step,"global_steps":global_step,
            "elapsed_seconds":time.monotonic()-started,"full_scene_context_preserved":True,"train_only":True,"model_discarded_for_formal_training":True,
            "NaN":0,"Inf":0,"optimizer_state_preserved_from_best_warmup":True,"git_commit_SHA":git('rev-parse','HEAD'),"config_sha256":sha256(tiny_config)}
    checkpoint=ROOT/"07_checkpoints/stage3_tiny_last.pt";temp=checkpoint.with_name(checkpoint.name+".tmp")
    torch.save({"state_dict":model.state_dict(),"optimizer_state_dict":optimizer.state_dict(),"metadata":result},temp);os.replace(temp,checkpoint)
    atomic_json(ROOT/"00_manifest/stage3_checkpoint_manifest.json",{"checkpoints":{str(checkpoint.relative_to(ROOT)):{"global_step":global_step,"phase":"tiny original_nll","sha256":sha256(checkpoint),"git_tracked":False,"config_sha256":sha256(tiny_config),"selection_metric":"tiny diagnostic only","git_commit_SHA":git('rev-parse','HEAD')}}})
    atomic_json(ROOT/"03_no_type_baseline/stage3_tiny_overfit.json",result)
    verify_frozen();update_manifest();print("TINY_OVERFIT="+status,flush=True)
    assert status=="PASS","At least one class failed tiny learning; stop before formal training"


if __name__=="__main__":
    torch.set_num_threads(4)
    try:run()
    except Exception as error:
        atomic_json(ROOT/"03_no_type_baseline/stage3_tiny_failure.json",{"status":"FAIL","error":repr(error),"formal_training_started":False})
        raise
