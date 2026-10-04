"""Source-traceable validation cases and an editable publication main figure."""
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "00_manifest"))
from stage2c_common import (SceneShardDataset, atomic_json, config, model_new, read_json,
                           sha256, update_manifest, verify_previous, write_csv)
import numpy as np
import torch
from models.constant_velocity import constant_velocity
from scripts.plot_hivt_predictions import plot_case
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def visible_lane_segments(g, prediction, node):
    history=g.positions[node,:5].numpy()[g.history_mask[node].numpy()]
    valid=np.concatenate([history,g.positions[node,5:].numpy(),prediction[node].numpy().reshape(-1,2)])
    lower,upper=valid.min(0)-10,valid.max(0)+10
    center=(lower+upper)/2;half=max((upper-lower).max()/2,15)
    endpoints=np.stack([g.lane_positions.numpy(),(g.lane_positions+g.lane_vectors).numpy()],axis=1)
    inside=((endpoints>=center-half+1)&(endpoints<=center+half-1)).all(axis=-1).any(axis=-1)
    return int(inside.sum())


def rows_from_csv(path):
    with open(path) as f:
        for row in csv.DictReader(f):
            for k in ("GT_endpoint_displacement_m", "minADE6", "minFDE6", "MR6", "independent_minADE6"):row[k]=float(row[k])
            row["node_in_graph"]=int(row["node_in_graph"])
            yield row


@torch.no_grad()
def numerical_case(model, dataset, index, row):
    g=dataset[index[(row["scene_token"],row["sample_token"])]];node=row["node_in_graph"]
    assert g.instance_tokens[node]==row["instance_token"]
    data=g.clone().cuda();model.eval();output=model(data)
    prediction=model.ego_predictions(output,data).cpu();prob=output["mode_prob"].cpu()
    cv,_,_=constant_velocity(g.positions[:,:5],g.history_mask,g.history_times,g.future_times)
    gt=g.positions[node,5:].numpy();best=int(np.linalg.norm(prediction[node].numpy()[:,-1]-gt[-1],axis=1).argmin())
    cv_error=np.linalg.norm(cv[node].numpy()-gt,axis=1)
    turn,lateral=turn_geometry(g,node)
    source={**row,"scene_name":g.scene_name,"history_trajectory_m":g.positions[node,:5].tolist(),
            "history_mask":g.history_mask[node].tolist(),"history_times_seconds":g.history_times.tolist(),
            "GT_trajectory_m":gt.tolist(),"CV_trajectory_m":cv[node].tolist(),
            "HiVT_trajectories_m":prediction[node].tolist(),"mode_probabilities":prob[node].tolist(),
            "best_FDE_mode_zero_based":best,"future_times_seconds":g.future_times.tolist(),"turn_degrees":turn,
            "turn_definition":"angle between first and last three future displacement steps; each three-step vector >=1m",
            "GT_lateral_deviation_from_initial_direction_m":lateral,
            "CV_ADE_m":float(cv_error.mean()),"CV_FDE_m":float(cv_error[-1]),
            "coordinate_frame":"t0 ego +x forward, +y left, meters", "source":"official VAL; primary overall-FDE checkpoint"}
    return g,prediction,prob,source


def turn_geometry(g,node):
    gt=g.positions[node,5:].numpy();start=g.positions[node,4].numpy();initial=gt[2]-start;final=gt[-1]-gt[-4]
    first_length,last_length=np.linalg.norm(initial),np.linalg.norm(final)
    turn=float(np.degrees(np.arccos(np.clip(np.dot(initial,final)/(first_length*last_length),-1,1)))) if min(first_length,last_length)>=1 else 0.
    delta=gt-start
    lateral=float(np.abs(delta[:,0]*initial[1]-delta[:,1]*initial[0]).max()/first_length) if first_length>=1 else 0.
    return turn,lateral


def main_figure(g,node,source):
    # Contract: one selected turning example illustrates a measured gain; aggregate claims live in the full-VAL tables/CI.
    contract={"conclusion":"In this real moving validation turning case, the oracle best-FDE HiVT mode has smaller ADE/FDE than past-only CV, with remaining trajectory error shown explicitly.",
              "evidence_chain":{"a":"map/history/GT/CV and all six primary-checkpoint trajectories",
                                "b":"per-future-step displacement error from the same stored numerical arrays",
                                "c":"unaltered model mode probabilities; best-FDE choice identified"},
              "archetype":"asymmetric mixed-modality figure","backend":"Python/matplotlib",
              "dimensions_mm":[183,110],"exports":["PNG 600dpi","SVG editable text","PDF embedded TrueType"],
              "review_risks":["Selected case is qualitative evidence, not an independent sample or aggregate superiority claim.",
                              "Best-FDE mode is oracle-selected for evaluation, not a top-probability deployment prediction."],
              "source_data":"04_evaluation/stage2c_qualitative_main_case.json"}
    atomic_json(ROOT / "00_manifest/stage2c_main_figure_contract.json",contract)
    style={"font.family":"sans-serif","font.sans-serif":["DejaVu Sans"],"font.size":7,"svg.fonttype":"none","pdf.fonttype":42,
           "axes.spines.top":False,"axes.spines.right":False,"axes.linewidth":.7,"legend.frameon":False,
           "axes.grid":False,"axes.facecolor":"white"}
    with plt.rc_context(style):
        fig=plt.figure(figsize=(183/25.4,110/25.4),layout="constrained");grid=fig.add_gridspec(2,2,width_ratios=[2.6,1.2])
        ax=fig.add_subplot(grid[:,0]);errors_ax=fig.add_subplot(grid[0,1]);prob_ax=fig.add_subplot(grid[1,1])
        lanes=torch.stack([g.lane_positions,g.lane_positions+g.lane_vectors],dim=1).numpy()
        ax.add_collection(LineCollection(lanes,colors="#B7BEC6",linewidths=.45,alpha=.65))
        history=np.array(source["history_trajectory_m"]);history[~np.array(source["history_mask"])]=np.nan
        gt=np.array(source["GT_trajectory_m"]);cv=np.array(source["CV_trajectory_m"]);pred=np.array(source["HiVT_trajectories_m"])
        start=history[-1];best=source["best_FDE_mode_zero_based"]
        ax.plot(history[:,0],history[:,1],"o-",color="#6B7280",ms=2,lw=1,label="History")
        first_other=True
        for k,p in enumerate(pred):
            if k==best:continue
            xy=np.vstack([start,p]);ax.plot(xy[:,0],xy[:,1],color="#8CAFCB",lw=.65,ls="--",alpha=.6,label="Other HiVT modes" if first_other else None)
            first_other=False
        for xy,color,ls,label,lw in ((np.vstack([start,cv]),"#CB8B3D","--","CV",1.3),
                                     (np.vstack([start,gt]),"#252525","-","GT",1.4),
                                     (np.vstack([start,pred[best]]),"#24618D","-",f"HiVT best-FDE mode {best+1}",1.6)):
            ax.plot(xy[:,0],xy[:,1],color=color,ls=ls,lw=lw,label=label)
        visible=np.concatenate([history[np.isfinite(history).all(1)],gt,cv,pred.reshape(-1,2)])
        lo=visible.min(0)-5;hi=visible.max(0)+5;center=(lo+hi)/2;half=(hi-lo).max()/2
        ax.set_xlim(center[0]-half,center[0]+half);ax.set_ylim(center[1]-half,center[1]+half);ax.set_aspect("equal")
        ax.set_xlabel("t0 ego forward x (m)");ax.set_ylabel("t0 ego left y (m)")
        ax.legend(fontsize=6,loc="lower left",bbox_to_anchor=(0,1.01),ncol=2,columnspacing=.9)
        times=np.array(source["future_times_seconds"])
        errors_ax.plot(times,np.linalg.norm(cv-gt,axis=1),color="#CB8B3D",ls="--",lw=1,label="CV")
        errors_ax.plot(times,np.linalg.norm(pred[best]-gt,axis=1),color="#24618D",lw=1,label="HiVT")
        errors_ax.set(xlabel="Time after t0 (s)",ylabel="Displacement error (m)");errors_ax.legend(fontsize=6)
        errors_ax.grid(axis="y",color="#D8DEE4",linewidth=.4)
        probabilities=source["mode_probabilities"]
        prob_ax.bar(np.arange(1,7),probabilities,color=["#24618D" if k==best else "#9BB6CA" for k in range(6)],width=.65)
        for k,p in enumerate(probabilities):prob_ax.text(k+1,p+.012,f"{p:.2f}",ha="center",fontsize=5.5)
        prob_ax.set(xlabel="Mode",ylabel="Probability",xticks=range(1,7),ylim=(0,max(probabilities)+.08))
        for label,a in zip("abc",[ax,errors_ax,prob_ax]):a.text(-.06,1.02,label,transform=a.transAxes,fontweight="bold",fontsize=9)
        fig.suptitle(f"{source['scene_name']} | sample {source['sample_token'][:8]} | actor {source['instance_token'][:8]} | {source['motion_state']}\n"
                     f"HiVT minADE/minFDE {source['minADE6']:.2f}/{source['minFDE6']:.2f}m; CV {source['CV_ADE_m']:.2f}/{source['CV_FDE_m']:.2f}m; turn {source['turn_degrees']:.0f}°",fontsize=8)
        stem=ROOT / "05_figures/stage2c_qualitative_main_case"
        fig.savefig(str(stem)+".png",dpi=600);fig.savefig(str(stem)+".svg");fig.savefig(str(stem)+".pdf");plt.close(fig)
        svg=Path(str(stem)+".svg");svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines())+"\n")


@torch.no_grad()
def main():
    c=config();hrows=list(rows_from_csv(ROOT / "04_evaluation/stage2c_val_actor_errors.csv"));crows=list(rows_from_csv(ROOT / "04_evaluation/stage2c_cv_actor_errors.csv"))
    pairs={(r["sample_token"],r["instance_token"]):r for r in crows}
    full=[r for r in hrows if r["horizon"]=="full_horizon"]
    moving=[r for r in full if r["motion_state"]=="vehicle.moving" and r["GT_endpoint_displacement_m"]>=c["visualization"]["moving_min_GT_displacement_m"]]
    pools={"moving_success":sorted([r for r in moving if r["minFDE6"]<=2 and r["minADE6"]<=2],key=lambda r:(r["minADE6"],r["minFDE6"])),
           "moving_failure":sorted([r for r in moving if r["minFDE6"]>2],key=lambda r:-r["minFDE6"]),
           "stopped_case":sorted([r for r in full if r["motion_state"]=="vehicle.stopped"],key=lambda r:r["minFDE6"]),
           "parked_case":sorted([r for r in full if r["motion_state"]=="vehicle.parked"],key=lambda r:r["minFDE6"])}
    required={"moving_success":5,"moving_failure":5,"stopped_case":3,"parked_case":3}
    ds=SceneShardDataset("val");index={(r["scene_token"],r["sample_token"]):i for i,r in enumerate(ds.rows)}
    checkpoint=ROOT / "07_checkpoints/stage2c_best_overall_minfde.pt";saved=torch.load(checkpoint,weights_only=False,map_location="cpu")
    model=model_new();model.load_state_dict(saved["state_dict"]);cases=[];coverage={}
    for name,pool in pools.items():
        produced=0;seen=set()
        for row in pool:
            if row["instance_token"] in seen:continue
            g,pred,prob,source=numerical_case(model,ds,index,row)
            visible_lanes=visible_lane_segments(g,pred,row["node_in_graph"])
            if visible_lanes<5:continue
            seen.add(row["instance_token"]);produced+=1
            source["visible_lane_segments_in_case_viewport"]=visible_lanes
            path=ROOT / "05_figures" / f"stage2c_{name}_{produced:03d}.png"
            label=f"{g.scene_name} | sample {row['sample_token'][:8]} | actor {row['instance_token'][:8]}\n{row['motion_state']} | minADE={row['minADE6']:.2f}m | minFDE={row['minFDE6']:.2f}m"
            plot_case(g,pred,prob,row["node_in_graph"],path,label,fit_all_modes=True)
            cases.append({**source,"figure_path":str(path.relative_to(ROOT))})
            if produced==required[name]:break
        assert produced==required[name],"Insufficient distinct map-visible cases; do not fabricate examples"
        coverage[name]={"requested":required[name],"produced":produced,"available":len(pool)}
    candidates=[r for r in moving
                if pairs[(r["sample_token"],r["instance_token"])]["minFDE6"]-r["minFDE6"]>=c["visualization"]["main_case_min_CV_FDE_gain_m"]
                and r["minFDE6"]<=.75*pairs[(r["sample_token"],r["instance_token"])]["minFDE6"]
                and r["minADE6"]<=.75*pairs[(r["sample_token"],r["instance_token"])]["minADE6"]]
    geometric=[];current_key=None;current_graph=None
    for number,row in enumerate(candidates,1):
        key=(row["scene_token"],row["sample_token"])
        if key!=current_key:current_graph=ds[index[key]];current_key=key
        turn,lateral=turn_geometry(current_graph,row["node_in_graph"])
        if turn>=max(60,c["visualization"]["main_case_min_turn_degrees"]) and lateral>=3:
            geometric.append({**row,"geometry_turn_degrees":turn,"geometry_lateral_deviation_m":lateral,
                              "paired_CV_ADE_m":pairs[(row["sample_token"],row["instance_token"])]["minADE6"],
                              "paired_CV_FDE_m":pairs[(row["sample_token"],row["instance_token"])]["minFDE6"]})
        if number%500==0:print("MAIN_GEOMETRY_SCREEN",number,"/",len(candidates),"qualified",len(geometric),flush=True)
    candidates=sorted(geometric,key=lambda r:(r["minFDE6"],r["minADE6"],-pairs[(r["sample_token"],r["instance_token"])]["minFDE6"]+r["minFDE6"]))
    write_csv(ROOT / "04_evaluation/stage2c_main_case_candidate_audit.csv",candidates)
    chosen_main=None
    for row in candidates:
        g,pred,prob,source=numerical_case(model,ds,index,row)
        if source["turn_degrees"]>=max(60,c["visualization"]["main_case_min_turn_degrees"]) and source["CV_ADE_m"]-source["minADE6"]>=.5 and source["GT_lateral_deviation_from_initial_direction_m"]>=3:
            chosen_main=source;chosen_main["selection_criteria"]=c["visualization"]
            chosen_main["visual_review_minimum_turn_degrees"]=60
            chosen_main["visual_review_minimum_lateral_deviation_m"]=3
            chosen_main["visual_review_maximum_HiVT_to_CV_error_ratio"]=.75
            chosen_main["primary_checkpoint_sha256"]=sha256(checkpoint)
            chosen_main["lane_positions_m"]=g.lane_positions.tolist()
            chosen_main["lane_vectors_m"]=g.lane_vectors.tolist()
            chosen_main["origin_global_m"]=g.origin.tolist()
            chosen_main["ego_yaw_global_rad"]=float(g.ego_yaw)
            atomic_json(ROOT / "04_evaluation/stage2c_qualitative_main_case.json",chosen_main)
            main_figure(g,row["node_in_graph"],source);break
    atomic_json(ROOT / "04_evaluation/stage2c_figure_case_manifest.json",{"coverage":coverage,"cases":cases,
               "main_case_found":chosen_main is not None,"checkpoint_sha256":sha256(checkpoint),
               "success_definition":"real t0 moving, GT travel>=5m, minADE/minFDE<=2m; failure FDE>2m; distinct instances",
               "visual_review":"Case views require at least five visible original lane segments. Main case requires sustained turn>=60 degrees, lateral deviation>=3m, and ADE/FDE<=75% of CV; it is ranked by HiVT FDE then ADE and need not meet the separate <=2m success-case threshold. Candidate audit CSV retained. Square metric axes. Qualitative selection only; fixed full VAL metrics/checkpoint unchanged."})
    assert chosen_main is not None,"No supported turning main case; report this limitation without inventing one"
    verify_previous();update_manifest();print("FIGURES_COMPLETE",coverage,"main turn",chosen_main["turn_degrees"],flush=True)


if __name__=="__main__":torch.set_num_threads(4);main()
