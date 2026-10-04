"""Evidence-based Stage 2C status, results and research decision report."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT / "00_manifest"))
from stage2c_common import GROUPS, atomic_json, config, git, read_json, sha256, update_manifest, verify_previous


def optional(relative):
    p=ROOT / relative
    return read_json(p) if p.exists() else None


def fmt(value):return "NOT_RUN" if value is None else f"{value:.6f}"


def research_decision(hivt,cv):
    c=config();guard=c["acceptance"]
    h=hivt["metrics"]["full_horizon"];v=cv["metrics"]["full_horizon"]
    ratios={g:{m:h[g][m]/v[g][m] for m in ("minADE6","minFDE6")} for g in ("overall","vehicle.moving")}
    better={g:all(r<1. for r in values.values()) for g,values in ratios.items()}
    mild=all(r<=guard["case_B_maximum_moving_to_CV_ratio"] for r in ratios["vehicle.moving"].values())
    clearly_worse={g:all(r>guard["case_B_maximum_moving_to_CV_ratio"] for r in values.values()) for g,values in ratios.items()}
    reasonable=h["overall"]["minFDE6"]<guard["maximum_overall_FDE_m"] and h["vehicle.moving"]["minFDE6"]<guard["maximum_moving_FDE_m"]
    if better["overall"] and better["vehicle.moving"]:case="A";reason="Overall and moving ADE/FDE both improve on CV."
    elif better["overall"] and mild:case="B";reason="Overall improves; moving remains within the predeclared 15% margin. Keep the moving weakness."
    elif clearly_worse["overall"] and clearly_worse["vehicle.moving"]:case="C";reason="Both overall and moving are clearly worse than CV; baseline investigation must precede Stage3."
    else:case="MIXED";reason="The result does not meet A/B; report the mixed metrics without automatic Stage3 authorization."
    return {"case":case,"reason":reason,"reasonable_generalization":reasonable,"HiVT_to_CV_ratios":ratios,
            "Stage2C":"PASS" if reasonable and case in ("A","B") else "FAIL",
            "Allow_Stage3":"YES" if reasonable and case in ("A","B") else "NO","Stage3_executed":False,
            "test_used":False,"criterion_note":"A/B compare point estimates; scene bootstrap CI separately quantifies uncertainty."}


def main():
    split=optional("01_data_audit/stage2c_nuscenes_split_audit.json")
    prep=optional("02_preprocessed/stage2c_preprocess_manifest.json")
    smoke=optional("00_manifest/stage2c_preprocess_smoke.json")
    plan=optional("00_manifest/stage2c_training_plan.json")
    warm=optional("03_training/stage2c_warmup_summary.json");nll=optional("03_training/stage2c_nll_summary.json")
    h=optional("04_evaluation/stage2c_val_metrics.json");cv=optional("04_evaluation/stage2c_cv_metrics.json")
    boot=optional("04_evaluation/stage2c_bootstrap_ci.json")
    figures=optional("04_evaluation/stage2c_figure_case_manifest.json")
    decision=research_decision(h,cv) if h and cv else {"Stage2C":"IN_PROGRESS","Allow_Stage3":"NO","Stage3_executed":False}
    if h and cv:
        artifacts_complete=bool(prep and prep["status"]=="COMPLETE" and smoke and smoke["status"]=="PASS" and warm and nll
                                and boot and figures and figures["main_case_found"] and all(x["produced"]>=x["requested"] for x in figures["coverage"].values()))
        if not artifacts_complete:decision.update(Stage2C="FAIL",Allow_Stage3="NO",incomplete_artifacts=True)
    atomic_json(ROOT / "09_reports/stage2c_decision.json",decision)
    lines=["# Stage 2C — official nuScenes trainval vehicle-only HiVT baseline", "",
           "Protocol 1: fixed-scale location warm-up → original learnable Laplace NLL. Architecture and original data definition unchanged; no Stage3, no merge into main.",
           "Official scene split, existing ped_intent environment and read-only source data; no new environment/dependency upgrade/sensor download.", "",
           "## 【Dataset】", ""]
    if split:lines += [f"train scenes={split['train_scenes']}；val scenes={split['val_scenes']}；overlap={split['overlap']}；test unused."]
    if prep:
        for name,s in prep["splits"].items():lines += [f"{name} supervised windows={s['windows']}；candidate windows={s.get('candidate_windows',s['windows'])}；empty-supervision windows={s.get('empty_supervision_windows',0)}；vehicle targets={s['full_horizon_targets']+s['partial_targets']}；full={s['full_horizon_targets']}；partial={s['partial_targets']}；context actor-windows={s['context_vehicle_actor_windows']}。"]
        lines += [f"Audit scope={prep['scope']}。Real t0 attributes define motion states; stopped-to-moving transitions remain stopped at t0."]
    lines += ["", "## 【Preprocessing】", ""]
    if prep:lines += [f"scene shards={prep['scene_shards']}；failed shards={len(prep['failed_shards'])}；NaN={prep['NaN']}；Inf={prep['Inf']}；resumed={prep['resumed_shards']}。"]
    lines += [f"PREPROCESS_SMOKE={smoke['status'] if smoke else 'NOT_RUN'}。One scene per atomic shard; stream minimal trajectory metadata; at most two scene shards in loader memory.",
              "5/12 frames, K6, HiVT64, heads8, temporal4/global3, radius50m/resolution2m; vehicle.* excluding bicycle, parked/stopped retained, ego context only. Existing numerical adapter and core modules are reused.", "",
              "## 【Training】", ""]
    if plan:lines += [f"batch size={plan['batch_size']}；steps/epoch={plan['steps_per_epoch']}。Largest stable batch below90% reserved VRAM on high-context train windows; benchmark models discarded."]
    if warm:lines += [f"warmup steps={warm['phase_steps']}；best overall FDE={warm['best_overall_FDE']:.6f}；early stop={warm['early_stop']}。"]
    if nll:lines += [f"NLL steps={nll['phase_steps']}；warm-up source step={nll['warmup_source_step']}；executed warm-up steps={nll['warmup_executed_steps']}；early stop={nll['early_stop']}。"]
    if h:lines += [f"best checkpoint global step={h['checkpoint_metadata']['global_step']}；NLL phase step={h['checkpoint_metadata']['phase_step']}；SHA256={h['checkpoint_sha256']}。"]
    lines += ["Warm-up LR=.001, min2500/max10000 optimizer steps; NLL LR=.0001, max10000; validation every500, no scheduler. Predeclared patience5 after warm-up minimum and during NLL. Best overall VAL full-horizon FDE selects primary; moving checkpoints are diagnostic only. AdamW/RNG state is restored from best warm-up overall checkpoint.",
              "Cumulative executed steps and selected warm-up source step are separately recorded because returning to the best checkpoint rewinds weights and optimizer state. Final selection uses post-update original-NLL checkpoints; no train/moving/test selection.", ""]
    if warm:lines += ["![Warm-up curve](../03_training/stage2c_warmup_curve.png)", ""]
    if nll:lines += ["![NLL curve](../03_training/stage2c_nll_curve.png)", ""]
    lines += ["## 【Validation Full Horizon】", "",
              "Main results require all12 future observations; partial futures are separate. Actor-windows are equally weighted; overlapping windows are correlated. minADE6 uses lowest-FDE mode as upstream HiVT; independent minimum-ADE is separately saved. MR6 means endpoint error>2m. CV uses one trajectory; shared table headers follow the requested metric names.", ""]
    if h and cv:
        lines += ["| Method | Group | Count | minADE6 | minFDE6 | MR6 |", "|---|---|---:|---:|---:|---:|"]
        for method,result in (("CV",cv),("HiVT",h)):
            for group,v in result["metrics"]["full_horizon"].items():lines.append(f"| {method} | {group} | {v['count']} | {fmt(v['minADE6'])} | {fmt(v['minFDE6'])} | {fmt(v['MR6'])} |")
        lines += ["",f"Paired actor identities/horizons/t0 motion states verified row by row; total full+partial paired actors={h['paired_CV_actor_rows']}。",
                  "Both methods use exactly the same VAL masks; no split changes or actor removal to obtain a CV win.", "", "Partial future:", "",
                  "| Method | Group | Count | ADE | FDE | MR |", "|---|---|---:|---:|---:|---:|"]
        for method,result in (("CV",cv),("HiVT",h)):
            for group,v in result["metrics"]["partial_future"].items():lines.append(f"| {method} | {group} | {v['count']} | {fmt(v['minADE6'])} | {fmt(v['minFDE6'])} | {fmt(v['MR6'])} |")
    else:lines += ["NOT_RUN"]
    lines += ["", "## 【Bootstrap】", ""]
    if boot:
        lines += [f"1000 paired scene-cluster resamples from {boot['scene_count']} VAL scenes; seed2022; percentile95% CI. Entire scenes, including correlated windows/actors, are repeated together. Pooled actor-window means are retained. Δ=HiVT−CV; negative favors HiVT.", "",
                  "| Group | Metric | Δ (m) | 95% CI (m) |", "|---|---|---:|---|"]
        for group,values in boot["groups"].items():
            for metric,v in values.items():
                if metric.startswith("delta_"):lines.append(f"| {group} | {metric} | {v['estimate_m']:.6f} | [{v['CI95_m'][0]:.6f}, {v['CI95_m'][1]:.6f}] |")
    else:lines += ["NOT_RUN"]
    lines += ["", "## 【Artifacts】", "",f"root directory={ROOT}",
              f"PNG count={len(list((ROOT/'05_figures').glob('stage2c_*.png')))}；table count={len(list((ROOT/'06_tables').glob('stage2c_*.csv')))}。",
              "[Artifact manifest](../00_manifest/stage2c_artifact_manifest.json) records path/size/SHA256/tracked status; checkpoint manifest also records epoch/step/selection metric. Processed shards, checkpoints, SQLite cache and large actor CSVs are local and ignored by Git."]
    if figures:
        lines += [f"Case coverage={figures['coverage']}；turning main case found={figures['main_case_found']}。",
                  "Each case uses a distinct instance within its category; moving success ADE/FDE≤2m, moving failure FDE>2m, both GT travel≥5m. Figures show lanes/history/GT/6 modes/best-FDE/probabilities and required identities/state/errors in titles."]
        if figures["main_case_found"]:
            source=read_json(ROOT / "04_evaluation/stage2c_qualitative_main_case.json")
            lines += [f"Main case turn={source['turn_degrees']:.2f}°；HiVT ADE/FDE={source['minADE6']:.6f}/{source['minFDE6']:.6f}；CV={source['CV_ADE_m']:.6f}/{source['CV_FDE_m']:.6f}。",
                      "[Main case source arrays](../04_evaluation/stage2c_qualitative_main_case.json), PNG600dpi + editable SVG/PDF. One selected turning case illustrates a local gain; aggregate results and scene bootstrap determine the broader claim.",
                      "![Qualitative main case](../05_figures/stage2c_qualitative_main_case.png)"]
    lines += ["", "## 【Git】", "",f"branch={git('branch','--show-current')}；report generation commit={git('rev-parse','HEAD')}。",
              "Push verification is recorded separately after final upload; no merge main.", "", "## 【Decision】", "",
              f"Stage2C={decision['Stage2C']}；Allow Stage3={decision['Allow_Stage3']}；Stage3 executed=False。"]
    if "case" in decision:lines += [f"Research case={decision['case']}。{decision['reason']}",f"Predeclared reasonable-generalization guard passed={decision['reasonable_generalization']}；HiVT/CV ratios={decision['HiVT_to_CV_ratios']}。",
                                   "Point-estimate A/B/C labels do not assert statistical significance; bootstrap intervals are reported above. No additional architecture/data/hyperparameter search was performed to obtain the chosen result."]
    lines += ["", "Full trainval and mini have different scene distributions; this report does not treat their unpaired scores as a controlled before/after comparison.",
              "Commands and experiment contract: [stage2c_execution_commands.md](stage2c_execution_commands.md)."]
    (ROOT / "09_reports/stage2c_final_report.md").write_text("\n".join(lines)+"\n")
    verify_previous();update_manifest();print(decision,flush=True)


if __name__=="__main__":main()
