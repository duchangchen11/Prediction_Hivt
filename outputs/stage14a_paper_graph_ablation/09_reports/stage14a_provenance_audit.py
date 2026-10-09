"""Read-only historical provenance audit; no inference, tensor loading or training.

Run with the existing ped_intent Python and PYTHONDONTWRITEBYTECODE=1.
Only Stage14A/09_reports receives new small metadata reports. Official VAL
metadata and historical reports are inspected, never VAL prediction arrays.
"""
import ast
from collections import Counter, OrderedDict
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import sqlite3
import sys
from types import SimpleNamespace

OUT = Path(__file__).resolve().parent
PROJECT = OUT.parents[2]
S3 = PROJECT / "outputs/stage3_multitype_hivt"
S5 = PROJECT / "outputs/stage5a_motion_aware_decoder"
S6 = PROJECT / "outputs/stage6a_future_interaction_reliability"
S11 = PROJECT / "outputs/stage11b_error_aware_ranking"
EXPECTED_CANDIDATE_SHA = "88fe3feb59b7e830ec1e40aa917484386e0d2799d0f6116f410adda2d97fdce7"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(path.read_text())


def rows(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def save(name, value):
    path = OUT / name
    assert path.resolve().parent == OUT.resolve()
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def table(name, values):
    path = OUT / name
    assert path.resolve().parent == OUT.resolve()
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(values[0]))
        w.writeheader()
        w.writerows(values)


def main():
    sources = {}

    def source(path):
        path = Path(path)
        sources[str(path.relative_to(PROJECT))] = sha(path)
        return path

    common = source(S3 / "00_manifest/stage3_common.py")
    source(S3 / "00_manifest/stage3b_common.py")
    source(S5 / "00_manifest/stage5a_common.py")
    train_code = source(S5 / "03_training/stage5a_train.py")
    source(S5 / "00_manifest/stage5a_config.yaml")
    split = load(source(S3 / "01_data_audit/stage3_official_split.json"))
    train_all = rows(source(S3 / "02_preprocessed/stage3_train_index.csv"))
    val_all = rows(source(S3 / "02_preprocessed/stage3_val_index.csv"))
    frozen = load(source(S5 / "00_manifest/stage5a_frozen_references.json"))
    summary = load(source(S5 / "03_training/stage5a_training_summary.json"))
    warm = load(source(S5 / "03_training/stage5a_warmup_summary.json"))
    nll = load(source(S5 / "03_training/stage5a_nll_summary.json"))
    curve = rows(source(S5 / "03_training/stage5a_training_curve.csv"))
    checkpoint = source(S5 / "07_checkpoints/stage5a_best_overall_minfde.pt")
    assert sha(checkpoint) == EXPECTED_CANDIDATE_SHA == summary["checkpoint_sha256"]
    assert summary["from_scratch"] and not summary["Stage3B_trained_weights_loaded"]
    for relative, digest in sources.items():
        if relative in frozen["files"]:
            assert digest == frozen["files"][relative], relative

    # Execute only the original pure metadata predicate and sampler method AST.
    # No historical training module, model, optimizer or Dataset is imported.
    tree = ast.parse(common.read_text())
    dataset = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SceneDataset")
    init = next(n for n in dataset.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    predicate = next(n.value for n in ast.walk(init) if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Attribute) and t.attr == "rows" for t in n.targets))
    def supervised(all_rows):
        return eval(compile(ast.Expression(predicate), str(common), "eval"),
                    {"self": SimpleNamespace(all_rows=all_rows)})
    train, val = supervised(train_all), supervised(val_all)
    assert len(train) == 16898 and len(val) == 3603
    sampler_class = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SceneSampler")
    method = next(n for n in sampler_class.body if isinstance(n, ast.FunctionDef) and n.name == "__iter__")
    ns = {"random": random}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(common), "exec"), ns)
    scene_indices = OrderedDict()
    for i, r in enumerate(train):
        scene_indices.setdefault(r["scene_token"], []).append(i)
    sampler = SimpleNamespace(seed=2022, epoch=0, dataset=SimpleNamespace(scene_indices=scene_indices))
    def exposures(steps, epoch_offset):
        used = Counter()
        seen_windows = set()
        remaining = steps
        epoch = 0
        while remaining:
            sampler.epoch = epoch + epoch_offset
            order = list(ns["__iter__"](sampler))
            assert sorted(order) == list(range(len(train)))
            take = min(remaining, math.ceil(len(order) / 16))
            for i in order[:take * 16]:
                used[train[i]["scene_token"]] += 1
                seen_windows.add(i)
            remaining -= take
            epoch += 1
        return used, seen_windows
    assert "sampler.epoch=iterator['epoch']+(0 if warm else 100000)" in train_code.read_text()
    selected_warm = summary["warmup_best_source_step"]
    selected_nll = summary["best_global_step"] - 5000
    used_warm, windows_warm = exposures(selected_warm, 0)
    used_nll, windows_nll = exposures(selected_nll, 100000)
    selected_used = used_warm + used_nll
    assert len(selected_used) == 700 and len(windows_warm | windows_nll) == 16898
    selected = next(r for r in curve if int(float(r["global_step"])) == summary["best_global_step"])
    assert int(float(selected["improved"])) == 1
    assert selected_warm == 3500 and selected_nll == 4000
    assert len(curve) == 23 and summary["final_executed_global_step"] == 11500

    names = {r["scene_token"]: r["scene_name"] for r in train_all + val_all}
    train_tokens, val_tokens = set(selected_used), {r["scene_token"] for r in val}
    assert len(val_tokens) == 150 and not train_tokens & val_tokens
    assert set(split["scenes"]["train"]) == {names[t] for t in train_tokens}
    assert set(split["scenes"]["val"]) == {names[t] for t in val_tokens}
    head = load(source(S6 / "00_manifest/stage6a_head_split.json"))
    assert set(head["HeadTrain"]) | set(head["HeadDev"]) == train_tokens
    assert not set(head["HeadTrain"]) & set(head["HeadDev"])
    folds = [load(source(S11 / f"02_splits/stage11b_fold{f}_split.json")) for f in (1, 2, 3)]
    outer = [set(f["OuterTest"]) for f in folds]
    assert set.union(*outer) == set(head["HeadTrain"])
    assert all(len(t) == 210 for t in outer)
    assert all(not outer[a] & outer[b] for a in range(3) for b in range(a))
    fold_audit = []
    for f in folds:
        fold_audit.append({"Fold": f["Fold"], "InnerTrain": len(f["InnerTrain"]),
                           "InnerDev": len(f["InnerDev"]), "OuterTest": len(f["OuterTest"]),
                           "OuterTestInSelectedPredictorTraining": len(set(f["OuterTest"]) & train_tokens),
                           "OverlapFraction": 1.0, "PredictorUntouchedOuterTest": 0})
    table("stage14a_predictor_fold_overlap.csv", fold_audit)

    shard_rows = []
    for relative in sorted({r["file_path"] for r in train}):
        path = S3 / relative
        digest = sha(path)
        assert digest == frozen["scene_shards"][relative], relative
        shard_rows.append({"Path": str(path.relative_to(PROJECT)), "Bytes": path.stat().st_size,
                           "SHA256": digest})
    table("stage14a_predictor_train_shard_manifest.csv", shard_rows)
    shard_sizes = {r["Path"]: r["Bytes"] for r in shard_rows}
    retrain_data = []
    for f in folds:
        for role in ("InnerTrain", "InnerDev", "OuterTest"):
            scene_set = set(f[role])
            subset = [r for r in train if r["scene_token"] in scene_set]
            size = sum(shard_sizes[str((S3 / p).relative_to(PROJECT))]
                       for p in {r["file_path"] for r in subset})
            retrain_data.append({"Fold": f["Fold"], "Role": role, "Scenes": len(scene_set),
                                 "SupervisedWindows": len(subset),
                                 "FullHorizonActorWindows": sum(int(r["full_horizon_target_count"]) for r in subset),
                                 "PartialActorWindows": sum(int(r["partial_target_count"]) for r in subset),
                                 "ReferencedShardBytes": size, "ReferencedShardGiB": size / (1 << 30)})
    table("stage14a_retraining_data_volume.csv", retrain_data)

    scenes = []
    for token in sorted(train_tokens | val_tokens):
        scenes.append({"SceneToken": token, "SceneName": names[token],
                       "OfficialSplit": "TRAIN" if token in train_tokens else "VAL",
                       "PredictorOptimization": token in train_tokens,
                       "PredictorCheckpointSelection": token in val_tokens,
                       "SelectedPredictorWindowOccurrences": selected_used.get(token, 0),
                       "RankingHeadPool": "HeadTrain" if token in head["HeadTrain"] else "HeadDev" if token in head["HeadDev"] else "officialVAL",
                       "OuterTestFold": next((f["Fold"] for f in folds if token in f["OuterTest"]), ""),
                       "UntouchedForEndToEndEvaluation": False})
    table("stage14a_predictor_scene_provenance.csv", scenes)

    evidence_specs = [
        (S5 / "09_reports/stage5a_final_report.md", "Complete VAL150 occurs every500 updates"),
        (S6 / "09_reports/stage6a_final_report.md", "HeadDev is held out from head optimization only"),
        (PROJECT / "outputs/stage8a_future_scene_compatibility_graph/09_reports/stage8a1_resumed_final_report.md", "选择指标只有HeadDev ranking loss"),
        (PROJECT / "outputs/stage9a_type_adaptive_future_graph/09_reports/stage9a_final_report.md", "Stage9设计受Stage8 VAL结果启发"),
        (PROJECT / "outputs/stage11a_mode_selection_audit/09_reports/stage11a_root_cause_report.md", "HeadDev 已反复用于历史 checkpoint 选择"),
        (S11 / "09_reports/stage11b_final_report.md", "假设来自Stage11A"),
    ]
    evidence = []
    for path, match in evidence_specs:
        source(path)
        found = [(i + 1, line) for i, line in enumerate(path.read_text().splitlines()) if match in line]
        assert found, (path, match)
        line_number, text = found[0]
        evidence.append({"Path": str(path.relative_to(PROJECT)), "Line": line_number, "Evidence": text})

    # The only local full corpus registered by the project is trainval850.
    db = PROJECT / "outputs/stage2c_trainval_vehicle_baseline/02_preprocessed/stage2c_metadata_cache/stage2c_trajectory_metadata.sqlite"
    with sqlite3.connect("file:" + str(db.resolve()) + "?mode=ro", uri=True) as conn:
        local_scene_tokens = {r[0] for r in conn.execute("SELECT token FROM scenes")}
    assert local_scene_tokens == train_tokens | val_tokens
    devkit = Path(sys.prefix) / "lib/python3.10/site-packages/nuscenes/utils/splits.py"
    metadata_lists = {}
    for n in ast.parse(devkit.read_text()).body:
        if isinstance(n, ast.Assign) and isinstance(n.value, ast.List):
            for target in n.targets:
                if isinstance(target, ast.Name):
                    metadata_lists[target.id] = ast.literal_eval(n.value)
    assert len(metadata_lists["test"]) == 150
    mini = set(metadata_lists["mini_train"]) | set(metadata_lists["mini_val"])
    assert mini <= set(names.values()) and len(mini) == 10

    seconds = warm["elapsed_seconds_this_invocation"] + nll["elapsed_seconds_this_invocation"]
    head_runs = rows(source(S11 / "03_training/stage11b_training_summary.csv"))
    head_seconds = sum(float(r["Seconds"]) for r in head_runs if r["Model"] in ("A", "C", "R2"))
    efficiency = load(source(S5 / "04_evaluation/stage5a_efficiency_audit.json"))
    predictor_ms = next(r["mean_inference_ms"] for r in efficiency["models"] if r["Model"] == "Stage5A")
    headtrain_windows = sum(r["SupervisedWindows"] for r in retrain_data if r["Role"] == "OuterTest")
    assert headtrain_windows == 15195
    forecast_seconds = predictor_ms / 1000 * math.ceil(headtrain_windows / 16)
    budget_scaled_seconds = seconds / 11500 * 21000
    cpu_cache_bytes = sum(r["Bytes"] for r in shard_rows)
    source(S3 / "01_data_audit/stage3_class_statistics.json")
    provenance = {
        "Status": "PASS", "Stage": "Stage14A", "ClientDate": "2026-10-10", "Timezone": "Asia/Shanghai",
        "AuditActions": "historical metadata/source-code/report reading, original pure sampler AST replay and TRAIN shard byte hashes only",
        "OptimizerUpdates": 0, "InferenceRuns": 0, "NewCheckpoints": 0,
        "OfficialVALPredictionArraysRead": False, "OfficialVALTestEvaluationExecuted": False,
        "FrozenPredictor": {"Path": str(checkpoint.relative_to(PROJECT)), "SHA256": EXPECTED_CANDIDATE_SHA,
                            "FromScratch": True, "ParameterCount": 650403, "Modes": 6,
                            "TrainingSourceCommit": summary["training_code_git_commit"]},
        "PredictorTraining": {"OfficialScenes": 700, "SceneTokens": sorted(train_tokens),
                              "SceneNames": sorted(names[t] for t in train_tokens),
                              "AllIndexWindows": len(train_all), "OptimizationEligibleWindows": len(train),
                              "ExcludedNoSupervisionWindows": len(train_all) - len(train),
                              "FullHorizonActorWindows": sum(int(r["full_horizon_target_count"]) for r in train),
                              "PartialActorWindows": sum(int(r["partial_target_count"]) for r in train),
                              "FullHorizonByType": {k: sum(int(r[k]) for r in train) for k in ("full_vehicle", "full_pedestrian", "full_bicycle")},
                              "SelectedCheckpointExposureProof": "execute original SceneDataset metadata predicate and SceneSampler.__iter__ AST, replay selected warmup3500 plus NLL4000; all16898 windows/all700 scenes appear",
                              "SelectedWarmupSourceStep": selected_warm, "SelectedNLLUpdates": selected_nll,
                              "SelectedCheckpointGlobalLabel": summary["best_global_step"],
                              "TotalHistoricalExecutedUpdates": summary["final_executed_global_step"],
                              "Caveat": "global step9000 includes nominal5000 warmup budget; selected optimizer path restores warmup3500, hence7500 actual inherited updates. No per-batch training log is claimed; exposure is reconstructed from unchanged code, original metadata and cursor rules."},
        "PredictorValidation": {"OfficialScenes": 150, "SceneTokens": sorted(val_tokens),
                                "SceneNames": sorted(names[t] for t in val_tokens), "SupervisedWindows": len(val),
                                "FullHorizonActorWindows": sum(int(r["full_horizon_target_count"]) for r in val),
                                "ValidationPassesRecordedInTrainingCurve": len(curve), "IntervalUpdates": 500,
                                "CheckpointSelection": summary["selection_rule"],
                                "EarlyStopping": "NLL patience5 using the same official VAL150; selected step9000, stopped11500"},
        "RankingOuterTestOverlap": {"UnionScenes": 630, "PredictorTrainingOverlap": 630, "OverlapFraction": 1.0,
                                    "PerFold": fold_audit, "HeadDev70PredictorTrainingOverlap": 70,
                                    "Meaning": "ranking head optimization/normalization/selection isolates its OuterTest; candidate predictor does not. Conditional frozen-candidate internal ranking OOF is valid as that stated estimand, not an independent end-to-end prediction test."},
        "DevelopmentReuse": {"OfficialVALUsedForPredictorSelection": True,
                             "OfficialVALUsedForHistoricalResearchDevelopment": True,
                             "OfficialVALInspiredLaterHypotheses": True, "HeadDev70RepeatedlyUsed": True,
                             "Stage14AUsesVALForNewEvaluation": False, "Evidence": evidence},
        "IndependentEvaluation": {"Status": "NO_VERIFIED_UNTOUCHED_LABELED_EVALUATION_SCENES",
                                  "KnownLocalRegisteredScenePool": 850, "UntouchedSceneCount": 0,
                                  "Mini10OverlapsPreviouslyUsedTrainval": True,
                                  "OfficialTestCanonicalSceneNamesCount": 150,
                                  "OfficialTestListedByDevkitButNotInRegisteredSQLite": True,
                                  "OfficialTestUsedInStage5A": summary["test_used"],
                                  "OfficialTestLabelsAndCompatibleCustomThreeTypeEvaluatorAvailable": "UNKNOWN_NOT_AUDITED",
                                  "ExternalDatasetOrUnregisteredSceneAvailability": "UNKNOWN_NOT_AUDITED",
                                  "ConclusionScope": "No clean evaluation set is verified within the registered local trainval corpus. The canonical official test name list is not proof of usable labels or of a supported test evaluation protocol; no test split was read or evaluated."},
        "RetrainingEstimate": {"ExecutionAuthorized": False, "ExecutionPerformed": False,
                               "HistoricalDevice": "NVIDIA GeForce RTX3080", "Runtime": "torch2.5.1+cu124 / CUDA12.4",
                               "MeasuredUnit": "single-device elapsed wall/GPU-reservation time including training, periodic validation, I/O and checkpoint saves; not CUDA active kernel time",
                               "ObservedWarmupSeconds": warm["elapsed_seconds_this_invocation"],
                               "ObservedNLLSeconds": nll["elapsed_seconds_this_invocation"],
                               "Observed11500UpdateRunHours": seconds / 3600,
                               "SingleRun21000UpdateBudgetLinearEstimateHours": budget_scaled_seconds / 3600,
                               "ThreePredictorRunsObservedScheduleEstimateHours": 3 * seconds / 3600,
                               "ThreePredictorRuns21000UpdateBudgetEstimateHours": 3 * budget_scaled_seconds / 3600,
                               "HistoricalA_C_R2NineHeadRunsSeconds": head_seconds,
                               "HistoricalA_C_R2NineHeadRunsHours": head_seconds / 3600,
                               "HistoricalBatch16PredictorMeanMilliseconds": predictor_ms,
                               "ThreeHeadTrain630ForecastPassesKernelOnlyEstimateSeconds": 3 * forecast_seconds,
                               "ThreePredictorsPlusNineGraphA_C_R2HeadsAndForecastAnchorHours":
                                   [(3 * seconds + head_seconds + 3 * forecast_seconds) / 3600,
                                    (3 * budget_scaled_seconds + head_seconds + 3 * forecast_seconds) / 3600],
                               "AdditionalNoGraphSixHeadRunsTime": "NOT_INCLUDED; Stage14A measured summary must supply it separately",
                               "Assumptions": ["same architecture, batch16, comparable actor/lane density and RTX3080; no additional seeds or searches",
                                               "each predictor follows historical11500 update schedule or full21000 budget; convergence on new isolated partitions is unknown",
                                               "threefold predictor retraining would fit only each fold InnerTrain378 and select on InnerDev42, never OuterTest210",
                                               "periodic dev workload differs from old VAL150; old elapsed schedule is a planning anchor, not a forecast guarantee",
                                               "candidate cache preprocessing, source restoration, full inference I/O and diagnostics are additional unmeasured wall time",
                                               "retraining with proper outer isolation fixes predictor training overlap; reused research scenes still are not pristine method-development holdout"]},
        "RetrainingDataEstimate": {"ExistingTrainSceneShards": 700, "ExistingTrainShardBytes": cpu_cache_bytes,
                                   "ExistingTrainShardGiB": cpu_cache_bytes / (1 << 30),
                                   "FullTrainingActorWindows": 430581, "FullHorizonOnlyRankingActorWindows": 290085,
                                   "PerFoldPredictorInnerTrainScenes": 378, "InnerDevScenes": 42, "OuterTestScenes": 210,
                                   "ExactFoldPartitionVolumes": retrain_data,
                                   "DataNeed": "only existing trajectory/map scene shards are required by this implementation; not camera/LiDAR raw sensor files. Same scenes may be referenced across folds without duplicating originals; new fold-specific candidate caches/checkpoints would be additional local artifacts.",
                                   "RawVolumeMounted": Path("/media/lrj/54926A1D926A0438/nuscenes-trainval").exists(),
                                   "RawSensorDatasetBytes": "NOT_MEASURED_NOT_REQUIRED_FOR_CACHED_CURRENT_IMPLEMENTATION"},
        "Unknowns": ["global absence of any unregistered external dataset cannot be established from repository records",
                     "canonical official test names do not establish available GT or custom evaluation compatibility",
                     "historical wall-time records do not measure pure GPU kernel utilization",
                     "future isolated predictor convergence and preprocessing/inference I/O cost remain unknown"],
        "SourceSHA256": sources, "InstalledDevkitSplitMetadataSHA256": sha(devkit),
        "SupplementaryTables": ["stage14a_predictor_scene_provenance.csv", "stage14a_predictor_fold_overlap.csv", "stage14a_predictor_train_shard_manifest.csv", "stage14a_retraining_data_volume.csv"],
        "NextAction": "No end-to-end retraining, no official VAL/test evaluation; await separate authorization and fresh evaluation design."}
    save("stage14a_predictor_data_provenance.json", provenance)
    print(json.dumps({"Status": "PASS", "PredictorTrainScenes": 700, "OuterTestOverlap": 630,
                      "FoldOverlaps": [r["OuterTestInSelectedPredictorTraining"] for r in fold_audit],
                      "UntouchedRegisteredScenes": 0, "RetrainOneHours": seconds / 3600,
                      "RetrainThreeHours": 3 * seconds / 3600}, ensure_ascii=False))


if __name__ == "__main__":
    main()
