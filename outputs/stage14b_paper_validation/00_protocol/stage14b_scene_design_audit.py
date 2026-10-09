"""Design-only scene accounting and historical resource estimates.

This script reads existing metadata/records and writes Stage14B design files.
It never imports model code, runs prediction, creates a checkpoint, or fits a
normalizer. All training plans here require a future explicit authorization.
"""
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
S3 = PROJECT / "outputs/stage3_multitype_hivt"
S6 = PROJECT / "outputs/stage6a_future_interaction_reliability"
S11 = PROJECT / "outputs/stage11b_error_aware_ranking"
S14 = PROJECT / "outputs/stage14a_paper_graph_ablation"
BASE = "4daa4ae82557270e4f43881fa22c21e3ba6c4068"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def csv_rows(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def save(name, value):
    path = ROOT / "00_protocol" / name
    assert path.resolve().parent == (ROOT / "00_protocol").resolve()
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def table(name, rows):
    path = ROOT / "00_protocol" / name
    assert path.resolve().parent == (ROOT / "00_protocol").resolve()
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    sources = {}
    def source(path):
        sources[str(path.relative_to(PROJECT))] = sha(path)
        return path
    provenance = load(source(S14 / "09_reports/stage14a_predictor_data_provenance.json"))
    official = load(source(S3 / "01_data_audit/stage3_official_split.json"))
    head = load(source(S6 / "00_manifest/stage6a_head_split.json"))
    index = csv_rows(source(S3 / "02_preprocessed/stage3_train_index.csv"))
    train = [r for r in index if int(r["full_horizon_target_count"]) + int(r["partial_target_count"]) > 0]
    scene_names = {r["scene_token"]: r["scene_name"] for r in index}
    all700 = set(scene_names)
    head630, quarantined70 = set(head["HeadTrain"]), set(head["HeadDev"])
    assert len(all700) == 700 and len(head630) == 630 and len(quarantined70) == 70
    assert not head630 & quarantined70 and head630 | quarantined70 == all700
    assert set(provenance["PredictorTraining"]["SceneTokens"]) == all700
    assert {scene_names[t] for t in all700} == set(official["scenes"]["train"])
    assert official["train_scenes"] == 700 and official["val_scenes"] == 150 and official["overlap"] == 0
    assert provenance["IndependentEvaluation"]["UntouchedSceneCount"] == 0
    folds = [load(source(S11 / f"02_splits/stage11b_fold{f}_split.json")) for f in (1, 2, 3)]
    # Independently reconstruct the registered original outer/inner split rules.
    outer_sorted = sorted(head630, key=lambda t: (hashlib.sha256(f"2022|outer|{t}".encode()).hexdigest(), t))
    for f in folds:
        fold = f["Fold"]
        assert f["OuterTest"] == outer_sorted[(fold - 1) * 210:fold * 210]
        inner_sorted = sorted(head630 - set(f["OuterTest"]),
            key=lambda t: (hashlib.sha256(f"2022|inner|fold{fold}|{t}".encode()).hexdigest(), t))
        assert f["InnerDev"] == inner_sorted[:42] and f["InnerTrain"] == inner_sorted[42:]
        tr, dv, te = [set(f[k]) for k in ("InnerTrain", "InnerDev", "OuterTest")]
        assert (len(tr), len(dv), len(te)) == (378, 42, 210)
        assert not tr & dv and not tr & te and not dv & te
        assert tr | dv | te == head630 and not (tr | dv | te) & quarantined70
        assert len(tr | quarantined70) == 448
    assert set.union(*(set(f["OuterTest"]) for f in folds)) == head630
    assert sum(len(set(f["OuterTest"])) for f in folds) == 630

    shard_manifest = csv_rows(source(S14 / "09_reports/stage14a_predictor_train_shard_manifest.csv"))
    size_by_path = {r["Path"]: int(r["Bytes"]) for r in shard_manifest}
    by_scene = defaultdict(list)
    for row in train:
        by_scene[row["scene_token"]].append(row)
    def volume(tokens):
        items = [r for t in tokens for r in by_scene[t]]
        paths = {str((S3 / r["file_path"]).relative_to(PROJECT)) for r in items}
        size = sum(size_by_path[p] for p in paths)
        return {"Scenes": len(tokens), "SupervisedWindows": len(items),
                "FullHorizonActorWindows": sum(int(r["full_horizon_target_count"]) for r in items),
                "PartialActorWindows": sum(int(r["partial_target_count"]) for r in items),
                "ReferencedShardBytes": size, "ReferencedShardGiB": size / (1 << 30)}

    outer_scene_rows, inner_scene_rows, volumes = [], [], []
    plans_a, plans_b = [], []
    for f in folds:
        fold = f["Fold"]
        tr, dv, te = [set(f[k]) for k in ("InnerTrain", "InnerDev", "OuterTest")]
        a = {"OuterFold": fold, "Train378": f["InnerTrain"], "Development42": f["InnerDev"],
             "Evaluation210": f["OuterTest"], "Quarantined70": sorted(quarantined70),
             "All700Accounted": True, "Alternative448Train": sorted(tr | quarantined70)}
        plans_a.append(a)
        for token in sorted(all700):
            role = "Train378" if token in tr else "Development42" if token in dv else "Evaluation210" if token in te else "Quarantined70"
            outer_scene_rows.append({"OuterFold": fold, "SceneToken": token, "SceneName": scene_names[token],
                "OriginalPool": "HeadTrain630" if token in head630 else "HeadDev70",
                "RecommendedA_Role": role, "Alternative448_Role": "Train448" if token in tr | quarantined70 else role,
                "HistoricalFrozenPredictorTrainedOnScene": True,
                "ProposedEvaluationUntouchedByCorrespondingPredictorFit": token in te,
                "PristineMethodDevelopmentHoldout": False})
        for label, tokens in [("A_PredictorTrain378", tr), ("A_Development42", dv), ("A_Evaluation210", te),
                              ("Quarantined70", quarantined70), ("AlternativeA_PredictorTrain448", tr | quarantined70)]:
            volumes.append({"OuterFold": fold, "Role": label, **volume(tokens)})
        inner_order = sorted(tr, key=lambda t: (hashlib.sha256(f"2022|inner_oof|fold{fold}|{t}".encode()).hexdigest(), t))
        inner_plans = []
        for inner in (1, 2, 3):
            held = set(inner_order[(inner - 1) * 126:inner * 126])
            fit = tr - held
            assert len(held) == 126 and len(fit) == 252 and not held & fit
            assert not (fit | held) & (dv | te | quarantined70)
            inner_plans.append({"InnerFold": inner, "PredictorTrain252": sorted(fit),
                "OOFOutput126": sorted(held), "DevelopmentSelection42": f["InnerDev"],
                "OuterEvaluationNeverRead": f["OuterTest"], "Quarantined70": sorted(quarantined70)})
            for token in sorted(held):
                inner_scene_rows.append({"OuterFold": fold, "InnerOOFHoldoutFold": inner,
                    "SceneToken": token, "SceneName": scene_names[token],
                    "PredictorTrainCountForItsOOFOutput": 252,
                    "LabelsExcludedFromThisPredictorOptimizationAndSelection": True,
                    "UsedByFinalOuterPredictorTrain378": True})
            volumes.append({"OuterFold": fold, "Role": f"B_Inner{inner}_PredictorTrain252", **volume(fit)})
            volumes.append({"OuterFold": fold, "Role": f"B_Inner{inner}_OOFOutput126", **volume(held)})
        assert set.union(*(set(p["OOFOutput126"]) for p in inner_plans)) == tr
        assert sum(len(p["OOFOutput126"]) for p in inner_plans) == 378
        plans_b.append({"OuterFold": fold, "InnerOOF": inner_plans,
                        "FinalPredictorTrain378": f["InnerTrain"], "FinalPredictorDevelopment42": f["InnerDev"],
                        "FinalPredictorOutputForHeadSelection42": f["InnerDev"],
                        "FinalPredictorOutputForEvaluation210": f["OuterTest"], "Quarantined70": sorted(quarantined70)})
    table("stage14b_scene_design_outer.csv", outer_scene_rows)
    table("stage14b_scene_design_inner_oof.csv", inner_scene_rows)
    table("stage14b_scene_design_data_volume.csv", volumes)
    assert len(outer_scene_rows) == 2100 and len(inner_scene_rows) == 1134
    old_cost = provenance["RetrainingEstimate"]
    old_freeze = load(source(S14 / "00_protocol/stage14a_frozen_history.json"))
    cache_paths = [PROJECT / "outputs/stage8a_future_scene_compatibility_graph/01_training/cache" / name
                   for name in ("arg0.npy", "arg1.npy", "arg2.npy", "arg6.npy", "fde.npy", "ade.npy")]
    cache_paths.append(PROJECT / "outputs/stage11a_mode_selection_audit/01_identity_audit/cache/candidates.npy")
    cache_stats = [{"Path": str(p.relative_to(PROJECT)), "Bytes": p.stat().st_size,
                    "SHA256FromHistoricalFreeze": old_freeze["data_files"][str(p.relative_to(PROJECT))],
                    "ReadScope": "filesystem byte size only; array contents not loaded"} for p in cache_paths]
    cache_bytes = sum(r["Bytes"] for r in cache_stats)
    modeled_target_rows = volume(head630)["FullHorizonActorWindows"] * 3
    projected_merged_cache_bytes = cache_bytes * modeled_target_rows / provenance["PredictorTraining"]["FullHorizonActorWindows"]
    one_observed = old_cost["Observed11500UpdateRunHours"]
    one_cap = old_cost["SingleRun21000UpdateBudgetLinearEstimateHours"]
    old_head_seconds = old_cost["HistoricalA_C_R2NineHeadRunsSeconds"]
    ng_runs = csv_rows(source(S14 / "03_training/stage14a_training_summary.csv"))
    assert len(ng_runs) == 6 and {r["PaperModel"] for r in ng_runs} == {"NG-A", "NG-C"}
    ng_seconds = sum(float(r["Seconds"]) for r in ng_runs)
    heads_hours = (old_head_seconds + ng_seconds) / 3600
    graph_runs = csv_rows(source(S11 / "03_training/stage11b_training_summary.csv"))
    matched_proxy_low = sum(float(r["Seconds"]) for r in ng_runs if r["Model"] == "C") / 3600
    matched_proxy_high = sum(float(r["Seconds"]) for r in graph_runs if r["Model"] == "C") / 3600
    ms = old_cost["HistoricalBatch16PredictorMeanMilliseconds"]
    final_windows = sum(volume(head630)["SupervisedWindows"] for _ in folds)
    inner_windows = sum(volume(set(f["InnerTrain"]))["SupervisedWindows"] for f in folds)
    # Kernel-only batching lower anchors: boundaries between scene subsets may
    # introduce more batches, while loading, graph features and transfer add time.
    forward_a_hours = final_windows / 16 * ms / 1000 / 3600
    forward_b_hours = (final_windows + inner_windows) / 16 * ms / 1000 / 3600
    cost_rows = []
    for name, predictors, forward in [("A_Recommended378", 3, forward_a_hours), ("B_InnerOOF378", 12, forward_b_hours)]:
        cost_rows.append({"Scheme": name, "HiVTRuns": predictors, "FreshRankingHeadRuns": 18,
            "PredictorObservedScheduleAnchorHours": predictors * one_observed,
            "PredictorFullBudgetLinearAnchorHours": predictors * one_cap,
            "FifteenRankingRunsMeasuredHistoricalAnchorHours": heads_hours,
            "AdditionalThreeMatchedRunsMeasuredHours": "NOT_YET_MEASURED",
            "AdditionalThreeMatchedRunsNG_CProxyHours": matched_proxy_low,
            "AdditionalThreeMatchedRunsG_CProxyHours": matched_proxy_high,
            "CandidateForwardKernelOnlyAnchorHours": forward,
            "KnownComponentLowerHistoricalScheduleAnchorHours": predictors * one_observed + heads_hours + forward,
            "KnownComponentUpperHistoricalBudgetAnchorHours": predictors * one_cap + heads_hours + forward,
            "All18RunsIllustrativeLowerProxyHours": predictors * one_observed + heads_hours + forward + matched_proxy_low,
            "All18RunsIllustrativeUpperProxyHours": predictors * one_cap + heads_hours + forward + matched_proxy_high,
            "Interpretation": "planning anchor, not guaranteed runtime; matched3 uses separately labeled NG-C/G-C workload proxies, not measured bounds; excludes cache I/O/preprocessing/graph construction/diagnostics/new seeds"})
    table("stage14b_scene_design_cost_comparison.csv", cost_rows)

    design = {"Status": "DESIGN_ONLY_NOT_EXECUTED", "ClientDate": "2026-10-10", "Timezone": "Asia/Shanghai",
        "BaseCommit": BASE, "Branch": "stage14b/paper-validation-design", "Recommendation": "A_Recommended378",
        "Reason": "minimum three-predictor study isolates outer evaluation from predictor fitting while preserving original630 threefold roles and explicitly quarantining70; B is a stronger optional stacking audit with4x predictor runs, not a pristine test",
        "TrainingAuthorizedByThisDesign": False,
        "Scope": "future end-to-end A/B validation designs, separate from Stage14B currently authorized3 Matched-NG-C ranking-only runs; noHiVT fit/inference performed by this audit",
        "OptimizerUpdates": 0, "InferenceRuns": 0, "NewCheckpoints": 0,
        "OfficialTrain700": sorted(all700), "HeadTrain630": sorted(head630), "QuarantinedHeadDev70": sorted(quarantined70),
        "OriginalFoldCounts": {"InnerTrain": 378, "InnerDev": 42, "OuterTest": 210},
        "A": {"HiVTRuns": 3, "Folds": plans_a, "RankingTrainPredictionSource": "corresponding final predictor applied to its ownTrain378 (in-sample predictor residuals)",
              "PredictorSelectionOnly": "Development42; never Outer210/HeadDev70/officialVAL/test",
              "BiasLimit": "head training candidates are from predictor training scenes; train-vs-heldout candidate quality can differ, but outer scenes are excluded from predictor and ranking optimization/selection"},
        "AlternativeA448": {"Status": "NOT_RECOMMENDED_NOT_AUTHORIZED", "HiVTRuns": 3,
              "TrainScenes": "original378 + explicitly disclosed historicalHeadDev70", "DevelopmentScenes": 42, "EvaluationScenes": 210,
              "Caveat": "uses all700 across roles but changes training budget/comparability and revives development-reused70; labels are not pristine. It requires a separate explicit design approval and cannot be silently substituted for A or B."},
        "B": {"HiVTRuns": 12, "Folds": plans_b, "InnerOOFSplitRule": "sort originalInnerTrain378 bySHA256(2022|inner_oof|foldN|scene_token), lexicaltoken tie; consecutive126 blocks",
              "InnerPredictorTrainingScenes": 252, "InnerPredictorOOFOutputScenes": 126,
              "InnerPredictorSelectionOnly": "fixedDevelopment42; held126/Outer210/HeadDev70 never used for fitting or model choice",
              "RankerTrainPredictionSource": "concatenate three innerOOF126 outputs to cover originalInnerTrain378 exactly once",
              "FinalPredictorRole": "fitTrain378, selectDevelopment42, forecastDevelopment42 and Outer210; not replace trainingOOF rows",
              "BiasLimit": "OOF removes same-scene base-learner training exposure for head-training candidates;252-vs378 model/data distributions differ, inner checkpoints share42selection, graph contexts must use the same source predictor per whole window, independent-mode semantics need auditing"},
        "FreshRankingHeads": {"PerOuterFold": ["NG-A", "NG-C", "G-A", "G-C", "Matched-NG-C", "R2"], "TotalRuns": 18,
              "CostScope": "original2x2 plus Matched-NG-C capacity control and correspondingfoldR2. Fifteen historical runs measured; three matched runs use separately labeled workload proxies until new measured timing is available.",
              "HistoricalG_A_C_orR2WeightsMayBeReused": False,
              "Reason": "new candidate predictor weights change trajectories, scores and graph/R2 features; all four heads must fit the same new candidate cache, using shared controlled initialization and fold-local training normalization. Bicycle routes only to the newly trained correspondingfold R2 across four models."},
        "EvaluationScope": {"OuterFoldUnionScenes": 630, "UntouchedByProposedCorrespondingTraining": True,
              "PristineResearchConfirmation": False, "HistoricalPredictorTrainingOverlap": "630/630, eachfold210/210",
              "HistoricalVALSelectionAndHypothesisDevelopment": True, "KnownRegisteredUntouchedTrainvalScenes": 0,
              "OfficialVALMayBeClaimedUntouchedAfterNewPredictorFit": False,
              "CurrentHiddenOfficialTESTUsable": False,
              "OfficialProtocolCompatibility": "PARTIAL_NEEDS_ADAPTER_AND_PROTOCOL_RESET",
              "LocalLabel": "training-isolated end-to-end internal scene CV; development-informed, not untouched official independent test"},
        "ProposedPredictorOptimization": {"SeedsByOuterFold": [2022, 2122, 2222],
              "WarmupUpdates": 5000, "WarmupLearningRate": 0.001, "NLLMaximumUpdates": 16000,
              "NLLLearningRate": 0.0001, "GlobalUpdateLimit": 21000, "BatchSize": 16,
              "SelectionIntervalUpdates": 500, "NLLPatience": 5,
              "Selection": "strictDevelopment42 full-horizonOverallminFDE6; finalcheckpointNLL only",
              "WarmupTransition": "restore ownwarmupbest model/optimizer/RNG; no global700-trained checkpoint",
              "ImplementationPrerequisite": "oldStage5A entrypoint hardcodesTRAIN700/VAL150 and assert16898/3603; future authorizedimplementation must provide explicitfold-local data/seed adapters without editing oldfiles"},
        "SourceSHA256": sources}
    save("stage14b_scene_design.json", design)
    save("stage14b_scene_design_cost_audit.json", {"Status": "DESIGN_ONLY_HISTORICAL_COST_ANCHORS",
        "MeasuredHardware": old_cost["HistoricalDevice"], "HistoricalWarmupAndNLLSeconds": [old_cost["ObservedWarmupSeconds"], old_cost["ObservedNLLSeconds"]],
        "OriginalMeasuredRunUpdates": 11500, "OriginalBudgetUpdates": 21000,
        "Historical15HeadAggregateSeconds": old_head_seconds + ng_seconds,
        "Schemes": cost_rows, "PerFoldVolumes": volumes,
        "DatasetOriginalsDuplicated": False,
        "ExistingFullTRAIN700GraphAndTargetCandidateCacheBytes": cache_bytes,
        "ExistingFullTRAIN700GraphAndTargetCandidateCacheGiB": cache_bytes / (1 << 30),
        "ExistingCacheByteStatOnlyEvidence": cache_stats,
        "EstimatedMergedThreefoldTargetCacheRows": modeled_target_rows,
        "EstimatedMergedAorBTargetCacheBytesAtUnchangedDtypeShapes": projected_merged_cache_bytes,
        "EstimatedMergedAorBTargetCacheGiBAtUnchangedDtypeShapes": projected_merged_cache_bytes / (1 << 30),
        "CacheEstimateLimits": "bothA/B can merge one630-target-row cache perouterfold; B need not retain densegraph caches for all12 predictors. Assumes historicaldense node/edge geometry layout and samefull-horizon targets. Originalscene shards are referenced once, not copied. Completeall-current-actor forecasts, provenance, checkpoints/optimizer snapshots, temporary dev/inference buffers anddisk overhead are additional unmeasured costs; duplicatepermethod caches are forbidden by sharedcandidate control.",
        "ImportantAssumptions": ["sameStage5A architecture/batch16/density/device; validation42 workload differs from historicalVAL150",
            "fixedupdate estimate is deliberately not scaled byscene count; full11500/21000 schedules can expose252scenes moreoften",
            "epoch-matched schedules would be a different unapproved protocol; convergence on reduced partitions is unknown",
            "15head costs use historical12G/NG plus3R2 runs; 3extraMatched-NG-C are not yet measured, workload proxies are not measured bounds; newcandidate quality may change earlystopping and runtime",
            "candidate-forward anchors omitbatch-boundary overhead, I/O, transfer, graph construction and cache writes",
            "neither estimate includes addedseed searches or hyperparameter sweeps"],
        "NewGPUJobsExecuted": 0, "SourceSHA256": sources})
    print(json.dumps({"Status": "DESIGN_METADATA_PASS", "Scenes": [700, 630, 70],
        "A_HiVT": 3, "B_HiVT": 12, "RankingRunsEach": 18,
        "CostAnchors": cost_rows}, ensure_ascii=False))


if __name__ == "__main__":
    main()
