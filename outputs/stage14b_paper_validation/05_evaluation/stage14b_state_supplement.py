"""Offline state supplement using frozen actor metrics, with no model import.

OtherVehicleState was declared in the frozen plan but was not an original
Stage11B groups() key. This script adds descriptive outputs only. It leaves
the original groups, metrics table, primary comparisons and bootstrap intact.
"""

from __future__ import annotations

import csv
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
HISTORICAL = PROJECT / "outputs/stage14a_paper_graph_ablation"
MODELS = ("NG-A", "NG-C", "G-A", "G-C", "Matched-NG-C")
COMPARISONS = (("G-C", "NG-C"), ("G-C", "Matched-NG-C"), ("NG-C", "NG-A"))
STATES = ("vehicle.moving", "vehicle.stopped", "vehicle.parked")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text())


def read_csv(path: Path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    dest = ROOT / "05_evaluation"
    out_csv = dest / "stage14b_state_supplement.csv"
    out_json = dest / "stage14b_state_supplement.json"
    assert not out_csv.exists() and not out_json.exists(), "new supplement only"

    gate_path = ROOT / "04_checkpoints/stage14b_all_frozen.json"
    gate = read_json(gate_path)
    assert gate["Status"] == "FROZEN_ALL_COMPLETE"
    assert gate["OuterTestEvaluationPermitted"] and not gate["FurtherTrainingPermitted"]
    expected = {
        f"outputs/stage14b_paper_validation/04_checkpoints/fold{k}/Matched-NG-C_best.pt"
        for k in (1, 2, 3)
    }
    assert len(gate["Checkpoints"]) == 3
    assert {row["Path"] for row in gate["Checkpoints"]} == expected
    for row in gate["Checkpoints"]:
        assert sha256(PROJECT / row["Path"]) == row["SHA256"]

    plan_path = ROOT / "stage14b_preregistered_plan.md"
    plan_registration = ROOT / "00_protocol/stage14b_plan_registration.json"
    assert sha256(plan_path) == read_json(plan_registration)["PlanSHA256"]
    assert "OtherVehicleState" in plan_path.read_text()
    old_audit_path = HISTORICAL / "05_evaluation/stage14a_identity_audit.json"
    new_audit_path = dest / "stage14b_identity_audit.json"
    bootstrap_audit_path = HISTORICAL / "06_bootstrap/stage14a_bootstrap_audit.json"
    old_audit, new_audit = read_json(old_audit_path), read_json(new_audit_path)
    bootstrap_audit = read_json(bootstrap_audit_path)
    assert old_audit["Status"] == new_audit["Status"] == bootstrap_audit["Status"] == "PASS"
    assert old_audit["Models"] == list(MODELS[:4])
    assert new_audit["Models"] == list(MODELS)
    fields = old_audit["Fields"]
    assert fields == new_audit["Fields"] and len(fields) == 15

    actor_path = dest / "cache/stage14b_actor_records.csv"
    old_actor_path = HISTORICAL / "05_evaluation/cache/stage14a_oof_actor_records.csv"
    old_metric_path = HISTORICAL / "05_evaluation/cache/stage14a_oof_metrics.npy"
    new_metric_path = dest / "cache/stage14b_matched_metrics.npy"
    weight_path = HISTORICAL / "06_bootstrap/cache/stage14a_scene_bootstrap_weights.npy"
    table_path = dest / "stage14b_capacity_metrics.csv"
    ci_path = ROOT / "06_statistics/stage14b_bootstrap_ci.csv"
    fold_ci_path = ROOT / "06_statistics/stage14b_fold_comparisons.csv"
    source_paths = [
        gate_path, plan_path, plan_registration, old_audit_path, new_audit_path,
        bootstrap_audit_path, actor_path, old_actor_path, old_metric_path,
        new_metric_path, weight_path, table_path, ci_path, fold_ci_path,
    ]
    source_hashes = {str(p.relative_to(PROJECT)): sha256(p) for p in source_paths}
    assert source_hashes[str(actor_path.relative_to(PROJECT))] == new_audit["cache_files"][actor_path.name]
    assert source_hashes[str(old_actor_path.relative_to(PROJECT))] == old_audit["cache_files"][old_actor_path.name]
    assert source_hashes[str(actor_path.relative_to(PROJECT))] == source_hashes[str(old_actor_path.relative_to(PROJECT))]
    assert source_hashes[str(old_metric_path.relative_to(PROJECT))] == old_audit["cache_files"][old_metric_path.name]
    assert source_hashes[str(new_metric_path.relative_to(PROJECT))] == new_audit["cache_files"][new_metric_path.name]
    assert source_hashes[str(weight_path.relative_to(PROJECT))] == bootstrap_audit["ReplicateWeightsSHA256"]

    actors = pd.read_csv(actor_path, dtype={"future_mask_bits": str})
    keys = ["scene_token", "sample_token", "instance_token"]
    assert len(actors) == 260151 and not actors.duplicated(keys).any()
    assert (actors.Partition == "OuterTest").all()
    assert actors.scene_token.nunique() == 630
    assert actors.groupby("scene_token").Fold.nunique().eq(1).all()
    old = np.load(old_metric_path, mmap_mode="r")
    new = np.load(new_metric_path, mmap_mode="r")
    assert old.shape == (len(actors), 4, 15) and new.shape == (len(actors), 15)

    vehicle = actors.agent_type.eq("Vehicle").to_numpy()
    named = actors.motion_state.isin(STATES).to_numpy()
    other = vehicle & ~named
    assert int(vehicle.sum()) == 191026 and int(other.sum()) == 4315
    state_counts = {
        state: int((vehicle & actors.motion_state.eq(state).to_numpy()).sum())
        for state in STATES
    }
    assert state_counts == {
        "vehicle.moving": 41728, "vehicle.stopped": 23044, "vehicle.parked": 121939,
    }
    assert sum(state_counts.values()) + int(other.sum()) == int(vehicle.sum())

    values = np.concatenate((np.asarray(old[other]), np.asarray(new[other])[:, None, :]), axis=1)
    assert values.shape == (4315, 5, 15) and np.isfinite(values).all()
    for index in (12, 13, 14):
        assert all(np.array_equal(values[:, 0, index], values[:, axis, index]) for axis in range(1, 5))
    metrics = [
        dict(Group="OtherVehicleState", Model=model, Count=4315,
             **dict(zip(fields, map(float, values[:, axis, :].mean(axis=0)))))
        for axis, model in enumerate(MODELS)
    ]

    scene_order = bootstrap_audit["SceneOrder"]
    assert len(scene_order) == len(set(scene_order)) == 630
    assert bootstrap_audit["Replicates"] == 2000 and bootstrap_audit["Seed"] == 2022
    expected_scene_order = []
    for fold in (1, 2, 3):
        split_path = PROJECT / f"outputs/stage11b_error_aware_ranking/02_splits/stage11b_fold{fold}_split.json"
        splits = read_json(split_path)
        expected_scene_order.extend(sorted(splits["OuterTest"]))
        assert set(actors.loc[actors.Fold.eq(fold), "scene_token"]) == set(splits["OuterTest"])
        source_hashes[str(split_path.relative_to(PROJECT))] = sha256(split_path)
    assert scene_order == expected_scene_order
    weights = np.load(weight_path, mmap_mode="r")
    assert weights.shape == (2000, 630)
    rng = np.random.default_rng(2022)
    for replicate in range(2000):
        for fold in range(3):
            expected_weights = np.bincount(rng.integers(0, 210, 210), minlength=210)
            assert np.array_equal(weights[replicate, fold * 210:(fold + 1) * 210], expected_weights)

    lookup = {token: index for index, token in enumerate(scene_order)}
    codes = actors.loc[other, "scene_token"].map(lookup).to_numpy()
    denominator = np.bincount(codes, minlength=630).astype(np.float64)
    bootstrap_denominator = weights @ denominator
    assert (bootstrap_denominator > 0).all()
    fde = values[:, :, fields.index("Top1FDE")]
    sums = np.stack([
        np.bincount(codes, weights=fde[:, axis], minlength=630)
        for axis in range(5)
    ], axis=-1)
    bootstrap_means = weights @ sums / bootstrap_denominator[:, None]
    selected_folds = actors.loc[other, "Fold"].to_numpy()
    comparisons = []
    for first, second in COMPARISONS:
        left, right = MODELS.index(first), MODELS.index(second)
        delta = fde[:, left] - fde[:, right]
        ci = np.quantile(bootstrap_means[:, left] - bootstrap_means[:, right], [.025, .975])
        row = dict(
            Group="OtherVehicleState", Comparison=first + "-" + second,
            Count=4315, Scenes=int((denominator > 0).sum()),
            DeltaTop1FDE=float(delta.mean()), CI95Lower=float(ci[0]), CI95Upper=float(ci[1]),
            Exploratory=True, PrimaryFamilyMember=False,
        )
        for fold in (1, 2, 3):
            selected = selected_folds == fold
            assert selected.any()
            row[f"Fold{fold}Count"] = int(selected.sum())
            row[f"Fold{fold}DeltaTop1FDE"] = float(delta[selected].mean())
        row["NegativeFolds"] = sum(row[f"Fold{fold}DeltaTop1FDE"] < 0 for fold in (1, 2, 3))
        comparisons.append(row)

    existing_metrics = read_csv(table_path)
    existing_cis = read_csv(ci_path)
    transparent_summary = []
    for group in ("StoppedVehicle", "ParkedVehicle"):
        model_rows = [row for row in existing_metrics if row["Group"] == group]
        assert len(model_rows) == 5
        group_cis = [row for row in existing_cis if row["Group"] == group]
        assert len(group_cis) == 3
        typed = []
        for row in group_cis:
            typed.append({
                "Comparison": row["Comparison"], "Count": int(row["Count"]),
                "Scenes": int(row["Scenes"]),
                **{key: float(row[key]) for key in [
                    "DeltaTop1FDE", "CI95Lower", "CI95Upper",
                    "Fold1DeltaTop1FDE", "Fold2DeltaTop1FDE", "Fold3DeltaTop1FDE",
                ]},
                "NegativeFolds": int(row["NegativeFolds"]),
                "Exploratory": True, "Source": str(ci_path.relative_to(PROJECT)),
            })
        transparent_summary.append(dict(
            Group=group, Count=int(model_rows[0]["Count"]),
            ModelMetrics=[dict(Model=row["Model"], **{key: float(row[key]) for key in [
                "minFDE6", "Top1ADE", "Top1FDE", "HitRate",
            ]}) for row in model_rows],
            Comparisons=typed,
        ))

    for path in source_paths:
        assert sha256(path) == source_hashes[str(path.relative_to(PROJECT))], "source changed during aggregation"
    source_hashes[str(Path(__file__).relative_to(PROJECT))] = sha256(Path(__file__))
    with out_csv.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Group", "Model", "Count", *fields])
        writer.writeheader()
        for row in metrics:
            writer.writerow({key: format(value, ".17g") if isinstance(value, float) else value
                             for key, value in row.items()})
    report = dict(
        Status="PASS", CheckedUTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        Scope="offline descriptive supplement from frozen actor metric arrays; no model forward or fitting",
        GroupDefinition="t0 Vehicle whose motion_state is not vehicle.moving, vehicle.stopped or vehicle.parked",
        GroupRegisteredBeforeFitting=True,
        GroupDefinitionImplementation="explicit complement; original Stage11B groups and main tables unchanged",
        Count=4315, Scenes=int((denominator > 0).sum()),
        PerFoldCount={str(fold): int((selected_folds == fold).sum()) for fold in (1, 2, 3)},
        OtherMotionStateCounts={str(key): int(value) for key, value in actors.loc[other, "motion_state"].value_counts(dropna=False).items()},
        FullVehiclePartition=dict(Vehicle=191026, MovingVehicle=41728, StoppedVehicle=23044,
                                 ParkedVehicle=121939, OtherVehicleState=4315),
        Models=list(MODELS), Fields=fields, ModelMetrics=metrics,
        ExploratoryComparisons=comparisons, ExistingStoppedParkedTransparency=transparent_summary,
        Bootstrap=dict(Replicates=2000, Seed=2022, Unit="paired whole scene within each outer fold",
                       ScenesDrawnPerFold=210, Weights="exact byte-frozen Stage14A weights; seed replay verified",
                       CoveragePercent=95, Percentiles=[2.5, 97.5],
                       MainFamilyModified=False, InferentialUse="exploratory only; no additional primary support decision"),
        AllOtherOracleMetricsBitwiseIdentical=True, InputSourceSHA256=source_hashes,
        OutputMetricsCSV=out_csv.name, OutputMetricsCSVSHA256=sha256(out_csv),
        OptimizerUpdates=0, ModelInferenceRuns=0, NewCheckpoints=0,
        OriginalGroupsModified=False, MainMetricTableModified=False, MainBootstrapModified=False,
    )
    with out_json.open("x") as handle:
        handle.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print("STAGE14B_STATE_SUPPLEMENT_PASS", report["Count"], report["PerFoldCount"], flush=True)
    for row in comparisons:
        print(row["Comparison"], row["DeltaTop1FDE"], row["CI95Lower"], row["CI95Upper"], flush=True)


if __name__ == "__main__":
    main()
