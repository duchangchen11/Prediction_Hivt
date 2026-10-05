"""Stage4A exact paired VAL, frozen t0 subgroups, diagnostics and fair efficiency.

Every analysis consumes the frozen Stage3B actor identity ledger. Interaction
membership is registered before formal updates and uses only t0 positions/types.
"""
import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
STAGE3_ROOT = ROOT.parent / 'stage3_multitype_hivt'
sys.path.insert(0, str(STAGE3_ROOT / '00_manifest'))
sys.path.insert(0, str(STAGE3_ROOT / '04_evaluation'))
sys.path.insert(0, str(ROOT / '00_manifest'))
sys.path.insert(0, str(ROOT / "00_manifest"))
sys.path.insert(0, str(ROOT / "04_evaluation"))
from stage3b_common import (CLASSES, GROUPS, HORIZONS, METRICS, SceneDataset,
    atomic_json, read_json, write_csv, sha256, state_digest)
from stage3b_pairing_ledger import actor_key
import numpy as np
import torch

BASE_ACTORS = STAGE3_ROOT / "04_evaluation/stage3b_type_embedding_actor_errors.csv"
ACTORS = ROOT / "04_evaluation/stage4a_actor_errors.csv"
MEMBERSHIP = ROOT / "04_evaluation/stage4a_interaction_membership.csv"
SUBGROUP_AUDIT = ROOT / "04_evaluation/stage4a_interaction_subgroup_audit.json"
BEST = ROOT / "07_checkpoints/stage4a_best_overall_minfde.pt"
SUMMARY = ROOT / "03_type_interaction/stage4a_training_summary.json"
CONFIG = ROOT / "00_manifest/stage4a_config.yaml"
FIELDS = ("minADE6", "minFDE6", "MR6", "Top1ADE6", "Top1FDE6", "NLL")
LABELS = {"overall": "Overall", "vehicle": "Vehicle", "pedestrian": "Pedestrian", "bicycle": "Bicycle"}
INTERACTION = ("Heterogeneous-20m", "Vehicle hetero-20m", "Pedestrian hetero-20m", "VP-context-20m")
MOTION = (("vehicle", 5), ("pedestrian", 1), ("pedestrian", 5), ("bicycle", 1), ("bicycle", 5))


def read_actors(path):
    with Path(path).open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 85027 and len({actor_key(row) for row in rows}) == len(rows)
    assert sum(r["horizon"] == "full_horizon" for r in rows) == 54990
    assert sum(r["horizon"] == "partial_future" for r in rows) == 30037
    values = np.array([[float(r[k]) for k in FIELDS + ("independent_minADE6", "GT_endpoint_displacement_m")] for r in rows])
    assert np.isfinite(values).all()
    return rows


def read_membership():
    audit = read_json(SUBGROUP_AUDIT)
    assert audit["registration_status"] == "REGISTERED_BEFORE_FORMAL_TRAINING"
    assert audit["membership_sha256"] == sha256(MEMBERSHIP)
    assert audit["Stage3B_actor_errors_sha256"] == sha256(BASE_ACTORS)
    with MEMBERSHIP.open() as f:
        rows = list(csv.DictReader(f))
    mapping = {actor_key(r): r for r in rows}
    assert len(mapping) == len(rows) == 85027
    return mapping


def select(rows, group, membership=None):
    full = [r for r in rows if r["horizon"] == "full_horizon"]
    if group in INTERACTION:
        assert membership is not None
        return [r for r in full if membership[actor_key(r)][group] == "1"]
    if " >" in group:
        cls, threshold = group.split(" >")
        threshold = float(threshold.rstrip("m"))
        return [r for r in full if r["agent_type"] == cls.lower() and float(r["GT_endpoint_displacement_m"]) > threshold]
    return [r for r in full if group == "overall" or r["agent_type"] == group
            or (r["agent_type"] == "vehicle" and r["motion_state"] == group)]


def summarize(rows):
    assert rows
    values = np.array([[float(r[k]) for k in FIELDS] for r in rows])
    assert np.isfinite(values).all()
    return {"Count": len(rows), "UniqueInstances": len({r["instance_token"] for r in rows}),
            "UniqueScenes": len({r["scene_token"] for r in rows}),
            **dict(zip(FIELDS, map(float, values.mean(0))))}


def register_subgroups():
    """No future tensor or prediction is used to compute any membership flag."""
    from stage4a_common import verify_previous
    verify_previous(shards=False)
    assert not (ROOT / "07_checkpoints/stage4a_last_checkpoint.pt").exists(), "Register before formal training"
    assert not SUMMARY.exists(), "Register before formal training"
    prereg = ROOT / "00_manifest/stage4a_preregistration.json"
    assert prereg.exists(), "Root scientific preregistration must exist first"
    base = read_actors(BASE_ACTORS)
    by_window = defaultdict(list)
    for row in base:
        by_window[(row["scene_token"], row["sample_token"])].append(row)
    dataset = SceneDataset("val")
    frozen = read_json(ROOT / "00_manifest/stage4a_frozen_stage3b.json")
    shards = sorted({r["file_path"] for r in dataset.all_rows})
    assert len(shards) == 150
    expected_shards = frozen.get("scene_shards", frozen.get("scene_shard_hashes", {}))
    assert expected_shards
    for name in shards:
        assert sha256(STAGE3_ROOT / name) == expected_shards[name]
    fields = ["scene_token", "sample_token", "instance_token", "horizon", "node_in_graph", "agent_type", "agent_type_id",
              *INTERACTION, "heterogeneous_neighbor_count_20m", "VP_neighbor_count_20m",
              "nearest_heterogeneous_instance_token", "nearest_heterogeneous_agent_type", "nearest_heterogeneous_node_in_graph",
              "nearest_heterogeneous_distance_m", "nearest_VP_instance_token", "nearest_VP_agent_type",
              "nearest_VP_node_in_graph", "nearest_VP_distance_m"]
    written = []
    seen = set()
    node_observations = 0
    for index in range(len(dataset)):
        graph = dataset[index]
        assert graph.num_nodes == len(graph.instance_tokens) == len(graph.annotation_tokens) == len(graph.category_names)
        assert len(set(graph.instance_tokens)) == graph.num_nodes
        assert all(isinstance(x, str) and len(x) == 32 for x in graph.instance_tokens)
        assert torch.all((graph.agent_type >= 0) & (graph.agent_type <= 2))
        # Actor nodes originate from annotation instances. Ego tensors are separate
        # coordinate-reference fields and are never appended to these arrays.
        positions = graph.positions[:, 4].double()
        types = graph.agent_type
        current_valid = ~graph.padding_mask[:, 4]
        assert all(isinstance(graph.annotation_tokens[i][4], str) and len(graph.annotation_tokens[i][4]) == 32
                   for i in torch.where(current_valid)[0].tolist())
        node_observations += int(current_valid.sum())
        dist = torch.cdist(positions, positions)
        same = torch.eye(graph.num_nodes, dtype=torch.bool)
        heterogeneous = (types[:, None] != types[None, :]) & ~same & current_valid[:, None] & current_valid[None, :]
        vp = ((types[:, None] == 0) & (types[None, :] == 1)) | ((types[:, None] == 1) & (types[None, :] == 0))
        vp &= ~same & current_valid[:, None] & current_valid[None, :]
        near_hetero = heterogeneous & (dist <= 20.0)
        near_vp = vp & (dist <= 20.0)
        for old in by_window[(graph.scene_token, graph.sample_token)]:
            node = int(old["node_in_graph"])
            assert current_valid[node]
            assert graph.instance_tokens[node] == old["instance_token"]
            assert CLASSES[int(types[node])] == old["agent_type"]
            assert int(types[node]) == int(old["agent_type_id"])
            key = actor_key(old)
            assert key not in seen
            seen.add(key)
            h = bool(near_hetero[node].any())
            row = {k: old[k] for k in fields[:7]}
            row.update(dict(zip(INTERACTION, map(int, (h, h and int(types[node]) == 0,
                                                     h and int(types[node]) == 1, bool(near_vp[node].any()))))))
            row.update(heterogeneous_neighbor_count_20m=int(near_hetero[node].sum()),
                       VP_neighbor_count_20m=int(near_vp[node].sum()))
            for label, mask in (("heterogeneous", heterogeneous), ("VP", vp)):
                candidates = torch.where(mask[node])[0]
                if len(candidates):
                    neighbor = int(candidates[dist[node, candidates].argmin()])
                    values = (graph.instance_tokens[neighbor], CLASSES[int(types[neighbor])], neighbor, float(dist[node, neighbor]))
                else:
                    values = ("", "", "", "")
                for suffix, value in zip(("instance_token", "agent_type", "node_in_graph", "distance_m"), values):
                    row[f"nearest_{label}_{suffix}"] = value
            written.append(row)
        if (index + 1) % 500 == 0:
            print("STAGE4A_T0_SUBGROUP_REGISTRATION", index + 1, "/", len(dataset), flush=True)
    dataset.clear()
    assert seen == {actor_key(r) for r in base}
    write_csv(MEMBERSHIP, written)
    stats = {}
    for horizon in HORIZONS:
        stats[horizon] = {}
        for group in INTERACTION:
            chosen = [r for r in written if r["horizon"] == horizon and r[group] == 1]
            stats[horizon][group] = {"Count": len(chosen), "UniqueInstances": len({r["instance_token"] for r in chosen}),
                                     "UniqueScenes": len({r["scene_token"] for r in chosen})}
    audit = {"status": "PASS", "registration_status": "REGISTERED_BEFORE_FORMAL_TRAINING",
             "registered_before_formal_training": True,
             "registered_at_UTC": datetime.now(timezone.utc).isoformat(), "radius_m": 20.0,
             "membership_definition": "At t0, at least one distinct current valid actor of a different V/P/B type at Euclidean distance <=20m",
             "VP_definition": "Vehicle target with pedestrian context or pedestrian target with vehicle context at t0 <=20m",
             "membership_uses_only": ["positions[:,4]", "agent_type", "padding_mask[:,4]", "distinct node identity"],
             "future_or_prediction_used_to_define_membership": False,
             "horizon_identity_source": "Frozen Stage3B CSV only; full/partial labels are carried through, not used to define membership",
             "ego_in_graph": False, "ego_proof": "All graph nodes correspond one-to-one to annotation and instance tokens; ego_history/ego_future remain separate reference tensors",
             "VAL_scenes": 150, "VAL_supervised_windows": len(dataset), "actor_windows": len(written),
             "current_context_node_observations": node_observations, "counts": stats,
             "membership_relative_path": str(MEMBERSHIP.relative_to(ROOT)), "membership_sha256": sha256(MEMBERSHIP),
             "Stage3B_actor_errors_sha256": sha256(BASE_ACTORS), "preregistration_sha256": sha256(prereg),
             "VAL_shard_sha256": {p: expected_shards[p] for p in shards}, "test_used": False,
             "data_preprocessing": False, "formal_optimizer_updates_at_registration": 0}
    atomic_json(SUBGROUP_AUDIT, audit)
    verify_previous(shards=False)
    print("STAGE4A_T0_SUBGROUP_REGISTERED", stats, flush=True)


def strict_pairing(baseline, current):
    a = {actor_key(r): r for r in baseline}
    b = {actor_key(r): r for r in current}
    membership = read_membership()
    assert a.keys() == b.keys() == membership.keys(), "Actor mismatch; stop comparison"
    maximum_displacement_difference = 0.0
    identity = ("scene_token", "sample_token", "instance_token", "horizon", "node_in_graph", "agent_type",
                "motion_state", "valid_future_steps", "GT_trajectory_sha256", "future_mask_bits", "agent_type_id")
    for key, old in a.items():
        new = b[key]
        for name in identity:
            assert old[name] == new[name], (key, name)
        for name in ("node_in_graph", "agent_type", "agent_type_id"):
            assert new[name] == membership[key][name], (key, name)
        assert int(new["agent_type_id"]) == CLASSES.index(new["agent_type"])
        valid = int(new["valid_future_steps"])
        assert new["future_mask_bits"].count("1") == valid
        assert valid == 12 if new["horizon"] == "full_horizon" else 1 <= valid < 12
        maximum_displacement_difference = max(maximum_displacement_difference,
            abs(float(old["GT_endpoint_displacement_m"]) - float(new["GT_endpoint_displacement_m"])))
        assert float(new["MR6"]) == float(float(new["minFDE6"]) > 2)
    assert maximum_displacement_difference <= 1e-6
    audit = {"status": "PASS", "paired_full_horizon_actors": 54990, "paired_partial_future_actors": 30037,
             "paired_total_actor_windows": 85027, "scene_sample_instance_horizon_node_identity_equal": True,
             "agent_type_equal": True, "motion_state_equal": True, "future_mask_bitwise_equal": True,
             "GT_trajectory_float32_tensor_SHA256_equal": True,
             "GT_audit_source": "Stage3B actor CSV stores whole GT trajectory SHA256 and future-mask bits, copied from unchanged frozen graph tensors",
             "maximum_GT_endpoint_displacement_difference_m": maximum_displacement_difference,
             "B_actor_errors_sha256": sha256(BASE_ACTORS), "C_actor_errors_sha256": sha256(ACTORS),
             "interaction_membership_sha256": sha256(MEMBERSHIP), "VAL_scenes": 150, "test_used": False}
    atomic_json(ROOT / "04_evaluation/stage4a_pairing_audit.json", audit)
    return a, b, membership


def main_tables(baseline, current, measured, membership):
    full = measured["metrics"]["full_horizon"]
    rows = []
    for group in GROUPS:
        values = full[group]
        chosen = select(current, group, membership)
        assert len(chosen) == values["count"]
        if chosen:
            check = summarize(chosen)
            assert all(abs(check[k] - values[k]) < 1e-10 for k in FIELDS)
        rows.append({"Group": LABELS.get(group, group), "Count": values["count"], **{k: values[k] for k in FIELDS}})
    write_csv(ROOT / "06_tables/stage4a_main_results.csv", rows)
    motion = []
    for cls, threshold in MOTION:
        name = cls.capitalize() + f" >{threshold}m"
        values = summarize(select(current, name))
        motion.append({"Group": name, "AgentType": cls, "GT_endpoint_displacement_gt_m": threshold,
                       **{k: values[k] for k in ("Count", "UniqueInstances", "UniqueScenes") + FIELDS[:-1]}})
    assert [r["Count"] for r in motion] == [9744, 8810, 7581, 155, 132]
    write_csv(ROOT / "06_tables/stage4a_nontrivial_motion_metrics.csv", motion)
    atomic_json(ROOT / "04_evaluation/stage4a_nontrivial_motion_metrics.json", {
        "status": "PASS", "metrics": motion, "source_actor_csv_sha256": sha256(ACTORS),
        "count_unit": "actor-window; overlapping windows not independent", "Bicycle_gt5_significance_test": False,
        "bicycle_sparse_unique_instances_scenes_descriptive_only": True})
    subgroup_rows = []
    for group in INTERACTION:
        for model, actors in (("B_Stage3B", baseline), ("C_Stage4A", current)):
            selected = select(actors, group, membership)
            subgroup_rows.append({"Group": group, "Model": model, **summarize(selected)})
    write_csv(ROOT / "06_tables/stage4a_interaction_subgroup_metrics.csv", subgroup_rows)
    ablation = []
    groups = GROUPS + tuple(cls.capitalize() + f" >{t}m" for cls, t in MOTION) + INTERACTION
    for group in groups:
        chosen_b = select(baseline, group, membership)
        chosen_c = select(current, group, membership)
        if not chosen_b:
            assert not chosen_c
            # Preserve predefined empty motion groups without invented averages.
            ablation.append({"Group": LABELS.get(group, group), "Count": 0,
                **{k: None for k in ("B_minADE", "C_minADE", "Delta_ADE", "B_minFDE", "C_minFDE", "Delta_FDE",
                    "relative_FDE_change", "B_MR", "C_MR", "Delta_MR", "B_Top1FDE", "C_Top1FDE", "Delta_Top1FDE")}})
            continue
        old, new = summarize(chosen_b), summarize(chosen_c)
        assert old["Count"] == new["Count"]
        row = {"Group": LABELS.get(group, group), "Count": old["Count"]}
        for metric, short in (("minADE6", "ADE"), ("minFDE6", "FDE")):
            row.update({f"B_min{short}": old[metric], f"C_min{short}": new[metric], f"Delta_{short}": new[metric] - old[metric]})
        row.update(relative_FDE_change=(new["minFDE6"] - old["minFDE6"]) / old["minFDE6"],
                   B_MR=old["MR6"], C_MR=new["MR6"], Delta_MR=new["MR6"] - old["MR6"],
                   B_Top1FDE=old["Top1FDE6"], C_Top1FDE=new["Top1FDE6"], Delta_Top1FDE=new["Top1FDE6"] - old["Top1FDE6"])
        ablation.append(row)
    write_csv(ROOT / "06_tables/stage4a_type_interaction_ablation.csv", ablation)
    atomic_json(ROOT / "04_evaluation/stage4a_type_interaction_ablation.json", {
        "status": "PASS", "delta": "C_Stage4A-B_Stage3B; negative favors Stage4A", "relative_change_unit": "fraction, not percent",
        "groups": ablation, "B_actor_errors_sha256": sha256(BASE_ACTORS), "C_actor_errors_sha256": sha256(ACTORS),
        "pairing_audit_sha256": sha256(ROOT / "04_evaluation/stage4a_pairing_audit.json"),
        "subgroups_preregistered_before_formal_training": True, "membership_sha256": sha256(MEMBERSHIP)})
    return ablation


def bootstrap(baseline, current, scenes, membership):
    assert len(scenes) == 150
    scenes = sorted(scenes)
    scene_index = {token: i for i, token in enumerate(scenes)}
    rng = np.random.default_rng(2022)
    draws = rng.integers(0, 150, size=(1000, 150))
    multiplicities = np.stack([np.bincount(row, minlength=150) for row in draws])
    old = {actor_key(r): r for r in baseline}
    planned = {"overall": ("minADE6", "minFDE6", "Top1FDE6"), "vehicle": ("minADE6", "minFDE6"),
               "pedestrian": ("minADE6", "minFDE6"), "bicycle": ("minADE6", "minFDE6"),
               "vehicle.moving": ("minFDE6",), "Vehicle >5m": ("minFDE6",), "Pedestrian >5m": ("minFDE6",),
               **{group: ("minFDE6",) for group in INTERACTION}}
    result = {"status": "PASS", "method": "paired scene-cluster percentile bootstrap", "replicates": 1000, "seed": 2022,
              "resampling_unit": "official VAL scene", "scene_count": 150, "confidence": .95,
              "delta": "C_Stage4A-B_Stage3B; negative favors Stage4A",
              "weighting": "pool equally weighted actor-window deltas across complete resampled scene clusters",
              "Bicycle_gt5_significance_test": False, "Bicycle_gt5_reason": "limited unique bicycle instances/scenes; descriptive only",
              "Bicycle_overall_CI_role": "pre-registered major-class reliable-degradation guard; no strong improvement claim",
              "groups": {}, "interaction_membership_sha256": sha256(MEMBERSHIP),
              "B_actor_errors_sha256": sha256(BASE_ACTORS), "C_actor_errors_sha256": sha256(ACTORS)}
    csv_rows = []
    for group, metrics in planned.items():
        selected = select(current, group, membership)
        count = np.zeros(150)
        sums = {k: np.zeros(150) for k in metrics}
        for row in selected:
            i = scene_index[row["scene_token"]]
            count[i] += 1
            base = old[actor_key(row)]
            for metric in metrics:
                sums[metric][i] += float(row[metric]) - float(base[metric])
        denominator = multiplicities @ count
        assert np.all(denominator > 0)
        fields = {}
        for metric, values in sums.items():
            samples = (multiplicities @ values) / denominator
            lo, hi = np.quantile(samples, [.025, .975])
            point = float(values.sum() / count.sum())
            fields[metric] = {"estimate_m": point, "CI95_m": [float(lo), float(hi)],
                "bootstrap_std_m": float(samples.std(ddof=1)), "valid_replicates": 1000,
                "actor_windows": int(count.sum()), "unique_scenes_with_targets": int((count > 0).sum())}
            csv_rows.append({"Group": LABELS.get(group, group), "Metric": metric, "Delta_C_minus_B_m": point,
                "CI95_lower_m": float(lo), "CI95_upper_m": float(hi), "ActorWindows": int(count.sum()),
                "UniqueScenes": int((count > 0).sum()), "ResamplingUnit": "scene", "Replicates": 1000})
        result["groups"][group] = fields
    atomic_json(ROOT / "04_evaluation/stage4a_bootstrap_ci.json", result)
    write_csv(ROOT / "06_tables/stage4a_bootstrap_ci.csv", csv_rows)
    return result


def load_final_model():
    from stage4a_common import model_new, verify_previous
    verify_previous(shards=False)
    training = read_json(SUMMARY)
    assert training["status"] == "COMPLETE"
    assert sha256(BEST) == training["checkpoint_sha256"]
    assert sha256(CONFIG) == training["config_sha256"]
    saved = torch.load(BEST, map_location="cpu", weights_only=False)
    assert saved["metadata"]["phase"] == "original_nll"
    model = model_new()
    model.load_state_dict(saved["state_dict"])
    assert state_digest(model.state_dict()) == training["model_state_content_sha256"]
    assert all(torch.isfinite(p).all() for p in model.parameters())
    model.eval()
    return model, saved, training


@torch.no_grad()
def fresh_val():
    from stage4a_common import evaluate, verify_previous
    read_membership()
    model, saved, training = load_final_model()
    measured = evaluate(SceneDataset("val"), model, actor_path=ACTORS, progress=True)
    assert measured["windows"] == 3603 and len(measured["scenes"]) == 150
    actual = measured["metrics"]["full_horizon"]
    selected = training["selected_full_horizon_metrics"]
    assert all(actual[g]["count"] == selected[g]["count"] for g in GROUPS)
    reconciliation = []
    for group in GROUPS:
        for metric in FIELDS:
            if actual[group]["count"]:
                # Argmax is discontinuous: near-tied probabilities can change
                # Top1 under nondeterministic GPU scatter. Selection metrics
                # retain their original strict tolerance. Fresh values remain
                # authoritative; neither checkpoint nor scientific rule changes.
                tolerance = 1e-4 if metric in ("Top1ADE6", "Top1FDE6") else 1e-6
                difference = abs(actual[group][metric] - selected[group][metric])
                reconciliation.append({"group": group, "metric": metric,
                    "training_best_value": selected[group][metric], "fresh_value": actual[group][metric],
                    "absolute_difference": difference, "tolerance": tolerance,
                    "passed": difference < tolerance})
    atomic_json(ROOT / "04_evaluation/stage4a_fresh_val_reconciliation.json", {
        "status": "PASS" if all(r["passed"] for r in reconciliation) else "FAIL",
        "checkpoint_sha256": sha256(BEST), "checks": reconciliation,
        "selection_metrics_tolerance_unchanged": 1e-6, "Top1_reporting_tolerance": 1e-4,
        "fresh_metrics_authoritative": True, "checkpoint_reselected": False,
        "reason": "Original GPU protocol is nondeterministic; argmax can switch at near-tied modes. Top1 reporting reconciliation is separate from best-FDE checkpoint selection and scientific inference."})
    assert all(r["passed"] for r in reconciliation), "Fresh/training reconciliation failed; see audit"
    read_actors(ACTORS)
    measured.update(status="PASS", checkpoint_sha256=sha256(BEST), checkpoint_relative_path=str(BEST.relative_to(ROOT)),
        checkpoint_metadata=saved["metadata"], actor_errors_sha256=sha256(ACTORS), NaN=0, Inf=0,
        fresh_complete_official_VAL=True, full_horizon_paired_actor_windows=54990,
        type_conditioned_interaction=True)
    atomic_json(ROOT / "04_evaluation/stage4a_metrics.json", measured)
    verify_previous(shards=False)
    print("STAGE4A_FRESH_EVALUATION=PASS", actual["overall"], flush=True)


def analyze():
    from stage4a_common import verify_previous
    verify_previous(shards=False)
    measured = read_json(ROOT / "04_evaluation/stage4a_metrics.json")
    assert measured["fresh_complete_official_VAL"] and measured["status"] == "PASS"
    assert measured["actor_errors_sha256"] == sha256(ACTORS)
    baseline, current = read_actors(BASE_ACTORS), read_actors(ACTORS)
    _, _, membership = strict_pairing(baseline, current)
    tables = main_tables(baseline, current, measured, membership)
    ci = bootstrap(baseline, current, measured["scenes"], membership)
    atomic_json(ROOT / "04_evaluation/stage4a_analysis_completion.json", {
        "status": "PASS", "strict_pairing": True, "motion_subgroups": True,
        "preregistered_interaction_subgroups": True, "scene_bootstrap": True,
        "actor_errors_sha256": sha256(ACTORS), "baseline_actor_errors_sha256": sha256(BASE_ACTORS),
        "membership_sha256": sha256(MEMBERSHIP), "bootstrap_sha256": sha256(ROOT / "04_evaluation/stage4a_bootstrap_ci.json"),
        "ablation_sha256": sha256(ROOT / "06_tables/stage4a_type_interaction_ablation.csv")})
    verify_previous(shards=False)
    print("STAGE4A_ANALYSIS=PASS", ci["groups"]["overall"], flush=True)


@torch.no_grad()
def diagnostics():
    from stage4a_common import model_input, config
    from torch_geometric.loader import DataLoader
    model, _, _ = load_final_model()
    model.global_interactor.record_relation_bias = True
    dataset = SceneDataset("val")
    counts = np.zeros(9, dtype=np.int64)
    sums = np.zeros((9, 3, 8), dtype=np.float64)
    squares = np.zeros_like(sums)
    absolute = np.zeros_like(sums)
    seen = 0
    for batch in DataLoader(dataset, batch_size=config()["batch_size"], shuffle=False, num_workers=0):
        data = batch.cuda()
        # Capture the exact relations used by this real forward. Observation
        # is detached by the model and never changes model parameters or inputs.
        model(model_input(data))
        diagnostic = model.global_interactor.relation_bias_observation
        pair = diagnostic["pair_ids"].detach().cpu().numpy()
        bias = diagnostic["bias"].detach().cpu().double().numpy()
        edge = diagnostic["edge_index"]
        assert bias.shape == (len(pair), 3, 8) and edge.shape == (2, len(pair))
        assert np.isfinite(bias).all()
        assert np.all((pair >= 0) & (pair < 9))
        expected = 3 * data.agent_type[edge[1]] + data.agent_type[edge[0]]
        assert np.array_equal(pair, expected.detach().cpu().numpy())
        for p in range(9):
            values = bias[pair == p]
            counts[p] += len(values)
            sums[p] += values.sum(0)
            squares[p] += (values * values).sum(0)
            absolute[p] += np.abs(values).sum(0)
        seen += batch.num_graphs
        if seen % 500 < config()["batch_size"]:
            print("STAGE4A_RELATION_DIAGNOSTICS", seen, "/", len(dataset), flush=True)
    dataset.clear()
    model.global_interactor.record_relation_bias = False
    assert seen == 3603
    rows, aggregates = [], []
    for p in range(9):
        target, source = CLASSES[p // 3], CLASSES[p % 3]
        label = f"{('V','P','B')[p//3]}<-{('V','P','B')[p%3]}"
        for layer in range(3):
            for head in range(8):
                n = int(counts[p])
                mean = sums[p, layer, head] / n if n else None
                variance = max(0., squares[p, layer, head] / n - mean ** 2) if n else None
                rows.append({"pair_id": p, "directed_pair": label, "target_type": target, "source_type": source,
                    "layer": layer, "head": head, "count": n, "mean_bias": mean,
                    "std_bias": float(np.sqrt(variance)) if n else None,
                    "mean_abs_bias": absolute[p, layer, head] / n if n else None})
        n = int(counts[p]) * 24
        mean = float(sums[p].sum() / n) if n else None
        variance = max(0., float(squares[p].sum() / n) - mean ** 2) if n else None
        aggregates.append({"pair_id": p, "directed_pair": label, "target_type": target, "source_type": source,
            "edge_observation_count": int(counts[p]), "layer_head_scalar_observation_count": n,
            "mean_bias": mean, "std_bias": float(np.sqrt(variance)) if n else None,
            "mean_abs_bias": float(absolute[p].sum() / n) if n else None})
    write_csv(ROOT / "06_tables/stage4a_relation_bias_statistics.csv", rows)
    write_csv(ROOT / "06_tables/stage4a_relation_bias_pair_aggregate.csv", aggregates)
    atomic_json(ROOT / "04_evaluation/stage4a_relation_bias_statistics.json", {
        "status": "PASS", "checkpoint_sha256": sha256(BEST), "VAL_scenes": 150, "VAL_windows": seen,
        "edge_observation_count": int(counts.sum()), "directed_type_pair_definition": "pair_id=3*target_type+source_type",
        "edge_observation_unit": "actual current valid agent-agent edge per scene window; overlapping observations not independent",
        "statistics": rows, "layer_head_aggregated": aggregates, "std_definition": "population standard deviation",
        "zero_observation_pair_statistics": "null, never zero imputed", "layer_head_indexing": "zero based",
        "causal_interpretation": False, "interpretation": "Learned internal pre-softmax bias diagnostic; magnitude is not causal impact or guaranteed importance",
        "source_csv_sha256": sha256(ROOT / "06_tables/stage4a_relation_bias_statistics.csv"),
        "aggregate_csv_sha256": sha256(ROOT / "06_tables/stage4a_relation_bias_pair_aggregate.csv")})
    print("STAGE4A_RELATION_DIAGNOSTICS=PASS", counts.tolist(), flush=True)


@torch.no_grad()
def efficiency():
    """500 alternating paired fixed VAL batches; CUDA events exclude I/O/H2D."""
    from stage4a_common import model_input, config
    from stage3b_common import model_new as baseline_new, model_input as baseline_input
    from torch_geometric.data import Batch
    model, _, _ = load_final_model()
    baseline_saved = torch.load(STAGE3_ROOT / "07_checkpoints/stage3b_best_overall_minfde.pt", map_location="cpu", weights_only=False)
    baseline = baseline_new()
    baseline.load_state_dict(baseline_saved["state_dict"])
    baseline.eval()
    model.eval()
    model.global_interactor.record_relation_bias = False
    assert not baseline.training and not model.training
    assert torch.cuda.device_count() >= 1
    dataset = SceneDataset("val")
    batch_size = config()["batch_size"]
    # Freeze the input sequence before timing. It cycles through complete VAL
    # order, yielding 500 real paired batches and exactly the same B/C tensors.
    windows = len(dataset)
    batch_indices = [list(range(start, min(start + batch_size, windows))) for start in range(0, windows, batch_size)]
    records = []
    warmup_batches = 20
    for repeat in range(warmup_batches):
        idx = batch_indices[repeat % len(batch_indices)]
        data = Batch.from_data_list([dataset[i] for i in idx]).cuda()
        baseline(baseline_input(data))
        model(model_input(data))
    torch.cuda.synchronize()
    baseline_peak = int(torch.cuda.memory_allocated())
    peaks = {"B_Stage3B": baseline_peak, "C_Stage4A": baseline_peak}
    models = {"B_Stage3B": (baseline, baseline_input), "C_Stage4A": (model, model_input)}
    # Both resident model parameter footprints are common to the peak-memory
    # baseline. Report total and forward incremental peaks to avoid attribution.
    for repeat in range(500):
        idx = batch_indices[repeat % len(batch_indices)]
        data = Batch.from_data_list([dataset[i] for i in idx]).cuda()
        torch.cuda.synchronize()
        order = ("B_Stage3B", "C_Stage4A") if repeat % 2 == 0 else ("C_Stage4A", "B_Stage3B")
        row = {"paired_batch_index": repeat, "VAL_batch_index": repeat % len(batch_indices), "graphs": len(idx), "nodes": data.num_nodes}
        for name in order:
            network, prepare = models[name]
            working = prepare(data)
            torch.cuda.synchronize()
            before = int(torch.cuda.memory_allocated())
            torch.cuda.reset_peak_memory_stats()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            output = network(working)
            end.record()
            end.synchronize()
            elapsed = float(start.elapsed_time(end))
            assert elapsed > 0 and torch.isfinite(output["raw_prediction"]).all()
            peak = int(torch.cuda.max_memory_allocated())
            peaks[name] = max(peaks[name], peak)
            row[name + "_ms"] = elapsed
            row[name + "_peak_cuda_bytes"] = peak
            row[name + "_forward_incremental_peak_bytes"] = peak - before
            del output, working
        records.append(row)
        del data
        if (repeat + 1) % 100 == 0:
            print("STAGE4A_EFFICIENCY_PAIRED_BATCHES", repeat + 1, "/ 500", flush=True)
    dataset.clear()
    # Measure each model's actual isolated allocated/reserved peak in a
    # separate untimed pass. Timing uses interleaved paired forwards; memory
    # uses one GPU-resident model after empty_cache and all 226 unique inputs.
    baseline.cpu()
    model.cpu()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    isolated_memory = {}
    for name in ("B_Stage3B", "C_Stage4A"):
        network, prepare = models[name]
        network.cuda()
        for repeat in range(warmup_batches):
            idx = batch_indices[repeat % len(batch_indices)]
            data = Batch.from_data_list([dataset[i] for i in idx]).cuda()
            output = network(prepare(data))
            del output, data
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        for idx in batch_indices:
            data = Batch.from_data_list([dataset[i] for i in idx]).cuda()
            output = network(prepare(data))
            assert torch.isfinite(output["raw_prediction"]).all()
            del output, data
        torch.cuda.synchronize()
        isolated_memory[name] = {
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "unique_batches_measured": len(batch_indices),
            "GPU_model_count": 1,
        }
        network.cpu()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    dataset.clear()
    rows = []
    baseline_parameters = sum(p.numel() for p in baseline.parameters())
    current_parameters = sum(p.numel() for p in model.parameters())
    for name, network in (("B_Stage3B", baseline), ("C_Stage4A", model)):
        values = np.array([r[name + "_ms"] for r in records])
        rows.append({"Model": name, "parameter_count": sum(p.numel() for p in network.parameters()),
            "additional_parameter_percent_vs_Stage3B": 100 * (sum(p.numel() for p in network.parameters()) - baseline_parameters) / baseline_parameters,
            "mean_inference_ms": float(values.mean()), "median_inference_ms": float(np.median(values)),
            "std_inference_ms": float(values.std(ddof=1)), "p95_inference_ms": float(np.quantile(values, .95)),
            "paired_measured_batches": len(records),
            "measured_scene_windows": sum(r["graphs"] for r in records), "batch_size": batch_size,
            "peak_CUDA_memory_MiB": isolated_memory[name]["peak_allocated_bytes"] / 2**20,
            "peak_CUDA_reserved_memory_MiB": isolated_memory[name]["peak_reserved_bytes"] / 2**20,
            "timing_both_models_resident_peak_MiB": peaks[name] / 2**20,
            "max_forward_incremental_peak_MiB": max(r[name + "_forward_incremental_peak_bytes"] for r in records) / 2**20})
    write_csv(ROOT / "06_tables/stage4a_efficiency.csv", rows)
    write_csv(ROOT / "06_tables/stage4a_efficiency_batch_timings.csv", records)
    audit = {"status": "PASS", "GPU": torch.cuda.get_device_name(), "device": "cuda:0",
        "torch_version": torch.__version__, "CUDA_version": torch.version.cuda, "eval_mode": True,
        "timing_method": "CUDA events around forward on preloaded identical GPU tensors; data loading/H2D and input clone excluded",
        "execution_order": "alternate B-C / C-B by paired batch to reduce order drift", "warmup_batches_per_model": warmup_batches,
        "paired_measured_batches": len(records), "minimum_500_batches_satisfied": True,
        "VAL_input_order": "Frozen official VAL sorted existing dataset order; deterministic cyclic batch repetition after one pass",
        "unique_VAL_batches": len(batch_indices), "unique_VAL_windows": windows,
        "identical_input_tensors_per_pair": True, "batch_size": batch_size,
        "both_models_resident_common_parameter_memory_baseline_bytes": baseline_peak,
        "peak_memory_method": "separate untimed isolated pass per model; other model CPU, empty_cache, warmup20, reset peak, all 226 unique identical VAL batches; actual max allocated/reserved",
        "isolated_memory": isolated_memory,
        "Stage3B_checkpoint_sha256": sha256(STAGE3_ROOT / "07_checkpoints/stage3b_best_overall_minfde.pt"),
        "Stage4A_checkpoint_sha256": sha256(BEST), "parameters_B": baseline_parameters, "parameters_C": current_parameters,
        "additional_parameters": current_parameters - baseline_parameters,
        "additional_parameters_percent": 100 * (current_parameters - baseline_parameters) / baseline_parameters,
        "mean_latency_overhead_percent": 100 * (rows[1]["mean_inference_ms"] / rows[0]["mean_inference_ms"] - 1),
        "median_latency_overhead_percent": 100 * (rows[1]["median_inference_ms"] / rows[0]["median_inference_ms"] - 1),
        "overhead_percent": 100 * (rows[1]["mean_inference_ms"] / rows[0]["mean_inference_ms"] - 1),
        "models": {"Stage3B": rows[0], "Stage4A": rows[1]},
        "metrics": rows, "batch_timing_source_sha256": sha256(ROOT / "06_tables/stage4a_efficiency_batch_timings.csv")}
    atomic_json(ROOT / "04_evaluation/stage4a_efficiency_audit.json", audit)
    print("STAGE4A_EFFICIENCY=PASS", audit["mean_latency_overhead_percent"], flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--register-subgroups", action="store_true")
    parser.add_argument("--fresh-val", action="store_true")
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--efficiency", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    assert any(vars(args).values()), "Select explicit stages; GPU work is never implicit"
    if args.register_subgroups:
        register_subgroups()
    if args.fresh_val:
        fresh_val()
    if args.analyze:
        analyze()
    if args.diagnostics:
        diagnostics()
    if args.efficiency:
        efficiency()


if __name__ == "__main__":
    main()
