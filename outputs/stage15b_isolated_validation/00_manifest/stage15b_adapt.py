"""One-time Stage15B adaptation of reviewed Stage15A entry; never edits history."""
from pathlib import Path
r=Path(__file__).resolve().parents[1]
p=r/'00_manifest/stage15b_common.py';s=p.read_text()
s=s.replace('output_dir, install=True):','output_dir, install=True, stage="predictor"):\n        assert stage in ("predictor", "candidate", "ranking", "evaluation")\n        self.stage = stage\n        if stage != "predictor":\n            assert read_json(ROOT / "04_predictor_checkpoints/stage15b_all_frozen.json")["Status"] == "FROZEN_ALL_COMPLETE"\n        if stage == "evaluation":\n            assert read_json(ROOT / "07_rank_checkpoints/stage15b_all_frozen.json")["Status"] == "FROZEN_ALL_COMPLETE"')
s=s.replace('self.allowed_files = {p for p, s in self.path_to_scene.items() if s in self.parts["InnerTrain"] | self.parts["InnerDev"]}', 'allowed_scenes = self.parts["InnerTrain"] | self.parts["InnerDev"]\n        if stage in ("candidate", "evaluation"):\n            allowed_scenes |= self.parts["OuterTest"]\n        self.allowed_files = {p for p, s in self.path_to_scene.items() if s in allowed_scenes}\n        reg = ROOT / "00_manifest/stage15b_registration.json"\n        assert reg.exists(), "Source registration required before any graph/model use"\n        for file, digest in read_json(reg)["sources"].items():\n            assert sha256(ROOT / file) == digest, "Frozen Stage15B source changed: " + file')
a=s.index('        assert role in ("InnerTrain", "InnerDev"), "OuterTest cannot be evaluated in Stage15B"');b=s.index('        self.batch_records.append',a)
s=s[:a]+'''        assert role in ("InnerTrain", "InnerDev", "OuterTest")
        scenes = list(data.scene_token)
        assert scenes and set(scenes) <= self.parts[role], "Batch scene role mismatch"
        assert not set(scenes) & self.parts["QuarantinedHeadDev"]
        if role == "OuterTest":
            assert self.stage in ("candidate", "evaluation")
            assert purpose in ("candidate_interface", "outer_evaluation")
        if purpose == "optimization":
            assert role == "InnerTrain" and self.stage == "predictor"
'''+s[b:]
s=s.replace('assert scope == "preflight_checkpoint" and p.is_relative_to(self.output)', 'assert scope in ("preflight_checkpoint", "formal_checkpoint") and p.is_relative_to(self.output)')
s=s.replace('assert role in ("InnerTrain", "InnerDev"), "Only train/dev datasets exist in this entry"','assert role in ("InnerTrain", "InnerDev") or role == "OuterTest" and context.stage in ("candidate", "evaluation")')
a=s.index('@torch.no_grad()\ndef evaluate_dev');s=s[:a]+'''def motion_values(batch):
    values = batch.t0_motion_state
    if values and isinstance(values[0], (tuple, list)):
        values = [v for row in values for v in row]
    assert len(values) == batch.num_nodes
    return values


@torch.no_grad()
def evaluate_dev(model, dataset, ids=None):
    assert dataset.role == "InnerDev"
    model.eval()
    batches = [list(range(i, min(i + 16, len(dataset)))) for i in range(0, len(dataset), 16)] if ids is None else [list(ids)]
    groups = ("Overall", "Vehicle", "Pedestrian", "Bicycle", "MovingVehicle")
    sums = {k: np.zeros(2, np.float64) for k in groups}
    counts = dict.fromkeys(groups, 0)
    for batch in DataLoader(dataset, batch_sampler=batches, num_workers=0):
        dataset.context.check_batch(batch, "InnerDev", "development_evaluation")
        moving = torch.tensor([s == "vehicle.moving" for s in motion_values(batch)], device="cuda")
        data = batch.cuda()
        observed = model_input(data)
        out = model(observed)
        pred = model.ego_predictions(out, observed)
        errors = multimodal_errors(pred, data.positions[:, 5:], data.future_mask, data.target_mask)
        full = errors["full_horizon"]
        assert torch.isfinite(errors["minFDE_K"][full]).all()
        masks = {"Overall": full, **{k: full & (data.agent_type == i) for i,k in enumerate(groups[1:4])},
                 "MovingVehicle": full & (data.agent_type == 0) & moving}
        for k, mask in masks.items():
            counts[k] += int(mask.sum())
            sums[k] += np.array([float(errors[m][mask].double().sum()) for m in ("minADE_K", "minFDE_K")])
    assert counts["Overall"] > 0
    metrics = {k: {"Count": counts[k], "minADE6": sums[k][0]/counts[k] if counts[k] else None,
                   "minFDE6": sums[k][1]/counts[k] if counts[k] else None} for k in groups}
    return {"full_horizon_Overall_minFDE6": metrics["Overall"]["minFDE6"], "Count": counts["Overall"],
            "selection_role": "InnerDev", "subset_only": ids is not None, "metrics": metrics,
            "ADE_definition": "ADE of FDE-minimizing mode, inherited multimodal_errors"}
'''
p.write_text(s)
p=r/'02_training/stage15b_train.py';s=p.read_text().replace('stage15b_common.py", ROOT / "02_training/stage15b_train.py"','stage15b_common.py", ROOT / "02_training/stage15b_train.py"')
s=s.replace('saved = context.load(path, "preflight_checkpoint")','saved = context.load(path, "formal_checkpoint" if scope == "FORMAL_SCENE_ISOLATED" else "preflight_checkpoint")')
a=s.index('def train_registered_phase');b=s.index('\ndef run_preflight_phases',a)
s=s[:a]+'''def train_registered_phase(context, phase):
    """Unchanged registered optimizer/selection with source and monitoring witnesses."""
    assert context.protocol["FullTrainingAuthorized"] and context.stage == "predictor"
    assert phase in ("fixed_scale", "original_nll")
    scope = "FORMAL_SCENE_ISOLATED"
    warm = phase == "fixed_scale"
    label = "warmup" if warm else "nll"
    budget, offset, lr = (5000, 0, .001) if warm else (16000, 5000, .0001)
    last = context.output / f"stage15b_{label}_last_checkpoint.pt"
    best = context.output / f"stage15b_{label}_best_overall_minfde.pt"
    warm_best = context.output / "stage15b_warmup_best_overall_minfde.pt"
    summary_path = context.output / f"stage15b_{label}_summary.json"
    if summary_path.exists():
        result = read_json(summary_path)
        assert result["status"] == "COMPLETE" and sha256(best) == result["selected_checkpoint_sha256"]
        return result
    state = new_phase_state()
    state.update(phase=phase, seen_train_scenes=[], seen_dev_scenes=[], monitoring=[], actual_invocation_seconds=0.)
    cursor = {"epoch": 0, "next_batch": 0}
    if last.exists():
        model, optimizer, saved = restore(last, context, scope)
        state, cursor = saved["phase_state"], saved["iterator"]
    elif warm:
        model = model_new(context.seed)
        atomic_json(context.output / "stage15b_initialization.json", {
            "fresh": True, "fold": context.fold, "seed": context.seed,
            "state_sha256": state_digest(model.state_dict()), "parameters": 650403,
            "HistoricalOrTinyWeightsLoaded": False, "protocol_sha256": sha256(PROTOCOL)})
        optimizer = model.optimizer(lr, .0001)
    else:
        summary = read_json(context.output / "stage15b_warmup_summary.json")
        assert summary["phase_steps"] == 5000 and summary["status"] == "COMPLETE"
        model, optimizer, cursor, transition = transition_from_own_warmup(warm_best, context, scope)
        atomic_json(context.output / "stage15b_formal_transition.json", transition)
    assert all(g["lr"] == lr for g in optimizer.param_groups)
    train, dev = FoldDataset(context, "InnerTrain"), FoldDataset(context, "InnerDev")
    seen_train, seen_dev = set(state["seen_train_scenes"]), set(state["seen_dev_scenes"])
    started = tick = time.monotonic()
    total = state.pop("pending_train_loss_sum", 0.)
    total_n = state.pop("pending_train_loss_steps", 0)
    witness_path = context.output / "stage15b_batch_sources.log"
    torch.cuda.reset_peak_memory_stats()
    with witness_path.open("a") as witness:
        while state["phase_step"] < budget and not state["completed"]:
            row = optimize_step(model, optimizer, train, cursor, phase)
            total += row["loss"]; total_n += 1
            seen_train.update(row["scene_tokens"])
            state["phase_step"] += 1
            global_step = offset + state["phase_step"]
            assert global_step <= 21000
            witness.write(__import__("json").dumps({"step": global_step, "phase": phase, "role": "InnerTrain",
                "scenes": row["scene_tokens"], "samples": row["sample_tokens"]}) + "\\n")
            if state["phase_step"] % 500 == 0:
                measured = evaluate_dev(model, dev)
                seen_dev.update(context.parts["InnerDev"])
                assert measured["subset_only"] is False
                improved = update_selection(state, measured["full_horizon_Overall_minFDE6"], global_step, phase, budget)
                state["last_development"] = measured
                state["seen_train_scenes"], state["seen_dev_scenes"] = sorted(seen_train), sorted(seen_dev)
                monitor = {"fold": context.fold, "seed": context.seed, "phase": phase, "global_step": global_step,
                    "training_loss_mean_last500_steps": total/total_n, "measured": measured,
                    "lr": lr, "improved": improved, "best_step": state["best_step"],
                    "best_FDE": state["best_FDE"], "bad_validations": state["bad_validations"],
                    "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                    "block_wall_seconds": time.monotonic()-tick, "checkpoint": str(best.relative_to(ROOT))}
                state["monitoring"].append(monitor)
                state["actual_invocation_seconds"] += time.monotonic()-tick
                if improved:
                    checkpoint_save(best, model, optimizer, context, state, cursor, scope)
                checkpoint_save(last, model, optimizer, context, state, cursor, scope)
                atomic_json(context.output / f"stage15b_dev_step_{global_step:05d}.json", monitor)
                witness.flush()
                print("PREDICTOR", context.fold, phase, global_step, "DEV", measured["full_horizon_Overall_minFDE6"],
                      "BEST", state["best_step"], "BAD", state["bad_validations"], flush=True)
                total, total_n, tick = 0., 0, time.monotonic()
    assert seen_train == set(context.parts["InnerTrain"]) and seen_dev == set(context.parts["InnerDev"])
    summary = {"status": "COMPLETE", "fold": context.fold, "seed": context.seed,
        "phase_steps": state["phase_step"], "phase": phase,
        "global_executed_step": offset + state["phase_step"], "best_global_step": state["best_step"],
        "best_overall_FDE": state["best_FDE"], "consecutive_nonimprovements": state["bad_validations"],
        "stop_reason": "warmup_fixed5000" if warm else ("patience_5" if state["bad_validations"] >= 5 else "global21000_budget"),
        "selected_checkpoint": str(best.relative_to(ROOT)), "selected_checkpoint_sha256": sha256(best),
        "last_checkpoint_sha256": sha256(last), "elapsed_seconds": state["actual_invocation_seconds"],
        "TrainScenes": sorted(seen_train), "DevScenes": sorted(seen_dev), "OuterTestUsed": False,
        "HeadDevUsed": False, "Monitoring": state["monitoring"], "BatchSourceWitnessSHA256": sha256(witness_path),
        "LoadedRawShardSHA256": context.loaded_shards}
    atomic_json(summary_path, summary)
    del model, optimizer
    train.clear(); dev.clear(); torch.cuda.empty_cache()
    return summary

'''+s[b:]
a=s.index('def main():');b=s.index('\n\nif __name__',a)
s=s[:a]+'''def main():
    args = parse_args()
    context = FoldContext(args.fold, args.seed, args.training_scenes, args.development_scenes, args.output_dir)
    if args.mode == "describe":
        atomic_json(context.output / "stage15b_entry_description.json", {"Status": "PASS", "Config": config_for(context),
            "TrainWindows": len(FoldDataset(context, "InnerTrain")), "DevWindows": len(FoldDataset(context, "InnerDev"))})
    elif args.mode == "preflight":
        run_preflight_phases(context)
    else:
        try:
            train_registered_phase(context, "fixed_scale")
            train_registered_phase(context, "original_nll")
        except BaseException as exc:
            atomic_json(context.output / "stage15b_failure.json", {"Status": "FAILED_STOP", "fold": context.fold,
                "exception": repr(exc), "traceback": __import__("traceback").format_exc(),
                "NoProtocolOrSeedChangePermitted": True})
            raise
'''+s[b:]
s=s.replace('Explicit fold/seed/list entry. Stage15B only permits bounded preflight updates.','Scene-isolated Stage15B formal entry; exact frozen Stage5A protocol.')
p.write_text(s)
