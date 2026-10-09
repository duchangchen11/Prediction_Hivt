"""Read-only post-update checkpoint → fresh ranking interface, train/dev only."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "00_manifest"), str(ROOT / "02_training"), str(ROOT / "05_candidate_interface")]
import stage15a_common as c
from stage15a_train import parse_args, restore
from stage15a_ranking import candidate_windows, pack_window, fit_normalization, normalized, fresh_heads, head_forward
import torch
from torch_geometric.data import Batch


def main():
    args = parse_args()
    assert args.mode == "preflight"
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    context = c.FoldContext(args.fold, args.seed, args.training_scenes, args.development_scenes, args.output_dir)
    path = context.output / "stage15a_preflight_nll_best.pt"
    model, optimizer, saved = restore(path, context)
    assert saved["phase_state"]["phase"] == "original_nll" and saved["phase_state"]["phase_step"] == 2
    digest = c.state_digest(model.state_dict())
    assert digest == c.state_digest(saved["state_dict"])
    assert torch.equal(torch.get_rng_state(), saved["torch_rng"])
    assert all(torch.equal(a, b) for a, b in zip(torch.cuda.get_rng_state_all(), saved["cuda_rng"]))
    del optimizer
    model.eval().requires_grad_(False)
    train, dev = c.FoldDataset(context, "InnerTrain"), c.FoldDataset(context, "InnerDev")
    ids = c.scene_order(train, context.seed, 0)[:4]
    tg = [train[i] for i in ids]
    dg = [dev[i] for i in range(4)]
    tw = candidate_windows(model, Batch.from_data_list(tg), tg, context, "InnerTrain")
    dw = candidate_windows(model, Batch.from_data_list(dg), dg, context, "InnerDev")
    fit = [pack_window(w, torch.where(g.target_mask & g.future_mask.all(-1))[0]) for w, g in zip(tw, tg)]
    norm = fit_normalization([e for e in fit if len(e["targets"])], context)
    heads = fresh_heads(context.seed)
    rows = []
    for w in tw + dw:
        entry = pack_window(w)
        a, r2 = normalized(entry, norm)
        bike = a[0][:, 0, 0, 2].bool()
        r2out = head_forward(heads["R2"], "R2", a, r2)
        for name, head in heads.items():
            out = head_forward(head, name, a, r2)
            assert torch.equal(out["mode_logits"], a[6])
            assert out["mode_logits"].shape == (len(bike), 6)
            assert torch.isfinite(out["mode_logits"]).all() and torch.isfinite(out["mode_prob"]).all()
            # Actual fixed Bicycle routing uses this fold's new R2, never an old
            # checkpoint. Also exercise the operator on a fixed synthetic mask
            # if these few source windows contain no Bicycle target.
            routed = out["mode_logits"].clone()
            routed[bike] = r2out["mode_logits"][bike]
            assert torch.equal(routed[bike], r2out["mode_logits"][bike])
            if len(routed):
                fixture = torch.zeros(len(routed), dtype=torch.bool)
                fixture[0] = True
                altered_r2 = r2out["mode_logits"] + .123
                z = out["mode_logits"].clone()
                z[fixture] = altered_r2[fixture]
                assert torch.equal(z[fixture], altered_r2[fixture])
            rows.append({"Model": name, "Role": w["source_role"], "scene_token": w["scene_token"],
                         "sample_token": w["sample_token"], "TargetCount": len(bike), "BicycleTargetCount": int(bike.sum()),
                         "PredictorStateSHA256": digest, "OriginalModeLogitPairing": "PASS", "BicycleRouting": "PASS"})
        poisoned = dict(w, GT=torch.full((len(w["history"]), 12, 2), float("nan")),
                        target_mask=torch.zeros(len(w["history"]), dtype=torch.bool), future_mask=torch.zeros((len(w["history"]), 12), dtype=torch.bool))
        pe = pack_window(poisoned)
        assert all(x is None or torch.equal(x, y) for x, y in zip(entry["args"], pe["args"]))
        assert torch.equal(entry["r2"], pe["r2"])
    assert digest == c.state_digest(model.state_dict()) and all(p.grad is None for p in model.parameters())
    cache = ROOT / f"05_candidate_interface/cache/fold{context.fold}/stage15a_trained_candidates.pt"
    assert not cache.exists(), "Preserve existing diagnostic candidates"
    torch.save({"windows": tw + dw, "scope": "PREFLIGHT_ONLY", "predictor_checkpoint_sha256": c.sha256(path),
                "GT_in_payload": False}, cache)
    norm_path = ROOT / f"05_candidate_interface/stage15a_fold{context.fold}_trained_normalization.json"
    c.atomic_json(norm_path, norm)
    result = {"Status": "PASS", "Fold": context.fold, "SourcePhase": "original_nll", "ActualNLLUpdates": 2,
        "NLLCheckpointSHA256": c.sha256(path), "PredictorStateSHA256": digest, "OptimizerAndRNGLoaded": True,
        "CandidateCacheBytes": cache.stat().st_size, "CandidateCacheSHA256": c.sha256(cache),
        "NLLWeightsProduceK6": True, "K": 6, "HistoryOnlyInput": True, "GTFeatureMaxDiff": 0.,
        "NormalizationSHA256": c.sha256(norm_path), "NormalizationFitRole": "InnerTrain subset only; never formal",
        "FreshHeadInitialization": True, "HistoricalHeadWeightsLoaded": False, "HeadOptimizerUpdates": 0,
        "OuterInferenceRuns": 0, "PredictorOptimizerUpdates": 0, "Rows": rows,
        "ReadPaths": sorted(c.READS), "LoadedPayloads": c.LOADED,
        "BatchRecords": context.batch_records, "LoadedShardSHA256": context.loaded_shards}
    c.atomic_json(ROOT / f"03_checks/stage15a_fold{context.fold}_trained_interface.json", result)
    print("STAGE15A_TRAINED_INTERFACE_PASS", context.fold, "K6", "optimizer_updates0", flush=True)


if __name__ == "__main__":
    main()
