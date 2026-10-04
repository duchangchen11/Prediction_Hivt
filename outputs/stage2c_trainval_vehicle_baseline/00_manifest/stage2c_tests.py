"""Behavioral checks for paired cluster uncertainty and bounded data access."""
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "00_manifest"))
from stage2c_common import MetricAccumulator, SceneShardDataset, SceneShuffleSampler, config
import numpy as np
import torch
from models.hivt_loss_recovery import HiVTLossRecovery
from scripts.train_hivt_stage2 import MODEL_KEYS
spec=importlib.util.spec_from_file_location("stage2c_evaluate", ROOT / "04_evaluation/stage2c_evaluate.py")
module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


class Stage2CBehaviorTests(unittest.TestCase):
    def test_scene_cluster_CI_does_not_narrow_when_correlated_actors_duplicated(self):
        def summaries(multiplier):
            h={"scenes":{}};c={"scenes":{}}
            for token,n,difference in (("a",1,1.),("b",9,-1.)):
                h["scenes"][token]={"full_horizon":{}};c["scenes"][token]={"full_horizon":{}}
                for group in ("overall","vehicle.moving"):
                    h["scenes"][token]["full_horizon"][group]={"count":n*multiplier,"minADE6":2+difference,"minFDE6":2+difference}
                    c["scenes"][token]["full_horizon"][group]={"count":n*multiplier,"minADE6":2.,"minFDE6":2.}
            return h,c
        a=module.paired_scene_bootstrap(*summaries(1));b=module.paired_scene_bootstrap(*summaries(100))
        for group in ("overall","vehicle.moving"):
            for metric in ("delta_ADE","delta_FDE"):
                self.assertAlmostEqual(a["groups"][group][metric]["estimate_m"],-.8)
                self.assertEqual(a["groups"][group][metric]["CI95_m"],[-1.,1.])
                self.assertEqual(a["groups"][group][metric]["CI95_m"],b["groups"][group][metric]["CI95_m"])

    def test_shard_loading_is_bounded_and_samples_are_independent_clones(self):
        ds=SceneShardDataset("train",cache_scenes=2)
        selected=[values[0] for values in ds.scene_indices.values()][:3]
        self.assertEqual(len(selected),3)
        first=ds[selected[0]];expected=first.x.clone();first.x.fill_(999)
        self.assertTrue((ds[selected[0]].x==expected).all())
        for i in selected:ds[i]
        self.assertLessEqual(len(ds.cache),2)
        sampler=SceneShuffleSampler(ds);indices=list(sampler)
        self.assertEqual(sorted(indices),list(range(len(ds))))
        sampler.epoch=1;self.assertNotEqual(indices,list(sampler))

    def test_future_targets_and_future_masks_cannot_change_predictions(self):
        ds=SceneShardDataset("train");g=ds[0];c=config()
        torch.manual_seed(2022);model=HiVTLossRecovery(**{k:c[k] for k in MODEL_KEYS}).eval()
        altered=g.clone();altered.y.fill_(1000);altered.positions[:,5:]=1000
        altered.future_mask[:]=False;altered.padding_mask[:,5:]=True
        with torch.no_grad():a=model(g);b=model(altered)
        torch.testing.assert_close(a["raw_prediction"],b["raw_prediction"],atol=1e-6,rtol=1e-6)
        torch.testing.assert_close(a["mode_prob"],b["mode_prob"],atol=1e-6,rtol=1e-6)

    def test_empty_windows_remain_indexed_without_losing_eligible_targets(self):
        ds=SceneShardDataset("train")
        self.assertEqual(len(ds.rows)+len(ds.empty_rows),len(ds.all_rows))
        self.assertGreater(len(ds.empty_rows),0)
        for field in ("full_horizon_target_count","partial_target_count"):
            self.assertEqual(sum(int(r[field]) for r in ds.rows),sum(int(r[field]) for r in ds.all_rows))
        context=next(r for r in ds.empty_rows if int(r["vehicle_count"])>0)
        saved=torch.load(ROOT/context["file_path"],weights_only=False,map_location="cpu")
        graph=saved["graphs"][int(context["window_index"])]
        self.assertGreater(graph.num_nodes,0)
        self.assertEqual(int(graph.target_mask.sum()),0)
        empty=next(r for r in ds.empty_rows if int(r["vehicle_count"])==0)
        saved=torch.load(ROOT/empty["file_path"],weights_only=False,map_location="cpu")
        self.assertIsNone(saved["graphs"][int(empty["window_index"])])


if __name__ == "__main__":unittest.main()
