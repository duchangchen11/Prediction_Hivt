import unittest

import numpy as np
import torch

from datasets.scene_split import split_scenes, validate_scene_split
from metrics.trajectory import displacement_errors
from models.constant_velocity import constant_velocity
from preprocessing.coordinates import (ego_heading_to_global, ego_to_global, global_heading_to_ego,
                                       global_to_ego, quaternion_yaw, wrap_angle)


class CoordinatesTest(unittest.TestCase):
    def test_direction_handedness_and_roundtrip(self):
        origin = np.array([100., 200.])
        yaw = np.pi/2
        points = np.array([[100, 210], [90, 200], [107, 204]])
        local = global_to_ego(points, origin, yaw)
        np.testing.assert_allclose(local[:2], [[10, 0], [0, 10]], atol=1e-12)
        np.testing.assert_allclose(ego_to_global(local, origin, yaw), points, atol=1e-12)
        # The quaternion is [w,x,y,z], not [x,y,z,w].
        self.assertAlmostEqual(quaternion_yaw([np.cos(yaw/2), 0, 0, np.sin(yaw/2)]), yaw)

    def test_heading_wrap(self):
        headings = np.array([-np.pi+.01, np.pi-.01, 0])
        yaw = np.pi-.1
        local = global_heading_to_ego(headings, yaw)
        np.testing.assert_allclose(wrap_angle(ego_heading_to_global(local, yaw)-headings), 0, atol=1e-12)
        self.assertAlmostEqual(float(global_heading_to_ego(yaw, yaw)), 0)


class ConstantVelocityTest(unittest.TestCase):
    def test_last_two_valid_points_and_irregular_timestamps(self):
        positions = torch.tensor([[[0., 0], [1000, 1000], [3, 6]], [[0., 0], [0, 0], [1, 2]]])
        mask = torch.tensor([[True, False, True], [False, False, True]])
        history = torch.tensor([-1.5, -.5, 0.], dtype=torch.float64)
        future = torch.tensor([.4, 1.1, 6.], dtype=torch.float64)
        predicted, valid, velocity = constant_velocity(positions, mask, history, future)
        torch.testing.assert_close(velocity[0], torch.tensor([2., 4.]))
        torch.testing.assert_close(predicted[0], torch.tensor([[3.8, 7.6], [5.2, 10.4], [15., 30.]]))
        self.assertEqual(valid.tolist(), [True, False])

    def test_stale_last_observation_uses_its_timestamp(self):
        positions = torch.tensor([[[0., 0], [1, 0], [0, 0]]])
        mask = torch.tensor([[True, True, False]])
        prediction, valid, _ = constant_velocity(positions, mask, torch.tensor([-1., -.5, 0.]), torch.tensor([.5]))
        torch.testing.assert_close(prediction[0, 0], torch.tensor([3., 0]))
        self.assertTrue(valid.item())

    def test_perfect_linear_motion(self):
        positions = torch.tensor([[[-2., 0], [-1, 0], [0, 0]]])
        prediction, valid, _ = constant_velocity(positions, torch.ones(1, 3, dtype=torch.bool),
                                                 torch.tensor([-1., -.5, 0.]), torch.tensor([.5, 1., 6.]))
        target = torch.tensor([[[1., 0], [2, 0], [12, 0]]])
        errors = displacement_errors(prediction, target, torch.ones(1, 3, dtype=torch.bool), valid)
        self.assertEqual(errors["ade"].item(), 0.)
        self.assertEqual(errors["fde"].item(), 0.)


class MaskedMetricTest(unittest.TestCase):
    def test_missing_future_and_last_valid_fde(self):
        target = torch.zeros(3, 3, 2)
        target[0, 1] = float("nan")
        prediction = torch.tensor([[[3., 4], [999, 999], [0, 2]], [[6., 8], [999, 999], [999, 999]], [[999., 999]]*3])
        mask = torch.tensor([[True, False, True], [True, False, False], [False, False, False]])
        result = displacement_errors(prediction, target, mask)
        torch.testing.assert_close(result["ade"], torch.tensor([3.5, 10., 0.]))
        torch.testing.assert_close(result["fde"], torch.tensor([2., 10., 0.]))
        self.assertEqual(result["valid"].tolist(), [True, True, False])
        self.assertEqual(result["fixed_horizon_valid"].tolist(), [True, False, False])

    def test_prediction_mask_excludes_short_history(self):
        result = displacement_errors(torch.zeros(1, 2, 2), torch.ones(1, 2, 2),
                                     torch.ones(1, 2, dtype=torch.bool), torch.tensor([False]))
        self.assertFalse(result["valid"].item())


class SceneSplitTest(unittest.TestCase):
    def test_disjoint_reproducible_full_coverage(self):
        tokens = [f"scene-{i}" for i in range(10)]
        split = split_scenes(tokens)
        self.assertEqual([len(split[p]) for p in ("train", "val", "test")], [6, 2, 2])
        self.assertEqual(split, split_scenes(list(reversed(tokens))))
        validate_scene_split(split, tokens)

    def test_leak_and_duplicates_rejected(self):
        with self.assertRaises(AssertionError):
            validate_scene_split({"train": ["a"], "val": ["a"], "test": ["c"]})
        with self.assertRaises(ValueError):
            split_scenes(["a", "a", "b"])


if __name__ == "__main__":
    unittest.main()
