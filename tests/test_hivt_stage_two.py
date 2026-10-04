import unittest

import torch
from torch_geometric.data import Batch

from datasets.nuscenes_hivt_vehicle_dataset import VehicleGraph
from metrics.hivt_forecasting import multimodal_errors
from models.hivt_nuscenes import HiVTNuScenesVehicle


def graph():
    n, th, tf = 2, 5, 12
    return VehicleGraph(num_nodes=n, x=torch.zeros(n, th, 2), positions=torch.zeros(n, th+tf, 2), y=torch.zeros(n, tf, 2),
                        edge_index=torch.tensor([[0, 1], [1, 0]]), padding_mask=torch.zeros(n, th+tf, dtype=torch.bool),
                        bos_mask=torch.tensor([[True, False, False, False, False]]*n), rotate_angles=torch.tensor([.2, -.4]),
                        lane_vectors=torch.tensor([[2., 0], [2., 0]]), lane_positions=torch.zeros(2, 2),
                        lane_actor_index=torch.tensor([[0, 1], [0, 1]]), lane_actor_vectors=torch.ones(2, 2),
                        is_intersections=torch.zeros(2, dtype=torch.uint8), turn_directions=torch.zeros(2, dtype=torch.uint8),
                        traffic_controls=torch.zeros(2, dtype=torch.uint8), future_mask=torch.ones(n, tf, dtype=torch.bool),
                        target_mask=torch.ones(n, dtype=torch.bool), agent_index=torch.arange(n))


class BaselineTest(unittest.TestCase):
    def test_batch_offsets(self):
        batch = Batch.from_data_list([graph(), graph()])
        self.assertEqual(batch.lane_actor_index.tolist(), [[0, 1, 2, 3], [0, 1, 2, 3]])
        self.assertEqual(batch.agent_index.tolist(), [0, 1, 2, 3])
        self.assertEqual(batch.edge_index.tolist(), [[0, 1, 2, 3], [1, 0, 3, 2]])

    def test_best_fde_mode_ade_not_independent_ade(self):
        pred = torch.tensor([[[[9., 0], [0, 0]], [[1., 0], [1., 0]]]])
        result = multimodal_errors(pred, torch.zeros(1, 2, 2), torch.ones(1, 2, dtype=torch.bool), torch.tensor([True]))
        self.assertEqual(result["minADE_K"].item(), 4.5)
        self.assertEqual(result["independent_minADE_K"].item(), 1.)
        self.assertEqual(result["minFDE_K"].item(), 0.)

    def test_partial_mask_and_strict_mr_threshold(self):
        pred = torch.tensor([[[[3., 0], [2., 0], [999., 999]]], [[[2., 0], [2., 0], [2.01, 0]]]])
        mask = torch.tensor([[True, True, False], [True, True, True]])
        target = torch.zeros(2, 3, 2)
        target[0, -1] = float("nan")
        result = multimodal_errors(pred, target, mask, torch.ones(2, dtype=torch.bool))
        self.assertEqual(result["MR_K"].tolist(), [0., 1.])
        self.assertEqual(result["full_horizon"].tolist(), [False, True])
        self.assertEqual(result["partial_future"].tolist(), [True, False])

    def test_repeated_forward_no_target_mutation_and_batch_equivalence(self):
        torch.manual_seed(7)
        model = HiVTNuScenesVehicle().eval()
        data = graph()
        saved = data.y.clone()
        with torch.no_grad():
            first = model(data)
            second = model(data)
            batched = model(Batch.from_data_list([data, data]))
            poisoned = data.clone()
            poisoned.y[:] = 999
            poisoned.positions[:, 5:] = 999
            poisoned.padding_mask[:, 5:] = True
            poisoned.future_mask[:] = False
            future_changed = model(poisoned)
        torch.testing.assert_close(data.y, saved)
        torch.testing.assert_close(first["raw_prediction"], second["raw_prediction"])
        torch.testing.assert_close(first["raw_prediction"], future_changed["raw_prediction"])
        torch.testing.assert_close(first["raw_prediction"], batched["raw_prediction"][:, :2], atol=1e-5, rtol=1e-4)
        self.assertEqual(model.ego_predictions(first, data).shape, (2, 6, 12, 2))
        torch.testing.assert_close(first["mode_prob"].sum(dim=1), torch.ones(2))


if __name__ == "__main__":
    unittest.main()
