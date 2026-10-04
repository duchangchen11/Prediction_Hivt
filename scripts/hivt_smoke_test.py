"""Real trainval batch sizes 1/2, forward/loss/backward/step checks."""
import torch
from torch_geometric.data import Batch
from nuscenes.nuscenes import NuScenes

from datasets.nuscenes_hivt_vehicle_dataset import NuScenesHiVTVehicleDataset
from models.hivt_nuscenes import HiVTNuScenesVehicle
from preprocessing.common import PROJECT_ROOT, write_json


def main():
    torch.set_num_threads(4)
    torch.manual_seed(2022)
    root = PROJECT_ROOT/"outputs/stage2/trainval/cache"
    nusc = NuScenes(version="v1.0-trainval", dataroot=str(root), verbose=False)
    tokens = [s["token"] for s in nusc.scene]
    dataset = NuScenesHiVTVehicleDataset(nusc, tokens, anchors=[(t, 10) for t in tokens])
    graphs = [dataset[i] for i in range(len(dataset))]
    model = HiVTNuScenesVehicle().cuda()
    optimizer = model.optimizer()
    tests = []
    for batch_size in (1, 2):
        batch = Batch.from_data_list(graphs[:batch_size]).to("cuda")
        output = model(batch)
        pred = model.ego_predictions(output, batch)
        assert pred.shape == (batch.num_nodes, 6, 12, 2)
        assert output["mode_prob"].shape == (batch.num_nodes, 6)
        assert torch.isfinite(pred).all()
        torch.testing.assert_close(output["mode_prob"].sum(dim=1), torch.ones(batch.num_nodes, device="cuda"))
        losses = model.loss(output, batch)
        assert all(torch.isfinite(v) for v in losses.values())
        optimizer.zero_grad(set_to_none=True)
        losses["loss"].backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        parameter = model.decoder.loc[-1].weight
        before = parameter.detach().clone()
        optimizer.step()
        assert torch.isfinite(parameter).all() and not torch.equal(before, parameter)
        tests.append({"batch_windows": batch_size, "vehicle_nodes": batch.num_nodes,
                      "pred_traj_shape": list(pred.shape), "mode_prob_shape": list(output["mode_prob"].shape),
                      "forward": "PASS", "prediction_finite": True,
                      "loss": {k: float(v) for k, v in losses.items()}, "loss_finite": True,
                      "backward": "PASS", "gradient_finite": True, "optimizer_step": "PASS"})
    result = {"status": "PASS", "dataset": "v1.0-trainval, five complete source scenes in metadata cache",
              "raw_sensor_files_read": 0, "graphs_read": len(graphs), "tests": tests,
              "parameter_count": sum(p.numel() for p in model.parameters()),
              "scene_tokens": tokens}
    write_json(PROJECT_ROOT/"outputs/reports/trainval_hivt_smoke.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    main()
