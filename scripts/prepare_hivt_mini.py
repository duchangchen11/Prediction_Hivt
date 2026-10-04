"""Materialize only mini graphs locally and validate 100+ assigned windows."""
import json
from pathlib import Path
import random
import time

import torch
from torch_geometric.data import Batch

from datasets.nuscenes_hivt_vehicle_dataset import NuScenesHiVTVehicleDataset
from datasets.scene_split import validate_scene_split
from preprocessing.common import PROJECT_ROOT, load_config, load_nuscenes, write_json


def validate_graph(data):
    n, l = data.num_nodes, len(data.lane_vectors)
    assert data.x.shape == (n, 5, 2) and data.y.shape == (n, 12, 2)
    assert data.positions.shape == (n, 17, 2) and data.padding_mask.shape == (n, 17)
    assert data.history_times.shape == (5,) and data.future_times.shape == (12,)
    assert torch.equal(data.padding_mask, ~torch.cat([data.history_mask, data.future_mask], dim=1))
    assert torch.equal(data.agent_index, torch.arange(n))
    assert len(data.instance_tokens) == len(set(data.instance_tokens)) == n
    assert all(c.startswith("vehicle.") and c != "vehicle.bicycle" for c in data.category_names)
    assert torch.equal(data.full_horizon_mask, (data.history_mask.sum(dim=1)>=2) & data.future_mask.all(dim=1))
    assert data.lane_mask.shape == (l,) and data.lane_mask.all()
    if data.lane_actor_index.numel():
        assert data.lane_actor_index[0].min() >= 0 and data.lane_actor_index[0].max() < l
        assert data.lane_actor_index[1].min() >= 0 and data.lane_actor_index[1].max() < n
        expected = data.lane_positions[data.lane_actor_index[0]]-data.positions[data.lane_actor_index[1], 4]
        torch.testing.assert_close(expected, data.lane_actor_vectors)
        assert (torch.linalg.vector_norm(data.lane_actor_vectors, dim=1)<50.0001).all()
    for key, value in data:
        if torch.is_tensor(value) and value.is_floating_point():
            assert torch.isfinite(value).all(), key


def main():
    config = load_config()
    nusc = load_nuscenes(config)
    split = json.loads((PROJECT_ROOT/"outputs/reports/scene_split.json").read_text())["scenes"]
    validate_scene_split(split, [s["token"] for s in nusc.scene])
    corpus = {}
    started = time.monotonic()
    for name, tokens in split.items():
        dataset = NuScenesHiVTVehicleDataset(nusc, tokens)
        graphs = []
        for i in range(len(dataset)):
            graph = dataset[i]
            assert graph.scene_token in tokens
            validate_graph(graph)
            graphs.append(graph)
            if i % 20 == 0:
                print(name, i+1, '/', len(dataset), f'elapsed={time.monotonic()-started:.1f}s', flush=True)
        corpus[name] = graphs
    population = [graph for graphs in corpus.values() for graph in graphs]
    rng = random.Random(42)
    chosen = rng.sample(population, 100)
    for graph in chosen:
        validate_graph(graph)
    batch = Batch.from_data_list(chosen[:4])
    assert batch.agent_index.max() < batch.num_nodes
    assert batch.lane_actor_index[0].max() < len(batch.lane_vectors)
    assert batch.lane_actor_index[1].max() < batch.num_nodes
    node_batch = batch.batch
    for row, col in batch.edge_index.T.tolist():
        assert node_batch[row] == node_batch[col], "actor edge crosses scenes"
    source_counts = {name: len(graphs) for name, graphs in corpus.items()}
    result = {"status": "PASS", "randomly_validated_windows": 100, "all_generated_windows_validated": len(population),
              "nan_count": 0, "inf_count": 0, "batch_size_checked": 4, "scene_split_disjoint": True,
              "counts": source_counts, "split_file": "outputs/reports/scene_split.json",
              "sampled_window_tokens": [g.sample_token for g in chosen],
              "map_radius_m": 50., "lane_resolution_m": 2., "lane_semantic_attributes": "constant zero placeholders; geometry only",
              "history_frames": 5, "future_frames": 12, "ego_prediction_target": False,
              "training_target": "current vehicle.* except bicycle, >=2 history observations, >=1 future observation",
              "metric_full_horizon": "all 12 future keyframes observed; elapsed duration recorded, approximately 6s",
              "full_horizon_actor_windows": {name: sum(int(g.full_horizon_mask.sum()) for g in graphs) for name, graphs in corpus.items()},
              "partial_future_actor_windows": {name: sum(int((g.target_mask & ~g.full_horizon_mask).sum()) for g in graphs) for name, graphs in corpus.items()},
              "future_duration_range_s": [min(float(g.future_times[-1]) for g in population), max(float(g.future_times[-1]) for g in population)]}
    write_json(PROJECT_ROOT/"outputs/reports/hivt_dataset_validation.json", result)
    destination = PROJECT_ROOT/"outputs/stage2/mini/processed/graphs.pt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(corpus, destination)
    print(result, flush=True)


if __name__ == "__main__":
    main()
