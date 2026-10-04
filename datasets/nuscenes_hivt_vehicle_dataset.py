"""Scene-first, in-memory nuScenes vehicle adapter for the frozen HiVT model."""
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from preprocessing.build_one_window import build_window
from preprocessing.common import scene_samples
from preprocessing.extract_lane_polylines import LaneExtractor


class VehicleGraph(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == "lane_actor_index":
            return torch.tensor([[self.lane_vectors.shape[0]], [self.num_nodes]])
        if key == "agent_index":
            return self.num_nodes
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in ("history_times", "future_times", "ego_history", "ego_future", "origin", "ego_yaw"):
            return None
        return super().__cat_dim__(key, value, *args, **kwargs)


class NuScenesHiVTVehicleDataset(Dataset):
    def __init__(self, nusc, scene_tokens, radius=50., resolution=2., anchors=None):
        self.nusc = nusc
        self.scene_tokens = frozenset(scene_tokens)
        self.extractor = LaneExtractor(nusc.dataroot, radius, resolution)
        self.chains = {token: scene_samples(nusc, nusc.get("scene", token)) for token in sorted(self.scene_tokens)}
        self.anchors = anchors if anchors is not None else [(token, i) for token, chain in self.chains.items() for i in range(4, len(chain)-12)]
        if any(token not in self.scene_tokens for token, _ in self.anchors):
            raise ValueError("Anchor scene outside assigned split")
        self.cache = {}

    def __len__(self):
        return len(self.anchors)

    def __getitem__(self, index):
        if index not in self.cache:
            token, t0 = self.anchors[index]
            scene = self.nusc.get("scene", token)
            window = build_window(self.nusc, scene, t0, samples=self.chains[token])
            selected = window["agent_type"] == 0  # taxonomy-derived vehicle, excluding bicycle
            if not selected.any():
                raise ValueError("Window has no vehicle at t0")
            indices = torch.where(selected)[0]
            history = window["agent_pos"][selected].clone()
            future = window["future_pos"][selected].clone()
            hmask = window["history_mask"][selected].clone()
            fmask = window["future_mask"][selected].clone()
            n = len(indices)
            eligible_history = hmask.sum(dim=1) >= 2
            # Upstream uses motion heading; retain annotation heading separately.
            angles = torch.zeros(n)
            for i in range(n):
                observed = torch.where(hmask[i])[0]
                if len(observed) >= 2:
                    direction = history[i, observed[-1]]-history[i, observed[-2]]
                    angles[i] = torch.atan2(direction[1], direction[0])
            x = torch.zeros_like(history)
            consecutive = hmask[:, :-1] & hmask[:, 1:]
            x[:, 1:] = torch.where(consecutive[..., None], history[:, 1:]-history[:, :-1], 0)
            y = torch.where(fmask[..., None], future-history[:, -1, None], 0)
            padding = ~torch.cat([hmask, fmask], dim=1)
            bos = torch.zeros_like(hmask)
            bos[:, 0] = hmask[:, 0]
            bos[:, 1:] = ~hmask[:, :-1] & hmask[:, 1:]
            row, col = torch.where(~torch.eye(n, dtype=torch.bool))
            location = self.nusc.get("log", scene["log_token"])["location"]
            lane = self.extractor.extract(location, window["ego_origin_global"].numpy(), window["ego_yaw_global"].item(), history[:, -1].numpy())
            lane_count = len(lane["lane_vectors"])
            data = VehicleGraph(
                x=x, positions=torch.cat([history, future], dim=1), y=y, num_nodes=n,
                edge_index=torch.stack([row, col]), padding_mask=padding, bos_mask=bos,
                rotate_angles=angles, agent_heading=window["agent_heading"][selected],
                history_mask=hmask, future_mask=fmask, target_mask=eligible_history & fmask.any(dim=1),
                full_horizon_mask=eligible_history & fmask.all(dim=1), agent_index=torch.arange(n),
                lane_vectors=torch.from_numpy(lane["lane_vectors"]), lane_positions=torch.from_numpy(lane["lane_positions"]),
                lane_actor_index=torch.from_numpy(lane["lane_actor_index"]), lane_actor_vectors=torch.from_numpy(lane["lane_actor_vectors"]),
                lane_mask=torch.ones(lane_count, dtype=torch.bool),
                is_intersections=torch.zeros(lane_count, dtype=torch.uint8),
                turn_directions=torch.zeros(lane_count, dtype=torch.uint8), traffic_controls=torch.zeros(lane_count, dtype=torch.uint8),
                history_times=window["history_times"], future_times=window["future_times"],
                ego_history=window["ego_history"], ego_future=window["ego_future"],
                origin=window["ego_origin_global"], ego_yaw=window["ego_yaw_global"],
                scene_token=token, scene_name=scene["name"], sample_token=window["sample_token"],
                instance_tokens=[window["instance_tokens"][i] for i in indices.tolist()],
                category_names=[window["category_names"][i] for i in indices.tolist()],
                map_location=location, lane_tokens=lane["lane_tokens"], th=5, tf=12)
            self.cache[index] = data
        return self.cache[index].clone()
