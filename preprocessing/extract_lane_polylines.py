"""nuScenes lane/connector centerlines; upstream 50m per-actor map radius."""
import numpy as np
from nuscenes.map_expansion.map_api import NuScenesMap

from .coordinates import ego_to_global, global_to_ego
from .map_paths import map_root_view


class LaneExtractor:
    def __init__(self, root, radius=50.0, resolution=2.0):
        self.root = map_root_view(root)
        self.radius, self.resolution = radius, resolution
        self.maps, self.polyline_cache, self.patch_cache = {}, {}, {}

    def api(self, location):
        if location not in self.maps:
            self.maps[location] = NuScenesMap(dataroot=str(self.root), map_name=location)
        return self.maps[location]

    def extract(self, location, origin, yaw, actor_positions):
        api = self.api(location)
        actors = ego_to_global(np.asarray(actor_positions), origin, yaw)
        # A cached enclosing rectangle is only for map lookup. Actual lane
        # segments/relations are filtered by the original Euclidean radius.
        lower = np.floor((actors.min(axis=0)-self.radius)/20)*20
        upper = np.ceil((actors.max(axis=0)+self.radius)/20)*20
        patch = (*lower, *upper)
        key = (location, patch)
        if key not in self.patch_cache:
            records = api.get_records_in_patch(patch, layer_names=["lane", "lane_connector"], mode="intersect")
            self.patch_cache[key] = sorted(set(records["lane"]+records["lane_connector"]))
        ids = self.patch_cache[key]
        missing = [token for token in ids if (location, token) not in self.polyline_cache]
        if missing:
            for token, line in api.discretize_lanes(missing, self.resolution).items():
                self.polyline_cache[(location, token)] = np.asarray(line, dtype=np.float64)[:, :2]
        positions, vectors, owners = [], [], []
        for token in ids:
            global_line = self.polyline_cache.get((location, token))
            if global_line is None or len(global_line) < 2:
                continue
            local_line = global_to_ego(global_line, origin, yaw)
            starts, delta = local_line[:-1], np.diff(local_line, axis=0)
            near = np.linalg.norm(global_line[:-1, None, :]-actors[None, :, :], axis=-1).min(axis=1) < self.radius
            valid = near & (np.linalg.norm(delta, axis=1)>1e-6)
            positions.extend(starts[valid])
            vectors.extend(delta[valid])
            owners.extend([token]*int(valid.sum()))
        starts = np.asarray(positions, dtype=np.float32).reshape(-1, 2)
        delta = np.asarray(vectors, dtype=np.float32).reshape(-1, 2)
        offsets = starts[:, None, :]-np.asarray(actor_positions)[None, :, :]
        lane, actor = np.where(np.linalg.norm(offsets, axis=-1) < self.radius)
        edges = np.stack([lane, actor]).astype(np.int64)
        return {"lane_positions": starts, "lane_vectors": delta, "lane_actor_index": edges,
                "lane_actor_vectors": offsets[lane, actor].astype(np.float32), "lane_tokens": owners,
                "radius_m": self.radius, "resolution_m": self.resolution, "location": location}
