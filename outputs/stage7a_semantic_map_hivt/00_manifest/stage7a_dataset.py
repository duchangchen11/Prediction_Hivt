"""Read-only Stage3 scene wrapper: attach audited token-level static semantics."""
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
STAGE3 = ROOT.parent / 'stage3_multitype_hivt'
sys.path.insert(0, str(STAGE3 / '00_manifest'))
from stage3_common import SceneDataset, read_json

SEMANTIC_FIELDS = ('is_connector', 'turn_left', 'turn_straight', 'turn_right',
                   'turn_unknown', 'traffic_light_controlled',
                   'stop_sign_controlled', 'other_control', 'crosswalk_intersects')

def semantic_vector(meta):
    connector = int(meta['is_connector'])
    controls = set(meta['control_types_present'])
    return np.array([connector, *[connector * int(meta['turn_type'] == turn)
        for turn in ('left', 'straight', 'right', 'unknown')],
        int('traffic_light' in controls), int('stop_sign' in controls),
        int(bool(controls & {'other_control', 'yield'})),
        int(meta['near_ped_crossing'])], dtype=np.float32)

class Stage7ASemanticDataset(SceneDataset):
    def __init__(self, split, cache_scenes=2, semantic_zero=False):
        super().__init__(split, cache_scenes)
        self.semantic_zero = semantic_zero
        self.metadata = read_json(ROOT / '02_semantic_cache/stage7a_semantic_metadata.json')
        self.vectors = {location: {token: semantic_vector(meta)
            for token, meta in rows.items()} for location, rows in self.metadata.items()}

    def __getitem__(self, index):
        graph = super().__getitem__(index)
        count = graph.lane_vectors.shape[0]
        assert len(graph.lane_tokens) == count
        if count and not self.semantic_zero:
            graph.lane_semantic = torch.from_numpy(np.stack([
                self.vectors[graph.map_location][token] for token in graph.lane_tokens]))
        else:
            graph.lane_semantic = torch.zeros((count, 9), dtype=torch.float32)
        assert graph.lane_semantic.shape == (count, 9)
        for field in ('is_intersections', 'turn_directions', 'traffic_controls'):
            assert not torch.count_nonzero(graph[field])
        return graph
