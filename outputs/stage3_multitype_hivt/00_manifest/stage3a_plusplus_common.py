"""Independent paths and read-only checks for the final No-Type budget."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '00_manifest'))
sys.path.insert(0, str(ROOT / '03_no_type_baseline'))
from stage3_common import (PROJECT, CONFIG, CLASSES, SceneDataset, SceneSampler,
                           config, atomic_json, read_json, sha256, git, write_csv)
from stage3a_plus_finalize_checkpoint import model_digest, state_equal

SOURCE = ROOT / '07_checkpoints/stage3a_plus_best_overall_minfde.pt'
LAST = ROOT / '07_checkpoints/stage3a_plusplus_last_checkpoint.pt'
NEW_BEST = ROOT / '07_checkpoints/stage3a_final_frozen_best_overall_minfde.pt'
TRAIN_MANIFEST = ROOT / '07_checkpoints/stage3a_plusplus_training_manifest.json'
FINAL_MANIFEST = ROOT / '07_checkpoints/stage3a_final_frozen_checkpoint_manifest.json'
FREEZE = ROOT / '00_manifest/stage3a_plusplus_frozen_previous.json'
PREREG = ROOT / '00_manifest/stage3a_plusplus_preregistration.json'
CURVE = ROOT / '03_no_type_baseline/stage3a_plusplus_nll_curve.csv'

def verify_previous(shards=False):
    frozen = read_json(FREEZE)
    for name, digest in frozen['files'].items():
        assert sha256(PROJECT / name) == digest, 'Protected artifact changed: ' + name
    if shards:
        for name, digest in frozen['scene_shards'].items():
            assert sha256(ROOT / name) == digest, 'Protected shard changed: ' + name
    return frozen
