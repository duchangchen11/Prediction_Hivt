"""Formal Stage7A paths and unchanged canonical losses/evaluation definitions."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
STAGE3 = ROOT.parent / 'stage3_multitype_hivt'
sys.path[:0] = [str(PROJECT), str(STAGE3/'00_manifest'), str(ROOT/'00_manifest')]
from stage3b_common import (SceneDataset, SceneSampler, CLASSES, GROUPS, HORIZONS,
    METRICS, MODEL_KEYS, Accumulator, errors_with_top1, loss_diagnostics,
    atomic_json, read_json, write_csv, sha256, git, seed_all, state_digest,
    GT_fingerprint, model_input, evaluate_graphs)
from stage7a_dataset import Stage7ASemanticDataset
from stage7a_model import model_new
import yaml
CONFIG = ROOT/'00_manifest/stage7a_config.yaml'
PREREG = ROOT/'00_manifest/stage7a_formal_preregistration.json'
FREEZE = ROOT/'00_manifest/stage7a_formal_frozen_previous.json'
BEST = ROOT/'07_checkpoints/stage7a_best_overall_minfde.pt'
LAST = ROOT/'07_checkpoints/stage7a_last.pt'
SUMMARY = ROOT/'03_training/stage7a_training_summary.json'
CURVE = ROOT/'03_training/stage7a_training_curve.csv'
BASE_BEST = STAGE3/'07_checkpoints/stage3b_best_overall_minfde.pt'

def config(): return yaml.safe_load(CONFIG.read_text())

def verify_previous(shards=False):
    frozen = read_json(FREEZE)
    for name, digest in {**frozen['files'], **frozen['unrelated_untracked']}.items():
        assert sha256(PROJECT/name) == digest, 'Frozen file changed: '+name
    if shards:
        for name, digest in frozen['scene_shards'].items():
            assert sha256(STAGE3/name) == digest, 'Frozen shard changed: '+name
    return frozen

# Isolated evaluation copy imports this module after its helper is generated.
from stage7a_evaluation_core import evaluate
