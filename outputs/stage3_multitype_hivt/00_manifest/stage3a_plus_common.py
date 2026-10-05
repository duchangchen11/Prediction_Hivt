"""Independent Stage3A+ paths and read-only verification of the frozen Stage3A."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '00_manifest'))
from stage3_common import (PROJECT, CONFIG, CLASSES, SceneDataset, SceneSampler, config,
                           atomic_json, read_json, sha256, git, write_csv)

FREEZE = ROOT / '00_manifest/stage3a_plus_frozen_stage3a.json'
CACHE = ROOT / '02_preprocessed/stage3_cache'
BEST_MANIFEST = ROOT / '07_checkpoints/stage3a_plus_best_checkpoint_manifest.json'


def verify_freeze(data=False):
    frozen = read_json(FREEZE)
    for name, digest in frozen['files'].items():
        assert sha256(PROJECT / name) == digest, 'Frozen artifact changed: ' + name
    if data:
        for name, digest in frozen['scene_shards'].items():
            assert sha256(ROOT / name) == digest, 'Frozen shard changed: ' + name
    return frozen


def final_checkpoint():
    manifest = read_json(BEST_MANIFEST)
    assert manifest['status'] == 'COMPLETE'
    selected = manifest['final_best']
    path = ROOT / selected['relative_path']
    assert sha256(path) == selected['sha256']
    return path, manifest


def plus_manifest():
    names = ('stage3a_plus_', 'stage3_nontrivial_motion_metrics', 'stage3_interaction_density')
    path = ROOT / '00_manifest/stage3a_plus_artifact_manifest.json'
    tracked = set(git('ls-files').splitlines())
    rows = []
    for p in sorted(ROOT.rglob('*')):
        if not p.is_file() or p == path or 'stage3_cache' in p.parts or '__pycache__' in p.parts:
            continue
        if p.name.startswith(names) and not p.name.endswith('.tmp'):
            rows.append({'relative_path': str(p.relative_to(ROOT)), 'bytes': p.stat().st_size,
                         'sha256': sha256(p), 'git_tracked': str(p.relative_to(PROJECT)) in tracked})
    atomic_json(path, {'stage': 'Stage3A+', 'self_hash_excluded': True, 'artifacts': rows})
