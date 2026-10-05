"""Frozen-input protection and shared read-only utilities for Stage4A-E."""
import csv
import hashlib
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
STAGE3 = ROOT.parent / 'stage3_multitype_hivt'
DIAG = ROOT / '04_evaluation/diagnostics'
sys.path[:0] = [str(ROOT / '00_manifest'), str(ROOT / '04_evaluation'),
    str(ROOT / '04_evaluation/diagnostics'), str(STAGE3 / '00_manifest')]
from stage4a_common import (atomic_json, read_json, write_csv, sha256, state_digest,
    SceneDataset, seed_all, verify_previous, model_input)
from stage4a_evaluate import (read_actors, read_membership, actor_key, summarize,
    BASE_ACTORS, ACTORS, MEMBERSHIP, BEST)
import numpy as np

BASE_COMMIT = 'f9619f7a442933cefd9f5bdf9704dff1212b0ab0'
BRANCH = 'stage4a-e/motion-gradient-diagnostic'
FREEZE = ROOT / '00_manifest/stage4ae_frozen_inputs.json'
BEST_B = STAGE3 / '07_checkpoints/stage3b_best_overall_minfde.pt'
CHECKPOINT_C = 'ff25266ba6a44630cfec01ae1596ef7de05f0d71f06f9676f6ea7ef327775ecb'
CHECKPOINT_B = '461dd9fc8ccc4a03bb72e6a92f3e328e34e1fefb6ebf890ff87eedffaa718547'
BINS = ((0, 1, 3192), (1, 2, 313), (2, 5, 916), (5, 10, 7321), (10, 20, 260))
FIELDS = ('minADE6', 'minFDE6', 'MR6', 'Top1ADE6', 'Top1FDE6', 'NLL')


def git(*args):
    return subprocess.check_output(['git', *args], cwd=PROJECT).decode().strip()


def read_csv(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def freeze_inputs():
    assert git('rev-parse', 'HEAD') == BASE_COMMIT and git('branch', '--show-current') == BRANCH
    assert not FREEZE.exists(), 'Keep the original frozen-input ledger'
    verify_previous(shards=True)
    assert sha256(BEST) == CHECKPOINT_C and sha256(BEST_B) == CHECKPOINT_B
    files = {}
    for p in ROOT.rglob('*'):
        if p.is_file() and p.name.startswith(('stage4a_', 'stage4ad_')):
            if {'__pycache__', 'stage4a_cache'}.intersection(p.parts):
                continue
            files[str(p.relative_to(PROJECT))] = sha256(p)
    for p in (BEST_B, BASE_ACTORS, STAGE3 / '03_type_embedding/stage3b_training_summary.json'):
        files[str(p.relative_to(PROJECT))] = sha256(p)
    for p in (PROJECT / 'outputs/stage2c_trainval_vehicle_baseline/04_evaluation').glob('stage2c_qualitative_main_case_new*'):
        if p.is_file():
            files[str(p.relative_to(PROJECT))] = sha256(p)
    atomic_json(FREEZE, {'status': 'FROZEN', 'base_commit': BASE_COMMIT, 'branch': BRANCH,
        'files': files, 'checkpoint_B_sha256': CHECKPOINT_B, 'checkpoint_C_sha256': CHECKPOINT_C,
        'fixed_motion_bins_m': BINS, 'batch_size': 16, 'gradient_batches': 100, 'seed': 2022,
        'loader_order': 'SceneDataset train index order; shuffle=False; workers=0',
        'class_loss': 'unchanged recovery_loss original_nll with class-only target_mask; full context graph',
        'requirements_sha256': sha256(ROOT / '00_manifest/stage4ae_requirements.txt'),
        'no_parameter_update': True})
    print('STAGE4AE_FROZEN', len(files), flush=True)


def verify_frozen(shards=False):
    frozen = read_json(FREEZE)
    for name, digest in frozen['files'].items():
        assert sha256(PROJECT / name) == digest, 'Frozen file changed: ' + name
    verify_previous(shards=shards)
    return frozen


def paired_actors():
    audit = read_json(ROOT / '04_evaluation/stage4a_pairing_audit.json')
    assert audit['status'] == 'PASS' and audit['paired_full_horizon_actors'] == 54990
    assert sha256(BASE_ACTORS) == audit['B_actor_errors_sha256']
    assert sha256(ACTORS) == audit['C_actor_errors_sha256']
    assert sha256(MEMBERSHIP) == audit['interaction_membership_sha256']
    b, c = read_actors(BASE_ACTORS), read_actors(ACTORS)
    old, new = {actor_key(r): r for r in b}, {actor_key(r): r for r in c}
    membership = read_membership()
    assert old.keys() == new.keys() == membership.keys()
    identity = ('node_in_graph', 'agent_type', 'agent_type_id', 'motion_state',
        'valid_future_steps', 'GT_trajectory_sha256', 'future_mask_bits', 'GT_endpoint_displacement_m')
    for key, row in old.items():
        assert all(row[k] == new[key][k] for k in identity), key
    full = [key for key in old if key[-1] == 'full_horizon']
    assert len(full) == 54990
    pedestrians = [key for key in full if old[key]['agent_type'] == 'pedestrian']
    assert len(pedestrians) == 12002
    selected = {}
    for lower, upper, count in BINS:
        name = f'{lower}-{upper}m'
        selected[name] = [key for key in pedestrians if lower <= float(old[key]['GT_endpoint_displacement_m']) < upper]
        assert len(selected[name]) == count, ('STOP: motion-bin count mismatch', name, len(selected[name]), count)
    assert sum(map(len, selected.values())) == 12002
    assert set().union(*map(set, selected.values())) == set(pedestrians)
    return old, new, membership, selected


if __name__ == '__main__':
    if sys.argv[1] == 'freeze':
        freeze_inputs()
    elif sys.argv[1] == 'verify':
        verify_frozen(shards=True)
