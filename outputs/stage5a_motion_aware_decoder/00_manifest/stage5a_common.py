"""Stage5A-only outputs; read-only Stage3B data, losses and canonical initialization."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
STAGE3_ROOT = ROOT.parent/'stage3_multitype_hivt'
sys.path[:0] = [str(PROJECT), str(STAGE3_ROOT/'00_manifest'), str(STAGE3_ROOT/'04_evaluation')]
from stage3b_common import (CLASSES, GROUPS, HORIZONS, METRICS, MODEL_KEYS, SceneDataset,
    SceneSampler, Accumulator, errors_with_top1, loss_diagnostics, atomic_json, read_json,
    sha256, git, write_csv, seed_all, state_digest, model_input, evaluate, evaluate_graphs)
from stage3b_common import model_new as canonical_stage3b_model
sys.path.insert(0, str(ROOT/'00_manifest'))
from stage5a_model import HiVTMotionAwareDecoder
import torch
import yaml

CONFIG = ROOT/'00_manifest/stage5a_config.yaml'
FREEZE = ROOT/'00_manifest/stage5a_frozen_references.json'
PREREG = ROOT/'00_manifest/stage5a_preregistration.json'
BEST = ROOT/'07_checkpoints/stage5a_best_overall_minfde.pt'
LAST = ROOT/'07_checkpoints/stage5a_last_checkpoint.pt'
SUMMARY = ROOT/'03_training/stage5a_training_summary.json'
CURVE = ROOT/'03_training/stage5a_training_curve.csv'
ACTORS = ROOT/'04_evaluation/stage5a_actor_errors.csv'
BASE_ACTORS = STAGE3_ROOT/'04_evaluation/stage3b_type_embedding_actor_errors.csv'
BASE_BEST = STAGE3_ROOT/'07_checkpoints/stage3b_best_overall_minfde.pt'
GT_LEDGER = STAGE3_ROOT/'02_preprocessed/stage3_cache/stage3b_frozen_no_type_GT_ledger.csv'


def config():
    return yaml.safe_load(CONFIG.read_text())


def verify_previous(shards=False):
    frozen = read_json(FREEZE)
    for name, digest in frozen['files'].items():
        assert sha256(PROJECT/name) == digest, 'Frozen input changed: '+name
    if shards:
        for name, digest in frozen['scene_shards'].items():
            assert sha256(STAGE3_ROOT/name) == digest, 'Frozen shard changed: '+name
    return frozen


def model_new(device='cuda', audit=False):
    c = config()
    baseline = canonical_stage3b_model(device='cpu', audit=False)
    shared = {n: t.detach().clone() for n, t in baseline.state_dict().items()}
    canonical_rng = torch.get_rng_state().clone()
    model = HiVTMotionAwareDecoder(**{k: c[k] for k in MODEL_KEYS})
    missing, unexpected = model.load_state_dict(shared, strict=False)
    assert not unexpected and len(missing) == 12
    assert all(n.startswith(('decoder.experts.', 'decoder.router.')) for n in missing)
    assert all(torch.equal(t, model.state_dict()[n]) for n,t in shared.items())
    baseline_count = sum(p.numel() for p in baseline.parameters())
    count = sum(p.numel() for p in model.parameters())
    assert (baseline_count, count, count-baseline_count) == (646001, 650403, 4402)
    init = {'status': 'PASS', 'seed': 2022, 'Stage3B_params': baseline_count,
            'Stage5A_params': count, 'Additional_params': count-baseline_count,
            'expert_params': 4256, 'router_params': 146,
            'max_shared_parameter_diff': 0.0, 'shared_buffers_bitwise_equal': True,
            'shared_state_SHA256': state_digest(shared), 'new_state_shapes': {n:list(model.state_dict()[n].shape) for n in missing},
            'initial_router': [0.5,0.5], 'expert_final_layers_zero': True,
            'source': 'canonical Stage3B seed2022 step0; fresh random weights only',
            'Stage3B_trained_checkpoint_loaded': False, 'CPU_RNG_restored_to_canonical_Stage3B': True}
    torch.set_rng_state(canonical_rng)
    if audit:
        atomic_json(ROOT/'00_manifest/stage5a_initialization_audit.json', init)
    return model.to(device)
