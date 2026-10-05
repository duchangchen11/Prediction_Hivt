"""Isolated Stage4F paths and canonical, from-scratch seed2022 initialization."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
OLD = ROOT.parent / 'stage4_type_conditioned_interaction'
STAGE3_ROOT = ROOT.parent / 'stage3_multitype_hivt'
sys.path[:0] = [str(PROJECT), str(OLD/'00_manifest'), str(OLD/'04_evaluation'),
               str(STAGE3_ROOT/'00_manifest'), str(STAGE3_ROOT/'04_evaluation')]
from stage4a_common import (CLASSES, GROUPS, HORIZONS, METRICS, MODEL_KEYS,
    SceneDataset, SceneSampler, Accumulator, errors_with_top1, loss_diagnostics,
    atomic_json, read_json, sha256, git, write_csv, seed_all, state_digest,
    model_input, evaluate, evaluate_graphs, canonical_stage3b_model)
from stage4a_common import model_new as canonical_stage4a_model
from stage4ae_common import verify_frozen as verify_older_inputs
sys.path.insert(0, str(ROOT/'00_manifest'))
from stage4f_model import HiVTNecessityGatedInteraction
import torch
import yaml

CONFIG=ROOT/'00_manifest/stage4f_config.yaml'
FREEZE=ROOT/'00_manifest/stage4f_frozen_references.json'
PREREG=ROOT/'00_manifest/stage4f_preregistration.json'
BEST=ROOT/'07_checkpoints/stage4f_best_overall_minfde.pt'
LAST=ROOT/'07_checkpoints/stage4f_last_checkpoint.pt'
SUMMARY=ROOT/'03_training/stage4f_training_summary.json'
CURVE=ROOT/'03_training/stage4f_training_curve.csv'
ACTORS=ROOT/'04_evaluation/stage4f_actor_errors.csv'


def config():
    return yaml.safe_load(CONFIG.read_text())


def verify_previous(shards=False):
    f=read_json(FREEZE)
    for name,digest in f['files'].items():
        assert sha256(PROJECT/name)==digest, 'Frozen input changed: '+name
    verify_older_inputs(shards=shards)
    return f


def model_new(device='cuda', audit=False):
    c=config()
    baseline=canonical_stage4a_model(device='cpu',audit=False)
    state={n:t.detach().clone() for n,t in baseline.state_dict().items()}
    parameters=dict(baseline.named_parameters())
    canonical_rng=torch.get_rng_state().clone()
    model=HiVTNecessityGatedInteraction(**{k:c[k] for k in MODEL_KEYS},
        pair_embedding_dim=16,relation_bias_hidden_dim=32)
    missing,unexpected=model.load_state_dict(state,strict=False)
    assert not unexpected and len(missing)==4
    assert all('necessity_gate_mlp' in n for n in missing)
    assert all(torch.equal(t,model.state_dict()[n]) for n,t in state.items())
    shared_count=sum(p.numel() for p in parameters.values())
    total=sum(p.numel() for p in model.parameters())
    assert shared_count==647609 and total==647786 and total-shared_count==177
    # The frozen Stage4A constructor already checks every Stage3B shared state.
    init={'status':'PASS','seed':2022,'max_shared_parameter_diff':0.0,
        'Stage3B_shared_parameters':646001,'Stage4A_shared_parameters':shared_count,
        'additional_gate_parameter_count':177,'additional_vs_Stage3B':1785,
        'total_parameter_count':total,'shared_state_SHA256':state_digest(state),
        'relation_step0_bitwise_equal_to_Stage4A':True,
        'relation_final_zero_initialized':True,'gate_shape':[9,16,1],
        'initial_gate':0.1,'gate_final_weight_zero':True,
        'Stage3B_trained_checkpoint_loaded':False,'Stage4A_trained_checkpoint_loaded':False,
        'training_CPU_RNG_restored_to_canonical_Stage3B':True,
        'gate_state_shapes':{n:list(model.state_dict()[n].shape) for n in missing}}
    torch.set_rng_state(canonical_rng)
    if audit:atomic_json(ROOT/'00_manifest/stage4f_initialization_audit.json',init)
    return model.to(device)
