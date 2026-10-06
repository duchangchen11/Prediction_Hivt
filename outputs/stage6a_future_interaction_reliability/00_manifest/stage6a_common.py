"""Stage6A paths and read-only access to the frozen predictor and scene shards."""
from pathlib import Path
import sys
import hashlib
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT.parents[1]
STAGE5=ROOT.parent/'stage5a_motion_aware_decoder'
STAGE3=ROOT.parent/'stage3_multitype_hivt'
sys.path[:0]=[str(STAGE5/'00_manifest'),str(STAGE3/'00_manifest'),str(PROJECT)]
from stage5a_common import (model_new as predictor_new, model_input, SceneDataset, CLASSES,
    atomic_json, read_json, write_csv, sha256, seed_all, state_digest, git)
from stage3b_common import GT_fingerprint
import torch

BASE_COMMIT='e5cec8f44d27608eddb7820a8f873b53d0b8ddab'
PREDICTOR=STAGE5/'07_checkpoints/stage5a_best_overall_minfde.pt'
PREDICTOR_SHA='88fe3feb59b7e830ec1e40aa917484386e0d2799d0f6116f410adda2d97fdce7'
CONFIG=ROOT/'00_manifest/stage6a_config.json'
SPLIT=ROOT/'00_manifest/stage6a_head_split.json'
FREEZE=ROOT/'00_manifest/stage6a_frozen_references.json'
NORM=ROOT/'02_features/stage6a_normalization.json'
MEMBERSHIP=ROOT.parent/'stage4_type_conditioned_interaction/04_evaluation/stage4a_interaction_membership.csv'
MEMBERSHIP_AUDIT=MEMBERSHIP.with_name('stage4a_interaction_subgroup_audit.json')
BASE_ACTORS=STAGE3/'04_evaluation/stage3b_type_embedding_actor_errors.csv'
STAGE5_ACTORS=STAGE5/'04_evaluation/stage5a_actor_errors.csv'
GT_LEDGER=STAGE3/'02_preprocessed/stage3_cache/stage3b_frozen_no_type_GT_ledger.csv'
PYTHON='/home/lrj/anaconda3/envs/ped_intent/bin/python'

def config():return read_json(CONFIG)

def tensor_sha(tensor):
    t=tensor.detach().cpu().contiguous()
    h=hashlib.sha256()
    h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode());h.update(t.numpy().tobytes())
    return h.hexdigest()

def verify_frozen(shards=False):
    f=read_json(FREEZE)
    for name,digest in f['files'].items():assert sha256(PROJECT/name)==digest,name
    if shards:
        for name,digest in f['scene_shards'].items():assert sha256(STAGE3/name)==digest,name
    return f

def frozen_predictor(device='cuda'):
    assert sha256(PREDICTOR)==PREDICTOR_SHA
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic=True
    model=predictor_new(device=device)
    saved=torch.load(PREDICTOR,map_location='cpu',weights_only=False)
    model.load_state_dict(saved['state_dict'],strict=True)
    model.eval();model.requires_grad_(False)
    assert not any(p.requires_grad for p in model.parameters())
    expected=read_json(STAGE5/'03_training/stage5a_training_summary.json')['model_state_content_sha256']
    assert state_digest(model.state_dict())==expected
    return model

def atomic_torch(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');torch.save(value,temp);temp.replace(path)

def actor_key(row):return tuple(str(row[k]) for k in ('scene_token','sample_token','instance_token','horizon'))

def cache_file(split,start):return ROOT/'01_cache'/split/f'stage6a_{split}_batch_{start:05d}.pt'

def load_features(partition):
    return torch.load(ROOT/'02_features'/f'stage6a_{partition}_features.pt',map_location='cpu',weights_only=False)

def head_path(variant):return ROOT/'07_checkpoints'/f'stage6a_{variant.lower()}_best.pt'
