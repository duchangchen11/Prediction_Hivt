"""Read-only frozen predictors and isolated Stage7C diagnostic paths."""
from pathlib import Path
import sys, json, hashlib, subprocess
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
STAGE7A = ROOT.parent / 'stage7a_semantic_map_hivt'
STAGE3 = ROOT.parent / 'stage3_multitype_hivt'
sys.path[:0] = [str(PROJECT), str(STAGE7A/'00_manifest'), str(STAGE3/'00_manifest')]
from stage7a_dataset import Stage7ASemanticDataset, SEMANTIC_FIELDS
from stage7a_model import model_new as semantic_new
from stage3b_common import (model_new as baseline_new, model_input, errors_with_top1,
    GT_fingerprint, state_digest, CLASSES, sha256, atomic_json, read_json)
BASE = STAGE3/'07_checkpoints/stage3b_best_overall_minfde.pt'
BEST = STAGE7A/'07_checkpoints/stage7a_best_overall_minfde.pt'
BASE_SHA = '461dd9fc8ccc4a03bb72e6a92f3e328e34e1fefb6ebf890ff87eedffaa718547'
BEST_SHA = 'e30b7dbe7260ef5765e10f95d5eadf58e42f480e79f15b58a3b149a9d271abc1'
SOURCE_COMMIT = '77e77b3e1f48a5e68e48d5665308232b97d058f4'
BRANCH = 'stage7c/actor-conditioned-semantic-relevance-audit'
IDS = ['scene_token','sample_token','instance_token']

def load_model(name):
    assert name in ('Stage3B','Stage7A')
    path,expected = (BASE,BASE_SHA) if name=='Stage3B' else (BEST,BEST_SHA)
    assert sha256(path)==expected
    model=(baseline_new if name=='Stage3B' else semantic_new)()
    cp=torch.load(path,map_location='cpu',weights_only=False)
    model.load_state_dict(cp['state_dict'],strict=True)
    del cp
    model.eval().requires_grad_(False)
    return model

def input_for(data, name):
    working=model_input(data)
    if name=='Stage3B':
        del working.lane_semantic
        assert 'lane_semantic' not in working
    return working

def old_errors(name):
    key={'Stage3B':'baseline','Stage7A-ON':'on','Stage7A-ZERO':'zero'}[name]
    f=pd.read_csv(STAGE7A/f'04_evaluation/stage7a_{key}_actor_errors.csv',dtype={'future_mask_bits':str})
    f=f[f.horizon=='full_horizon'].copy()
    assert len(f)==54990 and not f.duplicated(IDS).any()
    return f

def verify_frozen(shards=False):
    frozen=read_json(ROOT/'00_manifest/stage7c_frozen_inputs.json')
    for name,digest in frozen['files'].items():
        assert sha256(PROJECT/name)==digest, 'Frozen input changed: '+name
    if shards:
        for name,digest in frozen['scene_shards'].items():
            assert sha256(STAGE3/name)==digest, 'Frozen scene shard changed: '+name
    return {'files':len(frozen['files']),'scene_shards':len(frozen['scene_shards']) if shards else 0}

def paired_errors():
    b=old_errors('Stage3B').set_index(IDS).sort_index()
    o=old_errors('Stage7A-ON').set_index(IDS).sort_index()
    assert b.index.equals(o.index)
    for k in ['GT_trajectory_sha256','future_mask_bits','agent_type_id','node_in_graph']:
        assert (b[k].to_numpy()==o[k].to_numpy()).all()
    f=b.reset_index()
    for k in ['minFDE6','Top1FDE6']:
        f['Stage3B_'+k]=b[k].to_numpy();f['Stage7A_'+k]=o[k].to_numpy()
    f['DeltaMinFDE']=f.Stage7A_minFDE6-f.Stage3B_minFDE6
    f['DeltaTop1FDE']=f.Stage7A_Top1FDE6-f.Stage3B_Top1FDE6
    return f

def groups(f):
    v=f.agent_type=='vehicle';turn=f.TurningVehicle_GT==1
    return {'Overall':np.ones(len(f),bool),'Overall Vehicle':v.to_numpy(),
        'vehicle.moving':(v&(f.motion_state=='vehicle.moving')).to_numpy(),
        'Vehicle >5m':(v&(f.GT_endpoint_displacement_m>5)).to_numpy(),
        'TurningVehicle_GT':turn.to_numpy(),
        'GT-left':(turn&(f.GT_heading_change_deg>0)).to_numpy(),
        'GT-right':(turn&(f.GT_heading_change_deg<0)).to_numpy(),
        'TurnOptionCount20=0':(v&(f.TurnOptionCount20==0)).to_numpy(),
        'TurnOptionCount20=1':(v&(f.TurnOptionCount20==1)).to_numpy(),
        'TurnOptionCount20=2':(v&(f.TurnOptionCount20==2)).to_numpy(),
        'TurnOptionCount20>=3':(v&(f.TurnOptionCount20>=3)).to_numpy(),
        'Pedestrian':(f.agent_type=='pedestrian').to_numpy()}
