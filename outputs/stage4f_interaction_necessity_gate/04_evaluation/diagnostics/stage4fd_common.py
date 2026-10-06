"""Read-only frozen-model diagnostic paths, populations and integrity checks."""
from pathlib import Path
import csv
import hashlib
import json
import sys
ROOT=Path(__file__).resolve().parents[2]
DIAG=ROOT/'04_evaluation/diagnostics'
PROJECT=ROOT.parents[1]
sys.path[:0]=[str(ROOT/'00_manifest'),str(ROOT/'04_evaluation')]
from stage4f_common import (atomic_json,read_json,write_csv,sha256,state_digest,
    canonical_stage4a_model,model_new,SceneDataset,model_input,verify_previous,OLD)
from stage4a_evaluate import read_membership,read_actors,actor_key
import numpy as np
import torch

BASE_COMMIT='514f3809e837626b6976b0f5b17be65d465ca9e6'
CHECKPOINTS={'A':(OLD/'07_checkpoints/stage4a_best_overall_minfde.pt','ff25266ba6a44630cfec01ae1596ef7de05f0d71f06f9676f6ea7ef327775ecb'),
 'F':(ROOT/'07_checkpoints/stage4f_best_overall_minfde.pt','89f2bc886bd5e1cc2a78a05e07c9761d3317f4677237a458ea35450e4f56e110')}
FREEZE=DIAG/'stage4fd_frozen_inputs.json'
RAW=DIAG/'stage4fd_raw_target_head.bin'
IDENTITY=DIAG/'stage4fd_raw_target_identity.csv'
TARGETS=DIAG/'stage4fd_target_strength.csv'
METRICS=('A_base','A_raw','F_base','F_raw','F_effective','A_shift','F_shift','A_switch','F_switch','A_Hbase','A_Hfinal','F_Hbase','F_Hfinal')
DTYPE=np.dtype([('id','<u8'),('type','u1'),('population','u1'),('flags','u1'),('motion','u1'),('displacement','<f8'),('degree','<i4'),('gate','<f8'),('values','<f8',(3,8,len(METRICS)))])
PAIRS=('V<-V','V<-P','V<-B','P<-V','P<-P','P<-B','B<-V','B<-P','B<-B')
GROUPS=('Overall','Vehicle','Pedestrian','Bicycle','vehicle.moving','vehicle.stopped','vehicle.parked','unknown',
 'Pedestrian 0-1m','Pedestrian 1-2m','Pedestrian 2-5m','Pedestrian 5-10m','Pedestrian 10-20m','Pedestrian <5m',
 'Heterogeneous-20m','non-heterogeneous','Vehicle hetero-20m','Pedestrian hetero-20m','VP-context-20m',
 'Partial targets (auxiliary)','All current-valid (auxiliary)')
MOTIONS={'unknown':0,'vehicle.moving':1,'vehicle.stopped':2,'vehicle.parked':3}

def freeze():
    assert not FREEZE.exists(),'Existing diagnostic freeze must not be overwritten'
    verify_previous(shards=True)
    files={str(p.relative_to(PROJECT)):sha256(p) for p in ROOT.rglob('stage4f_*') if p.is_file() and '__pycache__' not in p.parts and not p.name.startswith('stage4f_git_upload_')}
    for p,digest in CHECKPOINTS.values():assert sha256(p)==digest
    for p in (OLD/'04_evaluation/stage4a_interaction_membership.csv',OLD/'04_evaluation/stage4a_actor_errors.csv'):
        files[str(p.relative_to(PROJECT))]=sha256(p)
    atomic_json(FREEZE,{'status':'FROZEN','base_commit':BASE_COMMIT,'files':files,'checkpoint_sha256':{k:d for k,(p,d) in CHECKPOINTS.items()},
        'original_scientific_decisions_unchanged':{'Stage3B':'SUPPORTED','Stage4A':'NOT SUPPORTED','Stage4F':'NOT SUPPORTED','Ready_Reliability_Head':'NO'},'no_training':True})

def verify():
    f=read_json(FREEZE)
    for name,digest in f['files'].items():assert sha256(PROJECT/name)==digest,'Frozen file changed: '+name
    for p,digest in CHECKPOINTS.values():assert sha256(p)==digest,'Frozen checkpoint changed: '+str(p)
    verify_previous(shards=False)
    return f

def models(device='cpu'):
    result={'A':canonical_stage4a_model(device='cpu',audit=False),'F':model_new(device='cpu',audit=False)}
    for name,model in result.items():
        p,d=CHECKPOINTS[name];assert sha256(p)==d
        model.load_state_dict(torch.load(p,map_location='cpu',weights_only=False)['state_dict'],strict=True)
        model.eval();model.requires_grad_(False);model.to(device)
    return result

def metadata(data,membership,actors,first_id):
    ptr=data.ptr.cpu().numpy();types=data.agent_type.cpu().numpy();current=(~data.padding_mask[:,4]).cpu().numpy()
    records=np.zeros(data.num_nodes,dtype=DTYPE);identities=[]
    records['id']=np.arange(first_id,first_id+data.num_nodes);records['type']=types
    for graph in range(data.num_graphs):
        tokens=data.instance_tokens[graph];assert len(set(tokens))==len(tokens)
        for local,token in enumerate(tokens):
            node=int(ptr[graph])+local;base=(data.scene_token[graph],data.sample_token[graph],token)
            found=[membership[k] for h in ('full_horizon','partial_future') if (k:=base+(h,)) in membership]
            assert len(found)<=1
            horizon='context_only';motion=data.t0_motion_state[graph][local]
            records['displacement'][node]=np.nan
            if found:
                row=found[0];horizon=row['horizon'];key=base+(horizon,);actor=actors[key]
                assert int(row['node_in_graph'])==local and int(row['agent_type_id'])==int(types[node])
                records['population'][node]=1 if horizon=='full_horizon' else 2
                records['flags'][node]=(1 if row['Heterogeneous-20m']=='1' else 0)|(2 if row['VP-context-20m']=='1' else 0)
                records['displacement'][node]=float(actor['GT_endpoint_displacement_m'])
                assert motion==actor['motion_state'] and current[node]
            records['motion'][node]=MOTIONS.get(motion,0)
            identities.append({'id':first_id+node,'node':node,'scene_token':base[0],'sample_token':base[1],'target_instance_token':token,
                'node_in_graph':local,'type':('vehicle','pedestrian','bicycle')[types[node]],'motion_group':motion,'horizon':horizon,'current_valid':int(current[node])})
    return records,identities,current

def group_masks(records):
    full=(records['population']==1)&(records['degree']>0);p=records['type']==1;v=records['type']==0;hetero=(records['flags']&1)>0
    masks={'Overall':full,'Vehicle':full&v,'Pedestrian':full&p,'Bicycle':full&(records['type']==2)}
    for name in ('vehicle.moving','vehicle.stopped','vehicle.parked','unknown'):
        code=MOTIONS[name];masks[name]=full&v&(records['motion']==code)
    for lo,hi in ((0,1),(1,2),(2,5),(5,10),(10,20)):
        masks[f'Pedestrian {lo}-{hi}m']=full&p&(records['displacement']>=lo)&(records['displacement']<hi)
    masks.update({'Pedestrian <5m':full&p&(records['displacement']<5),'Heterogeneous-20m':full&hetero,'non-heterogeneous':full&~hetero,
        'Vehicle hetero-20m':full&v&hetero,'Pedestrian hetero-20m':full&p&hetero,'VP-context-20m':full&((records['flags']&2)>0),
        'Partial targets (auxiliary)':(records['population']==2)&(records['degree']>0),'All current-valid (auxiliary)':records['degree']>0})
    assert tuple(masks)==GROUPS
    return masks

def fingerprint(data):
    digest=hashlib.sha256()
    for name,value in sorted(data.to_dict().items()):
        digest.update(name.encode())
        if isinstance(value,torch.Tensor):
            a=value.detach().cpu().contiguous();digest.update(str((a.dtype,tuple(a.shape))).encode());digest.update(a.numpy().tobytes())
        else:digest.update(json.dumps(value,sort_keys=True,default=str).encode())
    return digest.hexdigest()

def protocol():
    atomic_json(DIAG/'stage4fd_protocol.json',{'official_VAL_scenes':150,'windows':3603,'batch_size':16,'shuffle':False,'layers':3,'heads':8,
        'primary_population':'full-horizon prediction targets; all actual incoming neighbors retained',
        'edge_pairing_population':'all current-valid nodes and actual complete directed edges, including partial/context actors',
        'topology':'GlobalInteractor complete current-valid directed graph; no self edges. LocalEncoder radius50m; gate neighbor features <=50m existing edges.',
        'EdgeWeighted':'logit metrics average actual edge-layer-head observations; attention metrics additionally degree-weight targets. GateMean degree-weighted.',
        'TargetWeighted':'first average incoming edges per target/layer/head, then equally average targets/layers/heads. Attention metrics standard equal target-layer-head mean.',
        'target_without_incoming':'excluded from undefined edge/attention statistics; identities and explicit zero-degree count retained',
        'pair_population':'full-horizon targets with every incoming source type; EdgeWeighted; Pair Layer=All or1/2/3',
        'entropy':'natural log; 0log0=0','argmax_tie_break':'first edge in unchanged formal edge order',
        'eps':1e-8,'GT_motion_bins':'offline membership only; no new model features or altered input',
        'base_attention_reference':'actual-layer base logits passed to the unchanged PyG softmax over complete incoming neighborhoods; no base-only model rollout or model selection',
        'scientific_classification':'descriptive concordance across raw/effective/normalized/attention patterns and groups/layers; no significance cutoff, no performance-decision revision',
        'no_backward_optimizer_training_selection_counterfactual_variants':True})
