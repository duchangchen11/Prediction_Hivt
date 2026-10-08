"""Read-only historical models; diagnostic artifacts isolated in Stage11A."""
from pathlib import Path
import sys,os,json,csv,hashlib,time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
ROOT=Path(__file__).resolve().parents[1]; PROJECT=ROOT.parents[1]
S6=ROOT.parent/'stage6a_future_interaction_reliability'
S8=ROOT.parent/'stage8a_future_scene_compatibility_graph'
S9=ROOT.parent/'stage9a_type_adaptive_future_graph'
S10=ROOT.parent/'stage10a_type_specific_future_graph'
sys.path[:0]=[str(S10/'01_training/00_code'),str(S9/'01_training/00_code'),str(S8/'01_training/00_code'),str(PROJECT)]
import stage10a_common as legacy10
import stage8a1_common as legacy8
import stage9a_common as legacy9
from stage10a_model import Expert,DualExpert
from stage9a_model import TypeAdaptiveGraph
import numpy as np
import pandas as pd
import torch
MODELS=('R0','R2','G1','G3','T2','T3','VehicleExpert','PedestrianExpert','DualExpert')
FULL_MODELS=('R0','R2','G1','G3','T2','T3','DualExpert')
SWITCH=('G1','G3','DualExpert')
TYPES=('Vehicle','Pedestrian','Bicycle')
PROTOCOL=ROOT/'01_identity_audit/stage11a_protocol.json'
REG=ROOT/'01_identity_audit/stage11a_registration.json'
CACHE=ROOT/'01_identity_audit/cache'
METRICS=('SoftCE','Top1FDE','Top1ADE','Regret','HitRate','LogitGradL1','LogitGradL2','LogitGradSquaredL2')

def _readonly_guard(event,args):
    if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
        p=Path(os.path.abspath(os.fsdecode(args[0])))
        mode=args[1] if len(args)>1 else None; flags=args[2] if len(args)>2 else 0
        write=isinstance(mode,str) and any(c in mode for c in 'wax+')
        write=write or isinstance(flags,int) and bool(flags&(os.O_WRONLY|os.O_RDWR))
        if write and p.is_relative_to(PROJECT) and not p.is_relative_to(ROOT):
            raise RuntimeError('Stage11A forbids historical project writes: '+str(p))
        if write and p.suffix in ('.pt','.pth','.ckpt'):
            raise RuntimeError('Stage11A forbids checkpoint/cache PT writes')
sys.addaudithook(_readonly_guard)
read_json=legacy10.read_json
sha256=legacy10.sha256
atomic_json=legacy10.atomic_json
write_csv=legacy10.write_csv
tensor_sha=legacy10.tensor_sha
state_sha=legacy10.state_sha
seed=legacy10.seed

def verify(history=False):
    f=read_json(ROOT/'01_identity_audit/stage11a_frozen_history.json')
    for p,h in f['checkpoints'].items(): assert sha256(PROJECT/p)==h,p
    if history:
        for p,h in f['historical_files'].items(): assert sha256(PROJECT/p)==h,p
        for p,h in f['preserved_untracked_files'].items(): assert sha256(PROJECT/p)==h,p
    if REG.exists(): assert sha256(PROTOCOL)==read_json(REG)['protocol_sha256']
    return f
def metadata():
    return pd.read_csv(S10/'01_training/cache/identities.csv',dtype={'future_mask_bits':str})
def fde_ade():
    return tuple(np.load(S8/'01_training/cache'/n,mmap_mode='r') for n in ('fde.npy','ade.npy'))
def motion_groups(f):
    spec=read_json(PROTOCOL)
    def bins(x,edges,labels): return np.array(labels,dtype=object)[np.digitize(np.asarray(x),edges[1:],right=False)]
    result={'GTDisplacement':bins(f.GT_displacement,spec['GT_groups_edges_m'],spec['GT_groups_labels'])}
    for key,col,edges,labels in (
        ('HistoryRecentSpeed','recent_speed',spec['history_recent_speed_edges_mps'],spec['history_recent_speed_labels']),
        ('HistoryNetDisplacement','history_net',spec['history_net_path_edges_m'],spec['history_net_path_labels']),
        ('HistoryPathLength','history_path',spec['history_net_path_edges_m'],spec['history_net_path_labels'])):
        value=bins(f[col],edges,labels); value[np.asarray(f.history_valid_count)<2]='InsufficientHistory'; result[key]=value
    return result
def actor_frame():
    f=metadata(); obs=np.load(CACHE/'observed_motion.npy',mmap_mode='r')
    for j,col in enumerate(('recent_speed','history_net','history_path','history_valid_count','GT_displacement')): f[col]=obs[:,j]
    for key,value in motion_groups(f).items(): f[key]=value
    f['Partition']=np.where(f.HeadTrain==1,'HeadTrain','HeadDev')
    f['actor_id']=f.scene_token+'|'+f.sample_token+'|'+f.instance_token
    return f
def group_masks(f):
    out={'Overall':np.ones(len(f),bool),**{c:np.asarray(f.agent_type==c) for c in TYPES}}
    for name,state in (('MovingVehicle','vehicle.moving'),('StoppedVehicle','vehicle.stopped'),('ParkedVehicle','vehicle.parked')):
        out[name]=np.asarray((f.agent_type=='Vehicle')&(f.motion_state==state))
    out['OtherOrUnknownVehicleState']=np.asarray((f.agent_type=='Vehicle')&~f.motion_state.isin(['vehicle.moving','vehicle.stopped','vehicle.parked']))
    for kind in ('GTDisplacement','HistoryRecentSpeed','HistoryNetDisplacement','HistoryPathLength'):
        for typ in TYPES:
            for label in sorted(set(f.loc[f.agent_type==typ,kind])):
                out[typ+'/'+kind+'/'+label]=np.asarray((f.agent_type==typ)&(f[kind]==label))
    return out
def applicable(f,model):
    return np.asarray(f.agent_type==('Vehicle' if model=='VehicleExpert' else 'Pedestrian')) if model in ('VehicleExpert','PedestrianExpert') else np.ones(len(f),bool)
def scores(): return np.load(CACHE/'model_logits.npy',mmap_mode='r'),np.load(CACHE/'model_probabilities.npy',mmap_mode='r')
def metrics(): return np.load(CACHE/'actor_metrics.npy',mmap_mode='r')
def make_models(device='cuda'):
    frozen=read_json(ROOT/'01_identity_audit/stage11a_frozen_history.json')['checkpoints']
    def cp(path):
        relative=str(path.relative_to(PROJECT)); assert sha256(path)==frozen[relative]
        return torch.load(path,map_location='cpu',weights_only=False)['state_dict']
    r2=legacy10.frozen_r2(device)
    models={'R2':r2}
    for name in ('G1','G3'):
        m=legacy8.SparseGraphReranker(name).to(device)
        m.load_state_dict(cp(S8/'01_training'/name/'formal/best_dev_loss.pt')); models[name]=m
    for name in ('T2','T3'):
        m=TypeAdaptiveGraph(name).to(device);m.load_state_dict(cp(S9/'02_checkpoints'/f'{name}_best.pt')); models[name]=m
    experts=[]
    for key,name,sd in (('VehicleExpert','VehicleExpert_best.pt',2022),('PedestrianExpert','PedestrianExpert_best.pt',2123)):
        m=Expert(sd).to(device);m.load_state_dict(cp(S10/'02_checkpoints'/name)); models[key]=m;experts.append(m)
    models['DualExpert']=DualExpert(*experts,r2).to(device)
    for m in models.values():m.eval().requires_grad_(False)
    return models
def freeze_audit(models,before):
    assert all(not m.training and not any(p.requires_grad or p.grad is not None for p in m.parameters()) for m in models.values())
    assert before=={k:state_sha(m) for k,m in models.items()}
def logit_metrics(z,p,fde,ade):
    q=(-fde).softmax(-1); top=p.argmax(-1); best=fde.argmin(-1); ii=torch.arange(len(z),device=z.device)
    ce=-(q*z.log_softmax(-1)).sum(-1); gradient=p-q
    return torch.stack((ce,fde[ii,top],ade[ii,top],fde[ii,top]-fde[ii,best],(top==best).float(),
        gradient.abs().sum(-1),gradient.norm(dim=-1),gradient.square().sum(-1)),-1)
