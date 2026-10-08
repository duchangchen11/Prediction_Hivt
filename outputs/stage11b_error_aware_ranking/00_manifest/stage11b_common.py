"""Fold-restricted observation inputs and unchanged G1/R2 architectures."""
from pathlib import Path
import sys,os,json,hashlib,time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
ROOT=Path(__file__).resolve().parents[1];PROJECT=ROOT.parents[1]
S6=ROOT.parent/'stage6a_future_interaction_reliability';S8=ROOT.parent/'stage8a_future_scene_compatibility_graph'
S10=ROOT.parent/'stage10a_type_specific_future_graph';S11=ROOT.parent/'stage11a_mode_selection_audit'
sys.path[:0]=[str(S10/'01_training/00_code'),str(S8/'01_training/00_code'),str(PROJECT)]
import stage10a_common as old
from stage8a0c_model import SparseGraphReranker
from stage6a_head import ReliabilityHead,ranking_loss
from stage6a_features import observable_features,normalize as normalize_r2
import numpy as np
import pandas as pd
import torch
from functools import lru_cache
read_json=old.read_json;atomic_json=old.atomic_json;sha256=old.sha256
atomic_torch=old.atomic_torch;state_sha=old.state_sha;seed=old.seed;tensor_sha=old.tensor_sha
BASE='5617f9463b3f5baa4d941f52c157c25a341fd5e1';CACHE=ROOT/'01_preflight/cache'
PROTOCOL=ROOT/'00_manifest/stage11b_protocol.json';REG=ROOT/'00_manifest/stage11b_registration.json'
PREDICTOR=old.PREDICTOR;PREDICTOR_SHA=old.PREDICTOR_SHA
TYPES=('Vehicle','Pedestrian','Bicycle');VARIANTS=('A','B','C');MODELS=('R0','R2','A','B','C')

def guard(event,args):
    if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
        path=Path(os.path.abspath(os.fsdecode(args[0])))
        mode=args[1] if len(args)>1 else None;flags=args[2] if len(args)>2 else 0
        write=(isinstance(mode,str) and any(x in mode for x in 'wax+')) or (isinstance(flags,int) and bool(flags&(os.O_WRONLY|os.O_RDWR)))
        if write and path.is_relative_to(PROJECT) and not path.is_relative_to(ROOT):raise RuntimeError('Stage11B historical write prohibited: '+str(path))
        if path in (S6/'02_features/stage6a_headdev_features.pt',S6/'02_features/stage6a_val_features.pt'):
            raise RuntimeError('Stage11B forbidden data: '+str(path))
sys.addaudithook(guard)
def dump(relative,rows):
    p=ROOT/relative;p.parent.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).to_csv(p,index=False,float_format='%.12g')
def verify(history=False):
    frozen=read_json(ROOT/'00_manifest/stage11b_frozen_history.json')
    for p,h in frozen['checkpoints'].items():assert sha256(PROJECT/p)==h,p
    if history:
        for p,h in {**frozen['historical_files'],**frozen['preserved_untracked_files']}.items():assert sha256(PROJECT/p)==h,p
    if REG.exists():assert sha256(PROTOCOL)==read_json(REG)['protocol_sha256']
    return frozen
@lru_cache(None)
def frame():return pd.read_csv(CACHE/'identities.csv',dtype={'future_mask_bits':str})
@lru_cache(None)
def split(fold):return read_json(ROOT/f'02_splits/stage11b_fold{fold}_split.json')
@lru_cache(None)
def indices(fold,part):
    f=frame();s=split(fold);assert part in ('InnerTrain','InnerDev','OuterTest')
    return np.flatnonzero(f.scene_token.isin(s[part]).to_numpy())
def graph_normalize(node,edge,mask,stats):
    raw=[node,edge];out=[node.clone(),edge.clone()]
    nv=torch.cat((torch.ones((len(mask),1),dtype=torch.bool),mask),1)[...,None].expand(-1,9,6)
    ev=mask[:,None,:,None].expand(-1,6,8,6)
    for k,cols,valid in ((0,range(3,15),nv),(1,range(11),ev)):
        for c in cols:
            stat=stats[str(k)][str(c)];out[k][...,c]=torch.where(valid,(raw[k][...,c]-stat['mean'])/(stat['std']+1e-6),0.)
    out[0][:,1:]*=mask[...,None,None];out[1]*=mask[:,None,:,None,None]
    assert all(torch.isfinite(x).all() for x in out)
    return out[0],out[1],mask
class Store:
    def __init__(self,fold):
        self.fold=fold;f=frame();self.source=f.source_index.to_numpy();self.types=f.agent_type_id.to_numpy()
        src=S8/'01_training/cache';self.args=[np.load(src/f'arg{k}.npy',mmap_mode='r') for k in (0,1,2)]
        self.base=np.load(src/'arg6.npy',mmap_mode='r');self.fde=np.load(src/'fde.npy',mmap_mode='r');self.ade=np.load(src/'ade.npy',mmap_mode='r')
        self.raw_r2=np.load(CACHE/'r2_raw.npy',mmap_mode='r');self.flags=np.load(CACHE/'r2_flags.npy',mmap_mode='r')
        self.normpath=ROOT/f'02_splits/stage11b_fold{fold}_normalization.json';self.norm=read_json(self.normpath)
    def batch(self,ids,device='cuda',part=None):
        ids=np.asarray(ids,dtype=np.int64);assert ids.ndim==1 and np.all((ids>=0)&(ids<len(self.source)))
        if part is not None:assert np.isin(ids,indices(self.fold,part)).all()
        src=self.source[ids];raw=tuple(torch.from_numpy(np.array(a[src],copy=True)) for a in self.args)
        node,edge,mask=graph_normalize(*raw,self.norm['graph'])
        base=torch.from_numpy(np.array(self.base[src],copy=True))
        args=(node.to(device),edge.to(device),mask.to(device),None,None,None,base.to(device))
        fde=torch.from_numpy(np.array(self.fde[src],copy=True)).to(device);ade=torch.from_numpy(np.array(self.ade[src],copy=True)).to(device)
        return args,fde,ade
    def r2(self,ids,device='cuda'):
        x=torch.from_numpy(np.array(self.raw_r2[ids],copy=True));flags=torch.from_numpy(np.array(self.flags[ids],copy=True))
        return normalize_r2(x,flags,self.norm['R2']).to(device)
def fresh(fold,device='cuda'):
    m=SparseGraphReranker('G1',seed=2022+100*(fold-1));assert sum(p.numel() for p in m.parameters())==24066
    return m.to(device)
def objective(logits,fde,variant):
    errors=fde.detach();p=logits.softmax(-1)
    if variant=='A':return -((-errors).softmax(-1)*logits.log_softmax(-1)).sum(-1)
    if variant=='B':return -logits.log_softmax(-1).gather(-1,errors.argmin(-1,keepdim=True)).squeeze(-1)
    assert variant=='C';c=errors-errors.min(-1,keepdim=True).values;scale=c.mean(-1).clamp(min=1.)
    return (p*(c/scale[:,None])).sum(-1)
def model_forward(m,args):return m(*args)
def groups(f):
    typ=f.agent_type;gt=f.GT_displacement
    return {'Overall':np.ones(len(f),bool),**{t:np.asarray(typ==t) for t in TYPES},
        **{name:np.asarray((typ=='Vehicle')&(f.motion_state==state)) for name,state in
            [('MovingVehicle','vehicle.moving'),('StoppedVehicle','vehicle.stopped'),('ParkedVehicle','vehicle.parked')]},
        'Vehicle>5m':np.asarray((typ=='Vehicle')&(gt>5)),
        'Pedestrian<5m':np.asarray((typ=='Pedestrian')&(gt<5)),
        'Pedestrian>5m':np.asarray((typ=='Pedestrian')&(gt>5)),
        'Pedestrian5-8m':np.asarray((typ=='Pedestrian')&(gt>=5)&(gt<8))}
