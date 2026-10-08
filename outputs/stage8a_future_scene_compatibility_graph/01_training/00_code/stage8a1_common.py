"""Stage8A-1 isolation and frozen 0C dependencies. No fitting in graph builders."""
from pathlib import Path
import sys, os, json, csv, hashlib, time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT=Path(__file__).resolve().parents[2]; PROJECT=ROOT.parents[1]
C=ROOT/'00c_sparse_type_aware_graph_spec'; S6=ROOT.parent/'stage6a_future_interaction_reliability'
sys.path[:0]=[str(C/'00_manifest'),str(C/'01_selector'),str(C/'03_model_audit'),str(PROJECT)]
from stage8a0c_common import atomic_json,read_json,sha256,write_csv,atomic_npz,ego_to_global,TYPES
from stage8a0c_common import observable_window,node_features,neighbors,interaction_edges
from stage8a0c_selector import SparseSemanticIndex
from stage8a0c_model import SparseGraphReranker,pack_targets,interaction_attention
import numpy as np
import torch
BASE='df28e8638a1901422d7cded42430a32ea580f02c'
SPEC=C/'00_manifest/stage8a0c_frozen_graph_spec.json'
SPLIT=S6/'00_manifest/stage6a_head_split.json'
NORM=ROOT/'01_training/feature_normalization.json'
FROZEN=ROOT/'02_checkpoints/stage8a1_checkpoint_manifest.json'
PARAMS={'G1':24066,'G2':20546,'G3':37187}
SHAPES=[(9,6,15),(6,8,6,17),(8,),(6,8,18),(6,8,11),(6,8),(6,)]
DTYPES=['float32','float32','bool','float32','float32','bool','float32']
CONT={0:list(range(3,15)),1:list(range(11)),3:[15,16],4:[0,1,2,3,4,5,6,7,9]}

def seed():
    torch.manual_seed(2022);np.random.seed(2022)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic=True

def verify():
    spec=read_json(SPEC);assert spec['GraphSpecFrozen']=='YES'
    for p,h in spec['source_SHA256'].items():assert sha256(PROJECT/p)==h,p
    return spec

def atomic_torch(p,x):
    p=Path(p);tmp=p.with_suffix('.pt.tmp');torch.save(x,tmp);tmp.replace(p)

def state_sha(model):
    h=hashlib.sha256()
    for k,v in model.state_dict().items():h.update(k.encode());h.update(v.cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def graph_window(w,rec,j,index,sel,frame):
    """Observation allowlist only; supervision is joined by caller afterwards."""
    a,b=map(int,sel['node_offset'][j:j+2]);selected=sel['actor_node'][a:b]
    loc=sorted(index.regions)[int(frame['window_location'][j])]
    obs=observable_window(w,loc,frame['window_origin'][j],float(frame['window_yaw'][j]))
    node=node_features(obs);idx,keep=neighbors(obs)
    selection={k:sel[k][a:b].reshape((-1,)+sel[k].shape[2:]) for k in ('entity_ids','entity_types','geometry_distances','map_mask')}
    local=obs.predicted[selected].numpy().reshape(-1,12,2)
    points=ego_to_global(local,obs.origin,obs.yaw);delta=local[:,-1]-local[:,-3]
    feature=index.features(points,loc,obs.yaw,np.arctan2(delta[:,1],delta[:,0]),selection)
    feature={k:v.reshape((len(selected),6)+v.shape[1:]) for k,v in feature.items()}
    feature['map_mask']=sel['map_mask'][a:b]
    feature['entity_ids']=sel['entity_ids'][a:b]
    return obs,selected,node,idx,keep,feature

def valid_columns(args):
    node_valid=torch.cat((torch.ones((len(args[2]),1),dtype=torch.bool,device=args[2].device),args[2]),1)[...,None].expand(-1,9,6)
    inter_valid=args[2][:,None,:,None].expand(-1,6,8,6)
    masks={0:node_valid,1:inter_valid,3:args[5],4:args[5]}
    out={}
    for k,cols in CONT.items():
        for c in cols:
            mask=masks[k]
            if k==3:mask=mask&(args[3][...,17]==1)
            if k==4 and c in (6,7):mask=mask&(args[4][...,8]==1)
            if k==4 and c==9:mask=mask&(args[4][...,8]==0)
            out[k,c]=mask
    return out

def normalize_args(args,stats):
    out=[a.clone() for a in args];masks=valid_columns(args)
    for (k,c),mask in masks.items():
        s=stats[str(k)][str(c)];out[k][...,c]=torch.where(mask,(args[k][...,c]-s['mean'])/(s['std']+1e-6),0.)
    # Padding raw one-hots were originally taken from actor index0: eliminate
    # only padded neighbor node rows, which are never attended by frozen 0C.
    out[0][:,1:]*=args[2][...,None,None]
    out[1]*=args[2][:,None,:,None,None]
    for x in out:assert torch.isfinite(x).all()
    return tuple(out)

class Store:
    def __init__(self,split):
        self.root=ROOT/('01_training/cache' if split=='train' else '03_evaluation/cache')
        self.manifest=read_json(self.root/'manifest.json');assert self.manifest['status']=='PASS'
        self.args=[np.load(self.root/f'arg{k}.npy',mmap_mode='r') for k in range(7)]
        self.fde=np.load(self.root/'fde.npy',mmap_mode='r');self.ade=np.load(self.root/'ade.npy',mmap_mode='r')
        self.partition=np.load(self.root/'partition.npy',mmap_mode='r')
        self.stats=read_json(NORM)['statistics']
    def batch(self,ids,device='cuda'):
        a=tuple(torch.from_numpy(np.array(x[ids],copy=True)) for x in self.args)
        a=normalize_args(a,self.stats)
        return tuple(x.to(device) for x in a),torch.from_numpy(np.array(self.fde[ids],copy=True)).to(device),torch.from_numpy(np.array(self.ade[ids],copy=True)).to(device)

def loss(logits,fde):return -((-fde.detach()).softmax(-1)*logits.log_softmax(-1)).sum(-1).mean()

def fresh(variant,device='cuda'):
    m=SparseGraphReranker(variant);assert sum(p.numel() for p in m.parameters())==PARAMS[variant]
    return m.to(device)

def capture(model,args):
    """Read-only hooks on scalar scores; outputs computed by frozen forward."""
    result={};hooks=[]
    if model.variant in ('G1','G3'):
        def ih(_,inp,out):
            score=out.squeeze(-1).reshape(-1,6,48)
            mask=args[2][:,None,:,None].expand(-1,6,8,6).reshape(-1,6,48)
            result['interaction_alpha']=interaction_attention(score,mask).detach().cpu()
        hooks.append(model.interaction_attention.register_forward_hook(ih))
    if model.variant in ('G2','G3'):
        def mh(_,inp,out):
            mask=args[5].reshape(-1,8);r,s=mask.nonzero(as_tuple=True)
            scores=out.new_full(mask.shape,float('-inf'));scores[r,s]=out.squeeze(-1)
            alpha=out.new_zeros(mask.shape);active=mask.any(-1);alpha[active]=scores[active].softmax(-1)
            result['map_alpha']=alpha.reshape(-1,6,8).detach().cpu()
        hooks.append(model.map_aggregator.attention.register_forward_hook(mh))
    try:out=model(*args)
    finally:
        for h in hooks:h.remove()
    if model.variant in ('G2','G3') and 'map_alpha' not in result:result['map_alpha']=torch.zeros_like(args[5],dtype=torch.float32).cpu()
    return out,result
