"""Stage9 observation-only future interactions; historical dependencies read-only."""
from pathlib import Path
import sys,os,json,csv,hashlib,time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
ROOT=Path(__file__).resolve().parents[2];PROJECT=ROOT.parents[1]
S8=ROOT.parent/'stage8a_future_scene_compatibility_graph'
S6=ROOT.parent/'stage6a_future_interaction_reliability'
sys.path[:0]=[str(PROJECT),str(S8/'00_manifest'),str(S8/'00c_sparse_type_aware_graph_spec/03_model_audit'),str(S6/'00_manifest'),str(S6/'01_cache')]
import numpy as np
import torch
from stage8a_graph import observable_window,node_features,neighbors,interaction_edges,NODE_FIELDS,INTERACTION_FIELDS
from stage8a0c_model import SparseGraphReranker,interaction_attention
from stage6a_common import frozen_predictor,SceneDataset,tensor_sha
from stage6a_cache import forward_batch
from stage6a_features import observable_features,normalize as r2normalize
from stage6a_head import ReliabilityHead
BASE='31aa16dff59f393d6e2e3337e5861755ab069aec'
REG=ROOT/'00_manifest/stage9a_registration.json'
FROZEN=ROOT/'02_checkpoints/stage9a_checkpoint_manifest.json'
NORM=S8/'01_training/feature_normalization.json'
SPLIT=S6/'00_manifest/stage6a_head_split.json'
VARIANTS=('T1','T2','T3');CLASSES=('Vehicle','Pedestrian','Bicycle')
PARAMS={'T1':24066,'T2':27515,'T3':27515}
R2PATH=S6/'07_checkpoints/stage6a_r2_best.pt'
R2SHA='e3de457d635cd30c629ce316d73a87e419a3ded8abbe0e1f2601f1071527e314'

def read_json(p):return json.loads(Path(p).read_text())
def sha256(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def atomic_json(p,x):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(p)
def write_csv(p,rows):
    assert rows
    with Path(p).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def atomic_torch(p,x):
    p=Path(p);tmp=p.with_suffix('.pt.tmp');torch.save(x,tmp);tmp.replace(p)
def seed():
    torch.manual_seed(2022);np.random.seed(2022);torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True
def state_sha(model):
    h=hashlib.sha256()
    for k,v in model.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def verify():
    f=read_json(ROOT/'00_manifest/stage9a_frozen_history.json')
    for p,h in f['dependencies'].items():assert sha256(PROJECT/p)==h,p
    assert sha256(R2PATH)==R2SHA
    return f
def frozen_r2(device='cuda'):
    assert sha256(R2PATH)==R2SHA
    m=ReliabilityHead().to(device);m.load_state_dict(torch.load(R2PATH,map_location='cpu',weights_only=False)['state_dict'])
    return m.eval().requires_grad_(False)
def normalize_graph(node,edge,mask):
    stats=read_json(NORM)['statistics'];out=[node.clone(),edge.clone()]
    nv=torch.cat((torch.ones((len(mask),1),dtype=torch.bool,device=mask.device),mask),1)[...,None].expand(-1,9,6)
    ev=mask[:,None,:,None].expand(-1,6,8,6)
    for k,cols,valid in ((0,range(3,15),nv),(1,range(11),ev)):
        for c in cols:
            s=stats[str(k)][str(c)];out[k][...,c]=torch.where(valid,(out[k][...,c]-s['mean'])/(s['std']+1e-6),0.)
    out[0][:,1:]*=mask[...,None,None];out[1]*=mask[:,None,:,None,None]
    assert all(torch.isfinite(x).all() for x in out)
    return out[0],out[1],mask
def context_from_window(w,idx,keep,targets):
    count=keep[targets].sum(-1).float()/8.
    d=(w['history'][targets,4,None].double()-w['history'][idx[targets],4].double()).norm(dim=-1)
    nearest=d.masked_fill(~keep[targets],float('inf')).min(-1).values/50.
    nearest=torch.where(keep[targets].any(-1),nearest,torch.ones_like(nearest)).clamp(0,1).float()
    return torch.stack((count,nearest),-1)
def gate_features(normalized_node,context):
    out=torch.cat((normalized_node[:,0,0,:3],normalized_node[:,0,0,5:8],context),-1)
    assert out.shape==(len(context),8) and torch.isfinite(out).all()
    return out
def graph_from_window(w,targets):
    # Map coordinates are inert placeholders: no map/semantic object is created.
    obs=observable_window(w,'',np.zeros(2),0.);node=node_features(obs);idx,keep=neighbors(obs)
    edge=interaction_edges(obs,idx,keep,targets)
    local=torch.cat((node[targets,None],node[idx[targets]]),1)
    args=normalize_graph(local,edge,keep[targets]);context=context_from_window(w,idx,keep,targets)
    return args,gate_features(args[0],context),context,idx,keep
def per_actor_loss(logits,fde):return -((-fde.detach()).softmax(-1)*logits.log_softmax(-1)).sum(-1)
def objective(values,types,balanced):
    micro=values.mean();macro=torch.stack([values[types==t].mean() for t in range(3) if bool((types==t).any())]).mean()
    return (.5*micro+.5*macro if balanced else micro),micro,macro

class Store:
    def __init__(self):
        self.source=S8/'01_training/cache';self.root=ROOT/'01_training/cache'
        self.manifest=read_json(self.root/'manifest.json');assert self.manifest['status']=='PASS'
        # Only future-interaction tensors are opened; semantic arrays never read.
        self.args=[np.load(self.source/f'arg{k}.npy',mmap_mode='r') for k in (0,1,2)]
        self.fde=np.load(self.source/'fde.npy',mmap_mode='r');self.ade=np.load(self.source/'ade.npy',mmap_mode='r')
        self.partition=np.load(self.source/'partition.npy',mmap_mode='r')
        self.base=np.load(self.root/'r2_logits.npy',mmap_mode='r');self.context=np.load(self.root/'current_context.npy',mmap_mode='r')
        self.types=np.asarray(self.args[0][:,0,0,:3]).argmax(-1)
    def batch(self,ids,device='cuda'):
        raw=tuple(torch.from_numpy(np.array(a[ids],copy=True)) for a in self.args)
        args=normalize_graph(*raw);context=torch.from_numpy(np.array(self.context[ids],copy=True))
        gate=gate_features(args[0],context);base=torch.from_numpy(np.array(self.base[ids],copy=True))
        fde=torch.from_numpy(np.array(self.fde[ids],copy=True));ade=torch.from_numpy(np.array(self.ade[ids],copy=True))
        return tuple(a.to(device) for a in (*args,base,gate)),fde.to(device),ade.to(device)

def fresh(variant,device='cuda'):
    from stage9a_model import TypeAdaptiveGraph
    m=TypeAdaptiveGraph(variant);assert sum(p.numel() for p in m.parameters())==PARAMS[variant]
    return m.to(device)
