"""Stage10A isolation, frozen observation features, and HeadTrain/HeadDev stores."""
from pathlib import Path
from functools import lru_cache
import sys, os, json, csv, hashlib, time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT.parents[1]
S8 = ROOT.parent / 'stage8a_future_scene_compatibility_graph'
S6 = ROOT.parent / 'stage6a_future_interaction_reliability'
S9 = ROOT.parent / 'stage9a_type_adaptive_future_graph'
sys.path[:0] = [str(PROJECT), str(S8/'00_manifest'),
    str(S8/'00c_sparse_type_aware_graph_spec/03_model_audit'), str(S6/'00_manifest'), str(S6/'01_cache')]
import numpy as np
import torch
from stage8a_graph import observable_window, node_features, neighbors, interaction_edges, NODE_FIELDS, INTERACTION_FIELDS
from stage8a0c_model import SparseGraphReranker
from stage6a_common import frozen_predictor, SceneDataset, tensor_sha, PREDICTOR, PREDICTOR_SHA
from stage6a_cache import forward_batch
from stage6a_features import observable_features, normalize as r2normalize
from stage6a_head import ReliabilityHead
BASE = 'b4dd350e3f9488e284943e0eed9c93fd87adb8d8'
REG = ROOT/'00_manifest/stage10a_registration.json'
PROTOCOL = ROOT/'00_manifest/stage10a_protocol.json'
FROZEN = ROOT/'02_checkpoints/stage10a_checkpoint_manifest.json'
NORM = S8/'01_training/feature_normalization.json'
SPLIT = S6/'00_manifest/stage6a_head_split.json'
EXPERTS = ('vehicle', 'pedestrian')
CLASSES = ('Vehicle', 'Pedestrian', 'Bicycle')
SEEDS = {'vehicle': 2022, 'pedestrian': 2123}
CPNAMES = {'vehicle': 'VehicleExpert_best.pt', 'pedestrian': 'PedestrianExpert_best.pt'}
PARAMS = 24066
R2PATH = S6/'07_checkpoints/stage6a_r2_best.pt'
R2SHA = 'e3de457d635cd30c629ce316d73a87e419a3ded8abbe0e1f2601f1071527e314'
G1PATH = S8/'01_training/G1/formal/best_dev_loss.pt'
G1SHA = '718311dfc9f7634a8e78efe2aece9c2126b20ce87c7d34cb68b7f832c5642b0e'

# This stage cannot open official VAL/test data, even through historical helpers.
FORBIDDEN_DATA = (str(S6/'01_cache/val'), str(S8/'03_evaluation/cache'),
    str(S9/'03_evaluation/cache'), str(ROOT.parent/'stage3_multitype_hivt/02_preprocessed/val'),
    str(ROOT.parent/'stage2c_trainval_vehicle_baseline/02_preprocessed/val'))
def _data_guard(event, args):
    if event == 'open' and args and isinstance(args[0], (str, bytes, os.PathLike)):
        p = os.path.abspath(os.fsdecode(args[0]))
        if any(p == f or p.startswith(f + '/') for f in FORBIDDEN_DATA):
            raise RuntimeError('Stage10A forbids official VAL data access: ' + p)
        if '/02_preprocessed/test/' in p or '/01_cache/test/' in p:
            raise RuntimeError('Stage10A forbids test data access: ' + p)
sys.addaudithook(_data_guard)

def read_json(p): return json.loads(Path(p).read_text())
def sha256(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''): h.update(b)
    return h.hexdigest()
def atomic_json(p, x):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(x, indent=2, ensure_ascii=False, allow_nan=False)+'\n'); t.replace(p)
def write_csv(p, rows):
    assert rows
    with Path(p).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n'); w.writeheader(); w.writerows(rows)
def atomic_torch(p, x):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_suffix('.pt.tmp'); torch.save(x, t); t.replace(p)
def seed(s=2022):
    torch.manual_seed(s); np.random.seed(s); torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True); torch.backends.cudnn.deterministic=True
def state_sha(m):
    h = hashlib.sha256()
    for k,v in m.state_dict().items(): h.update(k.encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def verify(history=False):
    spec = read_json(PROTOCOL)
    assert not spec['official_VAL_permitted'] and not spec['Stage10B_execution_permitted']
    assert sha256(NORM)==spec['normalization']['sha256'] and sha256(SPLIT)==spec['split']['sha256']
    f = read_json(ROOT/'00_manifest/stage10a_frozen_history.json')
    for p,h in f['dependencies'].items(): assert sha256(PROJECT/p)==h, p
    if history:
        for p,h in f['historical_files'].items(): assert sha256(PROJECT/p)==h, p
    assert sha256(R2PATH)==R2SHA and sha256(PREDICTOR)==PREDICTOR_SHA and sha256(G1PATH)==G1SHA
    if REG.exists():
        reg = read_json(REG)
        assert sha256(PROTOCOL)==reg['protocol_sha256']
        for p,h in reg['training_sources'].items(): assert sha256(ROOT/'01_training/00_code'/p)==h, p
    return f
def frozen_r2(device='cuda'):
    assert sha256(R2PATH)==R2SHA
    m=ReliabilityHead().to(device)
    m.load_state_dict(torch.load(R2PATH, map_location='cpu', weights_only=False)['state_dict'])
    return m.eval().requires_grad_(False)
@lru_cache(None)
def graph_statistics(): return read_json(NORM)['statistics']
def normalize_graph(node, edge, mask):
    # Exact Stage8 G1 column/mask definitions; semantic arrays are never opened.
    stats=graph_statistics(); out=[node.clone(), edge.clone()]
    nv=torch.cat((torch.ones((len(mask),1), dtype=torch.bool, device=mask.device),mask),1)[...,None].expand(-1,9,6)
    ev=mask[:,None,:,None].expand(-1,6,8,6)
    for k,cols,valid in ((0,range(3,15),nv),(1,range(11),ev)):
        for c in cols:
            s=stats[str(k)][str(c)]
            out[k][...,c]=torch.where(valid,(out[k][...,c]-s['mean'])/(s['std']+1e-6),0.)
    out[0][:,1:]*=mask[...,None,None]; out[1]*=mask[:,None,:,None,None]
    assert all(torch.isfinite(x).all() for x in out)
    return out[0],out[1],mask
def graph_from_window(w, targets):
    obs=observable_window(w,'',np.zeros(2),0.)
    node=node_features(obs); idx,keep=neighbors(obs)
    edge=interaction_edges(obs,idx,keep,targets)
    local=torch.cat((node[targets,None],node[idx[targets]]),1)
    return (*normalize_graph(local,edge,keep[targets]),w['mode_logits'][targets]),idx,keep
def per_actor_loss(logits, fde):
    return -((-fde.detach()/1.).softmax(-1)*logits.log_softmax(-1)).sum(-1)
class Store:
    def __init__(self):
        self.source=S8/'01_training/cache'; self.root=ROOT/'01_training/cache'
        self.manifest=read_json(self.root/'manifest.json'); assert self.manifest['status']=='PASS'
        self.args=[np.load(self.source/f'arg{k}.npy',mmap_mode='r') for k in (0,1,2)]
        self.logits=np.load(self.source/'arg6.npy',mmap_mode='r')
        self.fde=np.load(self.source/'fde.npy',mmap_mode='r'); self.ade=np.load(self.source/'ade.npy',mmap_mode='r')
        self.partition=np.load(self.source/'partition.npy',mmap_mode='r')
        self.r2features=np.load(self.root/'r2_features.npy',mmap_mode='r')
        self.types=np.asarray(self.args[0][:,0,0,:3]).argmax(-1)
    def batch(self, ids, device='cuda'):
        raw=tuple(torch.from_numpy(np.array(a[ids],copy=True)) for a in self.args)
        args=(*normalize_graph(*raw),torch.from_numpy(np.array(self.logits[ids],copy=True)))
        fde=torch.from_numpy(np.array(self.fde[ids],copy=True)); ade=torch.from_numpy(np.array(self.ade[ids],copy=True))
        return tuple(x.to(device) for x in args),fde.to(device),ade.to(device)
    def r2_batch(self, ids, device='cuda'):
        return torch.from_numpy(np.array(self.r2features[ids],copy=True)).to(device)
def fresh(expert, device='cuda'):
    from stage10a_model import Expert
    assert expert in EXPERTS
    m=Expert(SEEDS[expert]); assert sum(p.numel() for p in m.parameters())==PARAMS
    return m.to(device)
