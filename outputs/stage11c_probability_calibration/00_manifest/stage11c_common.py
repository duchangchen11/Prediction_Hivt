"""Read-only frozen ranking inputs; Stage11C scalar calibration and diagnostics."""
from pathlib import Path
from functools import lru_cache
import ast, hashlib, json, os, sys
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
S11B = ROOT.parent/'stage11b_error_aware_ranking'
S11A = ROOT.parent/'stage11a_mode_selection_audit'
S8 = ROOT.parent/'stage8a_future_scene_compatibility_graph'
S6 = ROOT.parent/'stage6a_future_interaction_reliability'
BASE = '465d3541192303e43809dc553c12e445225ab228'
sys.path.insert(0, str(S8/'00c_sparse_type_aware_graph_spec/03_model_audit'))
import numpy as np
import pandas as pd
import torch
from scipy.special import logsumexp, xlogy
from stage8a0c_model import SparseGraphReranker
PROTOCOL = ROOT/'00_manifest/stage11c_protocol.json'
REG = ROOT/'00_manifest/stage11c_registration.json'
FROZEN = ROOT/'01_frozen_audit/stage11c_frozen_manifest.json'
OOF = S11B/'05_oof_evaluation/cache'
TYPES = ('Vehicle','Pedestrian','Bicycle')
MODELS = ('R0','FoldR2','A SoftCE','B HardCE','C Raw','C Calibrated')
FORBIDDEN = [S6/'01_cache/val', S8/'03_evaluation/cache',
    ROOT.parent/'stage9a_type_adaptive_future_graph/03_evaluation/cache',
    ROOT.parent/'stage3_multitype_hivt/02_preprocessed/val',
    ROOT.parent/'stage2c_trainval_vehicle_baseline/02_preprocessed/val',
    S6/'02_features/stage6a_headdev_features.pt', S6/'02_features/stage6a_val_features.pt']
READS = set()
def guard(event, args):
    if event != 'open' or not args or not isinstance(args[0], (str,bytes,os.PathLike)): return
    path = Path(os.path.abspath(os.fsdecode(args[0])))
    mode = args[1] if len(args)>1 else None
    flags = args[2] if len(args)>2 else 0
    write = (isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and bool(flags&(os.O_WRONLY|os.O_RDWR)))
    if write and path.is_relative_to(PROJECT) and not path.is_relative_to(ROOT):
        raise RuntimeError('Stage11C historical write forbidden: '+str(path))
    if write and path.is_relative_to(ROOT) and ('.pt' in path.suffixes or '.pth' in path.suffixes):
        raise RuntimeError('Stage11C checkpoint creation forbidden')
    if any(path == p or path.is_relative_to(p) for p in FORBIDDEN) or '/02_preprocessed/test/' in str(path) or '/01_cache/test/' in str(path):
        raise RuntimeError('Stage11C forbidden data: '+str(path))
    if os.environ.get('STAGE11C_PHASE')=='fit' and path.is_relative_to(OOF):
        raise RuntimeError('Temperature fitter cannot read OOF predictions: '+str(path))
    if not write and path.is_relative_to(PROJECT): READS.add(str(path.relative_to(PROJECT)))
sys.addaudithook(guard)
def read_json(p): return json.loads(Path(p).read_text())
def sha256(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()
def array_sha(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def atomic_json(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n');t.replace(p)
def dump(relative,rows):
    p=ROOT/relative;p.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(p,index=False,float_format='%.17g')
def state_sha(m):
    h=hashlib.sha256()
    for k,v in m.state_dict().items(): h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def seed():
    torch.manual_seed(2022);np.random.seed(2022);torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True
def verify(history=False):
    frozen=read_json(FROZEN)
    for p,h in frozen['checkpoints'].items(): assert sha256(PROJECT/p)==h,p
    for p,h in frozen['split_and_normalization'].items(): assert sha256(PROJECT/p)==h,p
    assert sha256(PROTOCOL)==read_json(REG)['protocol_sha256']
    if history:
        for p,h in {**frozen['historical_files'],**frozen['preserved_untracked_files'],**frozen['data_files']}.items(): assert sha256(PROJECT/p)==h,p
    return frozen
@lru_cache(None)
def frame():
    f=pd.read_csv(S11B/'01_preflight/cache/identities.csv',dtype={'future_mask_bits':str})
    assert len(f)==260151 and f.actor_id.is_unique and f.scene_token.nunique()==630
    assert (f.HeadTrain==1).all() and (f.future_mask_bits=='111111111111').all()
    return f
@lru_cache(None)
def split(fold): return read_json(S11B/f'02_splits/stage11b_fold{fold}_split.json')
def indices(fold,part):
    assert part in ('InnerTrain','InnerDev','OuterTest')
    return np.flatnonzero(frame().scene_token.isin(split(fold)[part]).to_numpy())
# Reuse exactly the frozen Stage11B normalizer AST, without importing its
# Stage11B-only write guard or any training/optimizer helper.
NORMALIZER_SOURCE = S11B/'00_manifest/stage11b_common.py'
_tree = ast.parse(NORMALIZER_SOURCE.read_text())
_function = next(n for n in _tree.body if isinstance(n,ast.FunctionDef) and n.name=='graph_normalize')
_scope={'torch':torch}
exec(compile(ast.fix_missing_locations(ast.Module(body=[_function],type_ignores=[])),str(NORMALIZER_SOURCE),'exec'),_scope)
graph_normalize=_scope['graph_normalize']
class DevStore:
    """Only corresponding InnerDev rows; future errors never become inputs."""
    def __init__(self,fold):
        self.fold=fold;self.allowed=indices(fold,'InnerDev');self.source=frame().source_index.to_numpy()
        src=S8/'01_training/cache';self.raw=[np.load(src/f'arg{k}.npy',mmap_mode='r') for k in (0,1,2)]
        self.base=np.load(src/'arg6.npy',mmap_mode='r');self.fde=np.load(src/'fde.npy',mmap_mode='r')
        self.normpath=S11B/f'02_splits/stage11b_fold{fold}_normalization.json'
        self.norm=read_json(self.normpath)
    def batch(self,ids):
        ids=np.asarray(ids,dtype=np.int64);assert np.isin(ids,self.allowed).all()
        src=self.source[ids];raw=tuple(torch.from_numpy(np.array(a[src],copy=True)) for a in self.raw)
        n,e,mask=graph_normalize(*raw,self.norm['graph'])
        base=torch.from_numpy(np.array(self.base[src],copy=True))
        args=(n.cuda(),e.cuda(),mask.cuda(),None,None,None,base.cuda())
        return args,np.array(self.fde[src],copy=True)
def stable_logprob(z):
    z=np.asarray(z,dtype=np.float64);assert np.isfinite(z).all()
    shifted=z-z.max(-1,keepdims=True)
    return shifted-logsumexp(shifted,axis=-1,keepdims=True)
def groups(f):
    typ=f.agent_type;gt=f.GT_displacement
    return {'Overall':np.ones(len(f),bool),**{t:np.asarray(typ==t) for t in TYPES},
        **{name:np.asarray((typ=='Vehicle')&(f.motion_state==state)) for name,state in
          [('MovingVehicle','vehicle.moving'),('StoppedVehicle','vehicle.stopped'),('ParkedVehicle','vehicle.parked')]},
        'Vehicle>5m':np.asarray((typ=='Vehicle')&(gt>5)),
        'Pedestrian<5m':np.asarray((typ=='Pedestrian')&(gt<5)),
        'Pedestrian>5m':np.asarray((typ=='Pedestrian')&(gt>5)),
        'Pedestrian5-8m':np.asarray((typ=='Pedestrian')&(gt>=5)&(gt<8))}
def reliability(p,top,best,mask):
    bins=read_json(PROTOCOL)['ECE']['bins'];conf=p.max(-1);hit=(top==best)
    binid=np.minimum((conf*bins).astype(np.int64),bins-1)
    count=np.bincount(binid[mask],minlength=bins)
    totalconf=np.bincount(binid[mask],weights=conf[mask],minlength=bins)
    totalhit=np.bincount(binid[mask],weights=hit[mask],minlength=bins)
    ece=float(np.abs(totalconf-totalhit).sum()/count.sum())
    return ece,count,totalconf,totalhit,binid
