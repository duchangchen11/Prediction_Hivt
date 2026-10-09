"""Stage12B isolation and exact read-only frozen graph forward interfaces."""
from pathlib import Path
from functools import lru_cache
import os,sys,json,hashlib,ast,time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
ROOT=Path(__file__).resolve().parents[1];PROJECT=ROOT.parents[1]
S12A=ROOT.parent/'stage12a_candidate_semantic_audit';S11B=ROOT.parent/'stage11b_error_aware_ranking'
S11C=ROOT.parent/'stage11c_probability_calibration';S11A=ROOT.parent/'stage11a_mode_selection_audit'
S8=ROOT.parent/'stage8a_future_scene_compatibility_graph';S6=ROOT.parent/'stage6a_future_interaction_reliability'
BASE='c2ef16631f6eb67da7d98097967558e7a65925db'
PROTOCOL=ROOT/'00_manifest/stage12b_protocol.json';REG=ROOT/'00_manifest/stage12b_registration.json'
FROZEN=ROOT/'00_manifest/stage12b_frozen_history.json';CACHE=ROOT/'02_input_cache/cache'
OOF=S11B/'05_oof_evaluation/cache';RAW=S8/'01_training/cache'
TEMPERATURES={1:93.47459065453494,2:41.89311387594937,3:77.85669786831706}
FEATURES=('predicted_displacement','endpoint_x','endpoint_y','trajectory_length','trajectory_curvature','recent_speed','history_displacement','interaction_minimum_distance')
VARIANTS=('G','S','P');MODELS=('C0','G','S','P');TYPES=('Vehicle','Pedestrian','Bicycle')
sys.path[:0]=[str(PROJECT),str(S8/'00c_sparse_type_aware_graph_spec/03_model_audit'),str(S6/'00_manifest')]
import numpy as np
import pandas as pd
import torch
from stage8a0c_model import SparseGraphReranker
from stage6a_head import ReliabilityHead
READS=set();FORBIDDEN=[S6/'01_cache/val',S8/'03_evaluation/cache',ROOT.parent/'stage9a_type_adaptive_future_graph/03_evaluation/cache',
    ROOT.parent/'stage3_multitype_hivt/02_preprocessed/val',ROOT.parent/'stage2c_trainval_vehicle_baseline/02_preprocessed/val',
    S6/'02_features/stage6a_headdev_features.pt',S6/'02_features/stage6a_val_features.pt']
def guard(event,args):
    if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
    p=Path(os.path.abspath(os.fsdecode(args[0])));mode=args[1] if len(args)>1 else None;flags=args[2] if len(args)>2 else 0
    write=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and bool(flags&(os.O_WRONLY|os.O_RDWR)))
    if write and p.is_relative_to(PROJECT) and not p.is_relative_to(ROOT):raise RuntimeError('Stage12B historical write forbidden: '+str(p))
    if any(p==q or p.is_relative_to(q) for q in FORBIDDEN) or '/02_preprocessed/test/' in str(p) or '/01_cache/test/' in str(p):raise RuntimeError('Stage12B forbidden data: '+str(p))
    if os.environ.get('STAGE12B_PHASE')=='train' and (p.is_relative_to(ROOT/'06_oof_evaluation') or p.is_relative_to(OOF)):raise RuntimeError('No OOF results during residual training')
    if not write and p.is_relative_to(PROJECT):READS.add(str(p.relative_to(PROJECT)))
sys.addaudithook(guard)
def read_json(p):return json.loads(Path(p).read_text())
def sha256(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def array_sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def state_sha(m):
    h=hashlib.sha256()
    for k,v in m.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def atomic_json(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(p)
def atomic_torch(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');torch.save(x,tmp);tmp.replace(p)
def dump(relative,rows):
    p=ROOT/relative;p.parent.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).to_csv(p,index=False,float_format='%.17g')
def seed(s=2022):
    torch.manual_seed(s);np.random.seed(s);torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True
def verify(history=False,data=False):
    frozen=read_json(FROZEN);assert sha256(PROTOCOL)==read_json(REG)['ProtocolSHA256']
    for path,h in {**frozen['checkpoints'],**frozen['split_normalization_temperature']}.items():assert sha256(PROJECT/path)==h,path
    if history:
        for path,h in {**frozen['historical_files'],**frozen['preserved_untracked_files']}.items():assert sha256(PROJECT/path)==h,path
    if data:
        for path,h in frozen['data_files'].items():assert sha256(PROJECT/path)==h,path
    return frozen
@lru_cache(None)
def frame():
    f=pd.read_csv(S11B/'01_preflight/cache/identities.csv',dtype={'future_mask_bits':str})
    assert len(f)==260151 and f.actor_id.is_unique and f.scene_token.nunique()==630 and (f.HeadTrain==1).all()
    return f
@lru_cache(None)
def split(fold):return read_json(S11B/f'02_splits/stage11b_fold{fold}_split.json')
def indices(fold,part,pedestrian=False):
    assert part in ('InnerTrain','InnerDev','OuterTest');f=frame();m=f.scene_token.isin(split(fold)[part]).to_numpy()
    if pedestrian:m&=f.agent_type_id.to_numpy()==1
    return np.flatnonzero(m)
def exact_function(path,name):
    tree=ast.parse(Path(path).read_text());node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name==name)
    scope={'torch':torch};exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(path),'exec'),scope);return scope[name]
graph_normalize=exact_function(S11B/'00_manifest/stage11b_common.py','graph_normalize')
normalize_r2=exact_function(S6/'00_manifest/stage6a_features.py','normalize')
class FrozenStore:
    def __init__(self,fold):
        self.fold=fold;self.source=frame().source_index.to_numpy();self.types=frame().agent_type_id.to_numpy()
        self.raw=[np.load(RAW/f'arg{k}.npy',mmap_mode='r') for k in (0,1,2)];self.base=np.load(RAW/'arg6.npy',mmap_mode='r')
        self.r2raw=np.load(S11B/'01_preflight/cache/r2_raw.npy',mmap_mode='r');self.r2flags=np.load(S11B/'01_preflight/cache/r2_flags.npy',mmap_mode='r')
        self.normpath=S11B/f'02_splits/stage11b_fold{fold}_normalization.json';self.norm=read_json(self.normpath)
    def batch(self,ids,part):
        assert np.isin(ids,indices(self.fold,part)).all();src=self.source[ids]
        raw=tuple(torch.from_numpy(np.array(a[src],copy=True)) for a in self.raw);n,e,mask=graph_normalize(*raw,self.norm['graph'])
        base=torch.from_numpy(np.array(self.base[src],copy=True));args=(n.cuda(),e.cuda(),mask.cuda(),None,None,None,base.cuda())
        x=torch.from_numpy(np.array(self.r2raw[ids],copy=True));flags=torch.from_numpy(np.array(self.r2flags[ids],copy=True))
        return args,normalize_r2(x,flags,self.norm['R2']).cuda()
def frozen_models(fold):
    c=SparseGraphReranker('G1',2022+100*(fold-1)).cuda();r2=ReliabilityHead().cuda()
    for name,m in [('C',c),('R2',r2)]:
        cp=S11B/f'04_checkpoints/fold{fold}/{name}_best.pt';saved=torch.load(cp,map_location='cpu',weights_only=False)
        assert saved['Fold']==fold and saved['Model']==name
        assert saved['normalization_sha256']==sha256(S11B/f'02_splits/stage11b_fold{fold}_normalization.json')
        m.load_state_dict(saved['state_dict'],strict=True);m.eval().requires_grad_(False)
    return c,r2
def semantic_source():
    manifest=read_json(S12A/'02_semantic_cache/stage12a_semantic_cache_manifest.json');assert manifest['Status']=='FROZEN_ALL_630'
    folder=S12A/'03_feature_statistics/cache';merged=read_json(folder/'stage12a_combined_manifest.json')
    return manifest,{k:np.load(folder/f'stage12a_{k}.npy',mmap_mode='r') for k in ('semantic','feature_valid','geometry','candidate_geometry_id','map_region')}
def raw_observable_inputs(geometry,semantic,fields):
    """No GT, future displacement, loss, error or future mask arguments."""
    geo=fields['GeometryFields'];sem=fields['SemanticFields'];x=np.asarray(geometry[..., [geo.index(k) for k in FEATURES]],np.float64)
    valid=np.ones_like(x,bool);count=geometry[...,geo.index('history_valid_count')]
    valid[...,5:7]=count[...,None]>=2;valid[...,7]=geometry[...,geo.index('interaction_valid')]>0
    x=np.where(valid,x,0.);walk=semantic[...,sem.index('walkway_inside_fraction')];wm=semantic[...,sem.index('walkway_valid')]
    assert np.isfinite(x).all() and np.isfinite(walk).all() and ((walk>=0)&(walk<=1)).all() and np.isin(wm,[0,1]).all()
    assert not walk[wm==0].any()
    return x,valid,np.stack([walk,wm],-1).astype(np.float32)
def label_arrays():
    src=frame().source_index.to_numpy();return tuple(np.array(np.load(RAW/name,mmap_mode='r')[src],copy=True) for name in ['fde.npy','ade.npy'])
def groups(f,walk=None):
    t=f.agent_type.to_numpy();gt=f.GT_displacement.to_numpy();ped=t=='Pedestrian'
    result={'Overall':np.ones(len(f),bool),**{typ:t==typ for typ in TYPES},
        'MovingVehicle':(t=='Vehicle')&(f.motion_state.to_numpy()=='vehicle.moving'),
        'Pedestrian<5m':ped&(gt<5),'Pedestrian>5m':ped&(gt>5),'Pedestrian5-8m':ped&(gt>=5)&(gt<8)}
    speed=f.recent_speed.to_numpy();bins=np.searchsorted([.5,1.5,3],speed,side='right')
    for b in range(4):result[f'PedestrianObservedSpeed{b}']=ped&(bins==b)
    if walk is not None:
        has=np.asarray(walk[...,1]>0).any(-1)
        result.update({'Pedestrian_walkway_valid_1':ped&has,'Pedestrian_walkway_valid_0':ped&~has})
    return result
def normalized_features(fold):return np.load(CACHE/f'fold{fold}/stage12b_geometry_inputs.npy',mmap_mode='r')
def variant_input(fold,part,name,ids,device='cuda'):
    assert name in VARIANTS and np.isin(ids,indices(fold,part,True)).all()
    x=np.array(normalized_features(fold)[ids],copy=True);walk=np.load(CACHE/'stage12b_walkway.npy',mmap_mode='r')
    if name=='G':w=np.zeros((len(ids),6,2),np.float32)
    elif name=='S':w=np.array(walk[ids],copy=True)
    else:
        permutation=np.load(CACHE/f'fold{fold}/stage12b_{part}_permutation.npy',mmap_mode='r');w=np.array(walk[ids],copy=True)
        w=w[np.arange(len(ids))[:,None],permutation[ids]]
    return torch.from_numpy(np.concatenate([x,w],-1)).to(device)
