"""Stage14B isolation and exact historical data/loss definitions, without old hooks."""
from pathlib import Path
from functools import lru_cache
import os, sys, json, hashlib, ast, time, subprocess
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
ROOT=Path(__file__).resolve().parents[1]; PROJECT=ROOT.parents[1]
BASE='4daa4ae82557270e4f43881fa22c21e3ba6c4068'
S11B=ROOT.parent/'stage11b_error_aware_ranking'; S11A=ROOT.parent/'stage11a_mode_selection_audit'
S8=ROOT.parent/'stage8a_future_scene_compatibility_graph'; S6=ROOT.parent/'stage6a_future_interaction_reliability'
S13=ROOT.parent/'stage13a_planning_feasibility'
S14A=ROOT.parent/'stage14a_paper_graph_ablation'
RAW=S8/'01_training/cache'; OOF=S11B/'05_oof_evaluation/cache'
PROTOCOL=ROOT/'00_protocol/stage14b_protocol.json'; REG=ROOT/'00_protocol/stage14b_registration.json'
FROZEN=ROOT/'00_protocol/stage14b_frozen_history.json'
VARIANTS=('C',); MODELS=('NG-A','NG-C','G-A','G-C','Matched-NG-C'); TYPES=('Vehicle','Pedestrian','Bicycle')
sys.path[:0]=[str(PROJECT),str(S8/'00c_sparse_type_aware_graph_spec/03_model_audit'),str(S6/'00_manifest'),str(ROOT/'02_models')]
import numpy as np
import pandas as pd
import torch
from stage8a0c_model import SparseGraphReranker
from stage6a_common import frozen_predictor,PREDICTOR,PREDICTOR_SHA
from stage6a_head import ReliabilityHead
from stage6a_features import normalize as normalize_r2
READS=set()
def guard(event,args):
    if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
    p=Path(os.path.abspath(os.fsdecode(args[0])));mode=args[1] if len(args)>1 else None;flags=args[2] if len(args)>2 else 0
    write=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and bool(flags&(os.O_WRONLY|os.O_RDWR)))
    if write and p.is_relative_to(PROJECT) and not p.is_relative_to(ROOT):raise RuntimeError('Stage14B historical write forbidden: '+str(p))
    if any(s in str(p) for s in ['/02_preprocessed/val/','/01_cache/val/','/02_preprocessed/test/','/01_cache/test/']) or p in (S6/'02_features/stage6a_headdev_features.pt',S6/'02_features/stage6a_val_features.pt'):raise RuntimeError('Stage14B forbidden evaluation data: '+str(p))
    if os.environ.get('STAGE14B_PHASE') in ('tiny','train') and (p.is_relative_to(OOF) or p.is_relative_to(ROOT/'05_evaluation') or p.is_relative_to(ROOT.parent/'stage14a_paper_graph_ablation/05_evaluation/cache')):raise RuntimeError('OuterTest evaluation forbidden until all3 checkpoints frozen')
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
def git(*args):return subprocess.check_output(['git',*args],cwd=PROJECT,text=True).strip()
def verify(history=False,data=False):
    f=read_json(FROZEN)
    for p,h in {**f['checkpoints'],**f['split_normalization']}.items():assert sha256(PROJECT/p)==h,p
    if REG.exists():assert sha256(PROTOCOL)==read_json(REG)['ProtocolSHA256']
    if history:
        for p,h in {**f['historical_files'],**f['preserved_untracked']}.items():assert sha256(PROJECT/p)==h,p
    if data:
        for p,h in f['data_files'].items():assert sha256(PROJECT/p)==h,p
    return f
# Reuse exact AST bodies. Importing stage11b_common would install its historical
# stage-specific write hook; the extracted read-only namespace avoids that hook.
original_path=S11B/'00_manifest/stage11b_common.py'
original_tree=ast.parse(original_path.read_text())
names={'frame','split','indices','graph_normalize','Store','objective','groups'}
nodes=[n for n in original_tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
scope=dict(ROOT=S11B,CACHE=S11B/'01_preflight/cache',S8=S8,torch=torch,np=np,pd=pd,lru_cache=lru_cache,
           read_json=read_json,normalize_r2=normalize_r2,TYPES=TYPES)
exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(original_path),'exec'),scope)
for name in names:globals()[name]=scope[name]
def fresh(fold,device='cuda'):
    from stage14b_matched_nograph import MatchedNoGraphReranker
    return MatchedNoGraphReranker(seed=2022+100*(fold-1)).to(device)
def cp_path(fold,name):return ROOT/f'04_checkpoints/fold{fold}/Matched-NG-{name}_best.pt'
def load_fold_r2(fold,device='cuda'):
    m=ReliabilityHead().to(device);p=S11B/f'04_checkpoints/fold{fold}/R2_best.pt'
    saved=torch.load(p,map_location='cpu',weights_only=False);assert saved['Fold']==fold and saved['Model']=='R2'
    m.load_state_dict(saved['state_dict']);return m.eval().requires_grad_(False)
