"""Stage13A read-only historical sources and stage-isolated generated artifacts."""
from pathlib import Path
from functools import lru_cache
import os,sys,json,hashlib,ast,sqlite3,time,subprocess
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
ROOT=Path(__file__).resolve().parents[1];PROJECT=ROOT.parents[1];BASE='020bf774e33b3269a463bbc2d03e0c6f9185a1d1'
S3=ROOT.parent/'stage3_multitype_hivt';S6=ROOT.parent/'stage6a_future_interaction_reliability';S8=ROOT.parent/'stage8a_future_scene_compatibility_graph'
S11A=ROOT.parent/'stage11a_mode_selection_audit';S11B=ROOT.parent/'stage11b_error_aware_ranking';S11C=ROOT.parent/'stage11c_probability_calibration'
S12A=ROOT.parent/'stage12a_candidate_semantic_audit';S12B=ROOT.parent/'stage12b_semantic_reranking';S2C=ROOT.parent/'stage2c_trainval_vehicle_baseline'
RAW=ROOT.parent/'stage2/trainval/cache';DB=S2C/'02_preprocessed/stage2c_metadata_cache/stage2c_trajectory_metadata.sqlite'
MAP_ROOT=ROOT.parent/'stage2/map_views/cache/bc39172aef2f';PROTOCOL=ROOT/'00_manifest/stage13a_protocol.json';FREEZE=ROOT/'00_manifest/stage13a_frozen_history.json'
TEMPERATURES={1:93.47459065453494,2:41.89311387594937,3:77.85669786831706};TYPES=('Vehicle','Pedestrian','Bicycle')
sys.path[:0]=[str(PROJECT),str(S3/'02_preprocessed'),str(S6/'00_manifest'),str(S8/'00_manifest'),str(S8/'00c_sparse_type_aware_graph_spec/03_model_audit'),str(S8/'00c_sparse_type_aware_graph_spec/01_selector'),str(S8/'00c_sparse_type_aware_graph_spec/00_manifest')]
import numpy as np
import pandas as pd
import torch
from preprocessing.coordinates import global_to_ego,ego_to_global,global_heading_to_ego,quaternion_yaw,wrap_angle
from preprocessing.common import scene_samples,category_mapping
from preprocessing.build_one_window import annotation_index,build_window
READS=set()
def guard(event,args):
    if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
    p=Path(os.path.abspath(os.fsdecode(args[0])));mode=args[1] if len(args)>1 else None;flags=args[2] if len(args)>2 else 0
    write=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and bool(flags&(os.O_WRONLY|os.O_RDWR)))
    if write and p.is_relative_to(PROJECT) and not p.is_relative_to(ROOT):raise RuntimeError('Stage13A historical write prohibited: '+str(p))
    if write and p.is_relative_to(ROOT) and p.suffix in ('.pt','.pth'):raise RuntimeError('Stage13A creates no checkpoints')
    if any(s in str(p) for s in ['/02_preprocessed/val/','/01_cache/val/','/02_preprocessed/test/','/01_cache/test/']):raise RuntimeError('Stage13A forbidden split read')
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
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(p)
def atomic_npz(p,**x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,**x)
    tmp.replace(p)
def dump(rel,rows):
    p=ROOT/rel;p.parent.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).to_csv(p,index=False,float_format='%.17g')
def seed():
    torch.manual_seed(2022);np.random.seed(2022);torch.set_num_threads(1);torch.set_grad_enabled(False);torch.use_deterministic_algorithms(True)
def git(*args):return subprocess.check_output(['git',*args],cwd=PROJECT,text=True).strip()
def verify(history=False,data=False):
    f=read_json(FREEZE)
    for p,h in {**f['checkpoints'],**f['splits_norm_temperatures']}.items():assert sha256(PROJECT/p)==h,p
    if history:
        for p,h in {**f['historical_files'],**f['preserved_untracked']}.items():assert sha256(PROJECT/p)==h,p
    if data:
        for p,h in f['data_files'].items():assert sha256(PROJECT/p)==h,p
    return f
@lru_cache(None)
def split(k):return read_json(S11B/f'02_splits/stage11b_fold{k}_split.json')
@lru_cache(None)
def scene_folds():
    result={s:k for k in (1,2,3) for s in split(k)['OuterTest']};assert len(result)==630;return result
@lru_cache(None)
def original_extras():
    source=RAW/'v1.0-trainval';annotations={r['token']:r for r in read_json(source/'sample_annotation.json')};poses={r['token']:r for r in read_json(source/'ego_pose.json')}
    return annotations,poses
def connection():return sqlite3.connect(f'file:{DB}?mode=ro',uri=True)
# Execute the exact historical bounded facade, without importing its writing
# preparation entry points. Only raw_root points at the existing local metadata.
_path=S2C/'02_preprocessed/stage2c_prepare.py';_node=next(n for n in ast.parse(_path.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='SceneMetadata')
_scope={'json':json,'raw_root':lambda:RAW};exec(compile(ast.fix_missing_locations(ast.Module(body=[_node],type_ignores=[])),str(_path),'exec'),_scope)
SceneMetadata=_scope['SceneMetadata']
def load_scene(con,token):
    row=con.execute('SELECT payload FROM scenes WHERE token=?',(token,)).fetchone();assert row
    return SceneMetadata(json.loads(row[0]),con)
class PastOnly:
    """Reject future annotations/poses/sample_data during observation construction."""
    def __init__(self,scene,history):
        self.source=scene;self.allowed={};self.reads=[];self.category=scene.category
        self.allowed['sample']={r['token'] for r in history}
        self.allowed['sample_annotation']={t for r in history for t in r['anns']}
        self.allowed['sample_data']={r['data']['LIDAR_TOP'] for r in history}
        self.allowed['ego_pose']={scene.get('sample_data',t)['ego_pose_token'] for t in self.allowed['sample_data']}
    def get(self,table,token):
        assert table in self.allowed and token in self.allowed[table],('future read prohibited',table,token)
        self.reads.append((table,token));return self.source.get(table,token)
def distribution(values):
    a=np.asarray(values,np.float64);return dict(Count=len(a),Min=float(a.min()) if len(a) else None,Mean=float(a.mean()) if len(a) else None,
        Median=float(np.median(a)) if len(a) else None,P95=float(np.quantile(a,.95)) if len(a) else None,Max=float(a.max()) if len(a) else None)
