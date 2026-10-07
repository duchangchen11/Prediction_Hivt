"""Isolated Stage8A-0 references and bounded, read-only input access."""
from pathlib import Path
import sys,os,json,hashlib,subprocess,csv
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT.parents[1]
STAGE6=ROOT.parent/'stage6a_future_interaction_reliability'
STAGE7A=ROOT.parent/'stage7a_semantic_map_hivt'
STAGE3=ROOT.parent/'stage3_multitype_hivt'
sys.path[:0]=[str(STAGE6/'00_manifest'),str(PROJECT)]
from stage6a_common import SceneDataset,frozen_predictor,tensor_sha,state_digest
import numpy as np
import torch
BASE='35b7811ce2ec0c7dd3b252995b49724e5971b4be'
BRANCH='stage8a/future-scene-compatibility-graph'
PYTHON='/home/lrj/anaconda3/envs/ped_intent/bin/python'
PREDICTOR=ROOT.parent/'stage5a_motion_aware_decoder/07_checkpoints/stage5a_best_overall_minfde.pt'
PREDICTOR_SHA='88fe3feb59b7e830ec1e40aa917484386e0d2799d0f6116f410adda2d97fdce7'
R2=STAGE6/'07_checkpoints/stage6a_r2_best.pt'
R2_SHA='e3de457d635cd30c629ce316d73a87e419a3ded8abbe0e1f2601f1071527e314'
CACHE_MANIFEST=STAGE6/'01_cache/stage6a_cache_manifest.json'
SEMANTICS=STAGE7A/'02_semantic_cache/stage7a_semantic_metadata.json'
CENTERLINES=STAGE7A/'02_semantic_cache/stage7a_centerlines.npz'
SPLIT=STAGE6/'00_manifest/stage6a_head_split.json'
CLASSES=('vehicle','pedestrian','bicycle')

def read_json(p):return json.loads(Path(p).read_text())
def atomic_json(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def sha256(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def git(*args):return subprocess.check_output(['git',*args],cwd=PROJECT,text=True).strip()
def write_csv(p,rows):
    assert rows
    with Path(p).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def verify_frozen(shards=False):
    f=read_json(ROOT/'00_manifest/stage8a_frozen_inputs.json')
    for n,h in f['files'].items():assert sha256(PROJECT/n)==h,n
    for n,h in f['unrelated_untracked'].items():assert sha256(PROJECT/n)==h,n
    if shards:
        for n,h in f['scene_shards'].items():assert sha256(STAGE3/n)==h,n
    assert sha256(PREDICTOR)==PREDICTOR_SHA and sha256(R2)==R2_SHA
    return f

def frame_index(split):
    # Read frames from the same immutable scene shard, never infer origin from GT.
    return SceneDataset(split,cache_scenes=2)

def atomic_npz(path,arrays):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('wb') as f:np.savez_compressed(f,**arrays)
    temp.replace(path)
