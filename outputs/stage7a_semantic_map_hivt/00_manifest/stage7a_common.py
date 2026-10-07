"""Stage7A-0 read-only data references and audit-only helpers."""
from pathlib import Path
import sys,json,csv,hashlib,subprocess
ROOT=Path(__file__).resolve().parents[1];PROJECT=ROOT.parents[1]
sys.path.insert(0,str(PROJECT))
STAGE3=ROOT.parent/'stage3_multitype_hivt';STAGE6=ROOT.parent/'stage6a_future_interaction_reliability'
BASE='e43347c2c2cd356775f4e0bb59fcf7dbda76f2b5';TAG='paper-v1-stage6a-r2'
MAP_ROOT=PROJECT/'outputs/stage2/map_views/cache/bc39172aef2f'
MAP_JSON=MAP_ROOT/'maps/expansion'
PREDICTOR=ROOT.parent/'stage5a_motion_aware_decoder/07_checkpoints/stage5a_best_overall_minfde.pt'
R2=STAGE6/'07_checkpoints/stage6a_r2_best.pt'
CLASSES=('vehicle','pedestrian','bicycle')
TURN=('left','straight','right','unknown')
ACTORS=STAGE6/'04_evaluation/stage6a_actor_ranking.csv'
def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(1024*1024),b''):h.update(data)
    return h.hexdigest()
def read_json(path):return json.loads(Path(path).read_text())
def atomic_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(path)
def write_csv(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);assert rows
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def git(*args):return subprocess.check_output(['git',*args],cwd=PROJECT,text=True).strip()
def key(r):return tuple(str(r[k]) for k in ('scene_token','sample_token','instance_token','horizon'))
def indexes(split):
    with (STAGE3/'02_preprocessed'/f'stage3_{split}_index.csv').open() as f:return list(csv.DictReader(f))
def verify_v1():
    ref=read_json(ROOT/'00_manifest/stage7a_v1_frozen_reference.json')
    for r in ref['files'].values():assert sha256(PROJECT/r['path'])==r['sha256'],r['path']
    assert git('rev-parse',TAG+'^{}')==BASE and git('cat-file','-t',TAG)=='tag'
    return ref
