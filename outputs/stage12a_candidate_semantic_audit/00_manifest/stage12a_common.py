"""Stage12A frozen sources, observation allowlist and historical write isolation."""
from pathlib import Path
from functools import lru_cache
import os,sys,json,hashlib
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
ROOT=Path(__file__).resolve().parents[1];PROJECT=ROOT.parents[1]
S8=ROOT.parent/'stage8a_future_scene_compatibility_graph';S8C=S8/'00c_sparse_type_aware_graph_spec'
S6=ROOT.parent/'stage6a_future_interaction_reliability';S11A=ROOT.parent/'stage11a_mode_selection_audit'
S11B=ROOT.parent/'stage11b_error_aware_ranking';S11C=ROOT.parent/'stage11c_probability_calibration'
BASE='165c497961afe9f3a196a0987128f9eed48da144'
sys.path[:0]=[str(PROJECT),str(S8C/'00_manifest'),str(S8C/'01_selector')]
import numpy as np
import pandas as pd
import torch
import shapely
from stage8a0c_selector import SparseSemanticIndex
from stage8a0c_common import TYPES as MAP_TYPES,QUOTAS,MAP_JSON,MAP_ROOT,CENTERLINES,SEMANTICS
from stage8a_graph import observable_window
from preprocessing.coordinates import ego_to_global,global_to_ego,rotation_matrix,wrap_angle
PROTOCOL=ROOT/'00_manifest/stage12a_protocol.json';REG=ROOT/'00_manifest/stage12a_registration.json'
FROZEN=ROOT/'00_manifest/stage12a_frozen_manifest.json'
OOF=S11B/'05_oof_evaluation/cache';TRAIN=S8/'01_training/cache'
SEM_FIELDS=('centerline_distance','centerline_mean_distance','centerline_endpoint_distance','centerline_heading_error',
 'centerline_valid','lane_count','connector_count','drivable_distance','drivable_inside_fraction','drivable_intersects',
 'drivable_boundary_crossing','drivable_boundary_distance','drivable_valid','carpark_distance','carpark_inside_fraction',
 'carpark_intersects','carpark_valid','crosswalk_distance','crosswalk_inside_fraction','crosswalk_intersects','crosswalk_valid',
 'walkway_distance','walkway_inside_fraction','walkway_intersects','walkway_valid','map_valid_mask')
GEO_FIELDS=('predicted_displacement','endpoint_x','endpoint_y','predicted_heading','trajectory_length','trajectory_curvature',
 'predicted_heading_valid','recent_speed','history_displacement','history_path_length','history_heading','history_heading_valid',
 'history_valid_count','current_x','current_y','original_probability','original_logit','neighbor_count','interaction_valid',
 'interaction_minimum_distance','interaction_mean_distance','interaction_endpoint_distance','interaction_closing_mean',
 'interaction_relative_step_minimum')
ACTOR_TYPES=('Vehicle','Pedestrian','Bicycle')
READS=set();FORBIDDEN=[S6/'01_cache/val',S8/'03_evaluation/cache',
 ROOT.parent/'stage9a_type_adaptive_future_graph/03_evaluation/cache',
 ROOT.parent/'stage3_multitype_hivt/02_preprocessed/val',ROOT.parent/'stage2c_trainval_vehicle_baseline/02_preprocessed/val',
 S6/'02_features/stage6a_headdev_features.pt',S6/'02_features/stage6a_val_features.pt',
 *[S11C/f'02_calibration/fold{k}/cache' for k in (1,2,3)]]
LABEL_FILES=[S11A/'01_identity_audit/cache/GT.npy',TRAIN/'fde.npy',TRAIN/'ade.npy',OOF/'stage11b_oof_metrics.npy']
def guard(event,args):
    if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
    p=Path(os.path.abspath(os.fsdecode(args[0])));m=args[1] if len(args)>1 else None;flags=args[2] if len(args)>2 else 0
    write=(isinstance(m,str) and any(x in m for x in 'wax+')) or (isinstance(flags,int) and bool(flags&(os.O_WRONLY|os.O_RDWR)))
    if write and p.is_relative_to(PROJECT) and not p.is_relative_to(ROOT):raise RuntimeError('Stage12A historical write forbidden: '+str(p))
    if write and p.is_relative_to(ROOT) and ('.pt' in p.suffixes or '.pth' in p.suffixes):raise RuntimeError('Stage12A checkpoint creation forbidden')
    if any(p==q or p.is_relative_to(q) for q in FORBIDDEN) or '/02_preprocessed/test/' in str(p) or '/01_cache/test/' in str(p):raise RuntimeError('Stage12A forbidden data: '+str(p))
    if os.environ.get('STAGE12A_PHASE')=='extract' and p in LABEL_FILES:raise RuntimeError('Future labels forbidden before semantic freeze: '+str(p))
    if not write and p.is_relative_to(PROJECT):READS.add(str(p.relative_to(PROJECT)))
sys.addaudithook(guard)
def read_json(p):return json.loads(Path(p).read_text())
def sha256(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def array_sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def atomic_json(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n');t.replace(p)
def atomic_npz(p,**x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp')
    with t.open('wb') as f:np.savez_compressed(f,**x)
    t.replace(p)
def dump(relative,rows):
    p=ROOT/relative;p.parent.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).to_csv(p,index=False,float_format='%.12g')
def verify(history=False,data=False):
    frozen=read_json(FROZEN);assert sha256(PROTOCOL)==read_json(REG)['protocol_sha256']
    for p,h in frozen['checkpoints'].items():assert sha256(PROJECT/p)==h,p
    for p,h in frozen['map_sources'].items():assert sha256(PROJECT/p)==h,p
    if history:
        for p,h in {**frozen['historical_files'],**frozen['preserved_untracked_files']}.items():assert sha256(PROJECT/p)==h,p
    if data:
        for p,h in frozen['data_files'].items():assert sha256(PROJECT/p)==h,p
    return frozen
@lru_cache(None)
def identities():
    f=pd.read_csv(OOF/'stage11b_oof_predictions_actor_records.csv',dtype={'future_mask_bits':str})
    assert len(f)==260151 and f.actor_id.is_unique and f.scene_token.nunique()==630 and (f.Partition=='OuterTest').all()
    return f
def observed_identities():
    # Future displacement, labels and motion classes are never returned to the extractor.
    cols=['source_index','scene_token','sample_token','instance_token','actor_id','agent_type','agent_type_id',
      'Fold','cache_path','dataset_index','node_in_graph','recent_speed','history_net','history_path','history_valid_count']
    return identities()[cols].copy()
def groups(f,geometry=None):
    typ=f.agent_type.to_numpy();gt=f.GT_displacement.to_numpy()
    out={'Overall':np.ones(len(f),bool),**{t:typ==t for t in ACTOR_TYPES},
      **{g:(typ=='Vehicle')&(f.motion_state.to_numpy()==state) for g,state in [('MovingVehicle','vehicle.moving'),('StoppedVehicle','vehicle.stopped'),('ParkedVehicle','vehicle.parked')]},
      'Vehicle>5m':(typ=='Vehicle')&(gt>5),'Pedestrian<5m':(typ=='Pedestrian')&(gt<5),
      'Pedestrian>5m':(typ=='Pedestrian')&(gt>5),'Pedestrian5-8m':(typ=='Pedestrian')&(gt>=5)&(gt<8)}
    if geometry is not None:
        g=geometry[:,0];ped=typ=='Pedestrian'
        for label,field,edges in [('RecentSpeed','recent_speed',[0,.5,1.5,3,np.inf]),('HistoryDisp','history_displacement',[0,.5,2,5,np.inf])]:
            b=np.searchsorted(edges[1:],g[:,GEO_FIELDS.index(field)],side='right')
            for j in range(len(edges)-1):out[f'Pedestrian_{label}_{j}']=ped&(b==j)
        valid=g[:,GEO_FIELDS.index('history_heading_valid')]>0
        heading=g[:,GEO_FIELDS.index('history_heading')];b=np.floor((heading+np.pi)/(2*np.pi)*8).astype(int)%8
        for j in range(8):out[f'Pedestrian_HistoryHeading_{j}']=ped&valid&(b==j)
        out['Pedestrian_HistoryHeading_Missing']=ped&~valid
    return out
def scopes(f):return [('Pooled',np.ones(len(f),bool))]+[(f'Fold{k}',f.Fold.to_numpy()==k) for k in (1,2,3)]
