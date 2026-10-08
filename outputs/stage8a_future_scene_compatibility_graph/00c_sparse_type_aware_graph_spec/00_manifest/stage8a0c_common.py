"""Isolated audit paths and utilities; no evaluation/predictor module imports."""
from pathlib import Path
import sys,json,csv,hashlib,subprocess,os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
STAGE8=ROOT.parent;PROJECT=STAGE8.parents[1]
STAGE0B=STAGE8/'00b_type_aware_semantic_audit'
STAGE6=PROJECT/'outputs/stage6a_future_interaction_reliability'
STAGE7=PROJECT/'outputs/stage7a_semantic_map_hivt'
sys.path[:0]=[str(PROJECT),str(STAGE8/'00_manifest')]
from stage8a_graph import ObservableWindow,observable_window,node_features,neighbors,interaction_edges,NODE_FIELDS,INTERACTION_FIELDS
from preprocessing.coordinates import ego_to_global,global_to_ego,rotation_matrix
BASE='dbb743ab79d3d0309419cd051624ff99c7a6a0ee'
BRANCH='stage8a0c/sparse-type-aware-graph-spec'
MAP_ROOT=PROJECT/'outputs/stage2/map_views/cache/bc39172aef2f'
MAP_JSON=MAP_ROOT/'maps/expansion'
CENTERLINES=STAGE7/'02_semantic_cache/stage7a_centerlines.npz'
SEMANTICS=STAGE7/'02_semantic_cache/stage7a_semantic_metadata.json'
TYPES=('lane','lane_connector','drivable_area','carpark_area','ped_crossing','walkway')
QUOTAS=np.array([[3,3,1,1,0,0],[0,0,1,0,2,3],[3,3,1,0,0,0]],dtype=np.int32)
LANE_SEMANTIC_FIELDS=('is_connector','turn_left','turn_straight','turn_right','turn_unknown','traffic_light_controlled','stop_sign_controlled','other_control','crosswalk_intersects')
MAP_NODE_FIELDS=tuple('entity_'+s for s in TYPES)+LANE_SEMANTIC_FIELDS+('local_tangent_sin','local_tangent_cos','tangent_valid')
MAP_EDGE_FIELDS=('minimum_distance_over10','mean_future_point_distance_over10','endpoint_distance_over10','closest_timestep_over_Tf',
    'fraction_future_points_within2m','fraction_future_points_within4m','heading_difference_sin','heading_difference_cos','heading_valid',
    'fraction_future_points_inside_polygon','polyline_intersects_polygon')
PREDICTOR=PROJECT/'outputs/stage5a_motion_aware_decoder/07_checkpoints/stage5a_best_overall_minfde.pt'
R2=STAGE6/'07_checkpoints/stage6a_r2_best.pt'
def read_json(p):return json.loads(Path(p).read_text())
def atomic_json(p,x):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(p)
def sha256(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def git(*args):return subprocess.check_output(['git',*args],cwd=PROJECT,text=True).strip()
def write_csv(p,rows):
    assert rows
    with Path(p).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def atomic_npz(p,arrays):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,**arrays)
    tmp.replace(p)
def groups(types,motion):
    return {'Overall':np.ones(len(types),dtype=bool),'Vehicle':types==0,'Pedestrian':types==1,'Bicycle':types==2,
        **{name:(types==0)&(motion==j) for j,name in enumerate(('MovingVehicle','StoppedVehicle','ParkedVehicle','UnknownVehicle'))}}
