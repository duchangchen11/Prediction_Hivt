"""Stage8A-0B isolated paths; historical modules are read-only dependencies."""
from pathlib import Path
import sys, json, csv, hashlib, subprocess
ROOT = Path(__file__).resolve().parents[1]
STAGE8 = ROOT.parent
PROJECT = STAGE8.parents[1]
sys.path.insert(0, str(STAGE8/'00_manifest'))
from stage8a_common import STAGE6, STAGE7A, STAGE3, SEMANTICS, CENTERLINES, PREDICTOR, R2, PREDICTOR_SHA, R2_SHA, read_json, atomic_json, sha256, write_csv, atomic_npz
from stage8a_graph import observable_window
from stage8a_map import MapIndex
import numpy as np
import torch
import shapely
from preprocessing.coordinates import ego_to_global, global_to_ego, rotation_matrix
MAP_ROOT = PROJECT/'outputs/stage2/map_views/cache/bc39172aef2f'
MAP_JSON = MAP_ROOT/'maps/expansion'
BASE = '232a0ad7cf73674a8c651eb34301356f62cd6f1a'
BRANCH = 'stage8a0b/type-aware-semantic-entity-audit'
TYPES = ('lane','lane_connector','drivable_area','carpark_area','ped_crossing','walkway','stop_line')
RELEVANT = np.array([[1,1,1,1,0,0],[0,0,1,0,1,1],[1,1,1,0,0,0]], dtype=bool)
MOTIONS = ('vehicle.moving','vehicle.stopped','vehicle.parked','unknown')
def git(*args): return subprocess.check_output(['git',*args],cwd=PROJECT,text=True).strip()
def groups(actor_type, motion):
    return {'Overall':np.ones(len(actor_type),dtype=bool), 'Vehicle':actor_type==0,
        'MovingVehicle':(actor_type==0)&(motion==0), 'StoppedVehicle':(actor_type==0)&(motion==1),
        'ParkedVehicle':(actor_type==0)&(motion==2), 'UnknownVehicle':(actor_type==0)&(motion==3),
        'Pedestrian':actor_type==1, 'Bicycle':actor_type==2}
