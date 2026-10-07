"""Preregister fixed semantics, offline groups and decisions BEFORE optimization."""
from pathlib import Path
import sys, collections, datetime, csv
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'00_manifest'))
from stage7a_formal_common import *
from stage7a_dataset import SEMANTIC_FIELDS, semantic_vector
import torch
import numpy as np
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree
from preprocessing.coordinates import ego_to_global, wrap_angle

def main():
    assert not BEST.exists() and not CURVE.exists(), 'Registration must precede training'
    verify_previous(shards=True)
    metadata=read_json(ROOT/'02_semantic_cache/stage7a_semantic_metadata.json')
    z=np.load(ROOT/'02_semantic_cache/stage7a_centerlines.npz')
    trees={};turn_tokens={};vectors=[]
    for loc,rows in metadata.items():
        predicates={'IntersectionVehicle20':lambda v:v['is_connector'],
          'NearTurnConnector20':lambda v:v['is_connector'] and v['turn_type'] in ('left','right'),
          'NearTrafficControl20':lambda v:bool(v['control_types_present'])}
        for name,predicate in predicates.items():
            tokens=[t for t,v in rows.items() if predicate(v)]
            trees[loc,name]=STRtree([LineString(z[loc+'__'+t]) for t in tokens])
            if name=='NearTurnConnector20':turn_tokens[loc]=tokens
        for token,v in rows.items():vectors.append({'map_location':loc,'lane_token':token,
            **dict(zip(SEMANTIC_FIELDS,semantic_vector(v).tolist()))})
    write_csv(ROOT/'06_tables/stage7a_formal_semantic_token_encoding.csv',vectors)
    with (ROOT/'01_data_audit/stage7a_actor_val_semantic_context.csv').open() as f:
        old={(a['scene_token'],a['sample_token'],a['instance_token']):a for a in csv.DictReader(f)}
    ds=Stage7ASemanticDataset('val');actors=[]
    for i in range(len(ds)):
        g=ds[i];nodes=torch.where(g.full_horizon_mask)[0].numpy()
        pos=ego_to_global(g.positions[nodes,4].numpy(),g.origin.numpy(),float(g.ego_yaw))
        points=np.array([Point(p) for p in pos],dtype=object)
        minima={};near_turn_pairs={}
        for name in predicates:
            tree=trees[g.map_location,name];pairs=tree.query(points,predicate='dwithin',distance=20)
            distances=np.array([points[a].distance(tree.geometries[b]) for a,b in pairs.T])
            minimum=np.full(len(nodes),np.inf)
            if pairs.size:np.minimum.at(minimum,pairs[0],distances)
            minima[name]=minimum
            if name=='NearTurnConnector20':
                for (a,b),distance in zip(pairs.T,distances):
                    if distance<20:near_turn_pairs.setdefault(int(a),[]).append((float(distance),int(b)))
        for j,node in enumerate(nodes):
            vehicle=int(g.agent_type[node])==0;key=(g.scene_token,g.sample_token,g.instance_tokens[node]);a=old[key]
            trajectory=g.positions[node,4:].numpy();displacement=float(np.linalg.norm(trajectory[-1]-trajectory[0]))
            vin=trajectory[2]-trajectory[0];vout=trajectory[-1]-trajectory[-3]
            eligible=vehicle and displacement>5 and np.linalg.norm(vin)>0.5 and np.linalg.norm(vout)>0.5
            delta=float(np.degrees(wrap_angle(np.arctan2(vout[1],vout[0])-np.arctan2(vin[1],vin[0])))) if eligible else 0.
            turning=int(eligible and abs(delta)>20);assert turning==int(a['TurningVehicle_GT'])
            flags={name:int(vehicle and minima[name][j]<20) for name in predicates}
            assert flags['IntersectionVehicle20']==int(a['IntersectionVehicle_A'])
            context='unassigned';near=sorted(near_turn_pairs.get(j,[]))
            if vehicle and near and (turning or flags['NearTurnConnector20']):
                # Unique geometric nearest connector; <=1cm ties remain unassigned.
                if len(near)==1 or near[1][0]-near[0][0]>0.01:
                    context=metadata[g.map_location][turn_tokens[g.map_location][near[0][1]]]['turn_type']
            row={'scene_name':g.scene_name,'scene_token':g.scene_token,'sample_token':g.sample_token,
                'instance_token':g.instance_tokens[node],'horizon':'full_horizon','node_in_graph':int(node),
                'agent_type':CLASSES[int(g.agent_type[node])],'motion_state':g.t0_motion_state[node],
                'GT_endpoint_displacement_m':float(torch.linalg.vector_norm(g.y[node,-1])),
                **flags,'TurningVehicle_GT':turning,'turn_context':context,
                **GT_fingerprint(g.positions[node,5:],g.future_mask[node],g.agent_type[node])}
            actors.append(row)
        if (i+1)%500==0:print('GROUP_REGISTRATION',i+1,'/',len(ds),flush=True)
    ds.clear();assert len(actors)==54990 and sum(a['TurningVehicle_GT'] for a in actors)==1663
    sidecar=ROOT/'01_data_audit/stage7a_formal_actor_val_groups.csv';write_csv(sidecar,actors)
    counts={name:sum(a[name] for a in actors) for name in [*predicates,'TurningVehicle_GT']}
    atomic_json(ROOT/'00_manifest/stage7a_offline_group_registration.json',{
        'registered_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'registered_before_training':True,'sidecar_sha256':sha256(sidecar),'counts':counts,
        'distance_rule':'t0 global XY to whole centerline; strict <20m; all regional tokens, not truncated graph segments',
        'turn_context_rule':'unique nearest left/right connector within20m; second distance minus first >0.01m; ties unassigned',
        'TurningVehicle_GT':'vehicle full future; endpoint>5m; first/last1s secants>0.5m; abs wrapped angle>20deg; identical Stage7A-0 membership',
        'model_input':False,'baseline_prediction_errors_used':False})
    sources=[*sorted((ROOT/'00_manifest').glob('stage7a_*.py')),ROOT/'03_training/stage7a_train.py']
    # Audit-stage utilities are preserved but not part of the formal training dependency set.
    deps=['stage7a_dataset.py','stage7a_model.py','stage7a_local_encoder.py',
          'stage7a_formal_common.py','stage7a_evaluation_core.py']
    sources=[ROOT/'00_manifest'/n for n in deps]+[ROOT/'03_training/stage7a_train.py']
    atomic_json(PREREG,{'registered_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
      'training_authorized':True,'Stage7B_authorized':False,'from_scratch':True,'formal_runs':1,
      'seed':2022,'config_sha256':sha256(CONFIG),'training_source_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in sources},
      'semantic_fields':SEMANTIC_FIELDS,'ordinary_lane_turn_slots':'all zero',
      'controls':'multi-hot; YIELD merged into other_control','crosswalk':'whole token centerline intersects polygon',
      'baseline_checkpoint_sha256':sha256(BASE_BEST),'baseline_checkpoint_used_for_initialization':False,
      'group_registration_sha256':sha256(ROOT/'00_manifest/stage7a_offline_group_registration.json'),
      'group_sidecar_sha256':sha256(sidecar),'frozen_source_sha256':sha256(FREEZE),
      'warmup_updates':5000,'NLL_updates_max':16000,'global_hard_max':21000,
      'validation_interval':500,'patience':5,'primary_selection':'original-NLL overall full-horizon minFDE6 strict improvement',
      'bootstrap':{'unit':'whole official VAL scene','scenes':150,'replicates':1000,'seed':2022,
          'pooling':'actor-window sums/counts','interval':'percentile2.5/97.5','delta':'Stage7A minus Stage3B'},
      'difficult_groups':['vehicle.moving','Vehicle >5m',*predicates,'TurningVehicle_GT'],
      'scientific_rules':{
        'marked_reliable_harm':'Vehicle or Pedestrian minFDE increases >5% relative AND paired CI lower>0',
        'basically_flat':'overall relative point increase<=1%; for TARGETED_SUPPORTED CI must cross/include0',
        'SUPPORTED':'overall delta<0 and CI upper<0; no marked reliable harm; >=1 difficult-group point delta<0',
        'TARGETED_SUPPORTED':'overall point improved or <=1% increase, CI includes0; >=2 difficult-group CI upper<0; no marked reliable harm',
        'PaperUsableSemantic':'YES iff SUPPORTED or TARGETED_SUPPORTED',
        'ReadyStage7B':'YES iff PaperUsableSemantic YES; otherwise overall point increase<=1%, overall CI not wholly>0, and BOTH moving and TurningVehicle_GT CI upper<0; no marked reliable harm',
        'secondary_multiplicity':'exploratory unadjusted subgroup CIs; correlated/overlapping groups, not independent replications'},
      'semantic_zero':'input nine features set0; learned SemanticMLP bias remains; same checkpoint once, no retraining',
      'tiny':'six fixed TRAIN graphs,200 fixed-scale updates, no architecture tuning',
      'test_used':False})
    print('PREREGISTERED',counts,flush=True)

if __name__=='__main__':torch.set_num_threads(1);main()
