"""Test analytic centerline tangents against independent finer discretization."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'00_manifest'),str(ROOT/'01_data_audit')]
from stage7a_common import *
from stage7a_map_inventory import load_maps,BINS
from preprocessing.coordinates import wrap_angle
import numpy as np
def angle(v):return float(np.arctan2(v[1],v[0]))
def difference(a,b):return float(abs(np.degrees(wrap_angle(a-b))))
def main():
    inventory=read_json(ROOT/'02_semantic_cache/stage7a_geometry_inventory.json');coarse=np.load(ROOT/'02_semantic_cache/stage7a_centerlines.npz');rows=[];hist=[]
    for path,raw,api in load_maps():
        loc=path.stem;records=raw['lane_connector'];fine=api.discretize_lanes([r['token'] for r in records],0.2)
        for r in records:
            token=r['token'];arcs=raw['arcline_path_3'][token];tin=float(arcs[0]['start_pose'][2]);tout=float(arcs[-1]['end_pose'][2]);delta=float(np.degrees(wrap_angle(tout-tin)))
            p=np.asarray(fine[token])[:,:2];v=np.diff(p,axis=0);v=v[np.linalg.norm(v,axis=-1)>1e-6];assert len(v)
            finedelta=float(np.degrees(wrap_angle(angle(v[-1])-angle(v[0]))))
            line=coarse[loc+'__'+token];meta=inventory[loc][token];data={}
            for side,ids,endpoint in [('incoming',meta['incoming'],line[0]),('outgoing',meta['outgoing'],line[-1])]:
                candidates=[]
                for neighbor in ids:
                    if neighbor not in inventory[loc]:continue
                    ln=coarse[loc+'__'+neighbor];ar=raw['arcline_path_3'][neighbor]
                    pt=ln[-1] if side=='incoming' else ln[0]
                    yaw=ar[-1]['end_pose'][2] if side=='incoming' else ar[0]['start_pose'][2]
                    candidates.append((float(np.linalg.norm(pt-endpoint)),difference(yaw,tin if side=='incoming' else tout)))
                best=min(candidates) if candidates else (None,None)
                data[side+'_minimum_endpoint_gap_m']=best[0];data[side+'_heading_difference_deg']=best[1]
            rows.append({'location':loc,'token':token,'theta_in_rad':tin,'theta_out_rad':tout,'delta_theta_deg':delta,
                'fine_0p2m_delta_deg':finedelta,'fine_analytic_difference_deg':difference(np.radians(finedelta),np.radians(delta)),
                'coarse_2m_delta_deg':meta['delta_theta_deg'],'coarse_analytic_difference_deg':difference(np.radians(meta['delta_theta_deg']),np.radians(delta)),
                'incoming_count':len(meta['incoming']),'outgoing_count':len(meta['outgoing']),**data})
        counts=np.histogram([r['delta_theta_deg'] for r in rows if r['location']==loc],bins=BINS)[0]
        for i,n in enumerate(counts):hist.append({'location':loc,'bin_low_deg':BINS[i],'bin_high_deg':BINS[i+1],'right_closed':i==len(counts)-1,'connector_count':int(n)})
    write_csv(ROOT/'06_tables/stage7a_connector_tangent_validation.csv',rows)
    write_csv(ROOT/'06_tables/stage7a_analytic_angle_histogram.csv',hist)
    atomic_json(ROOT/'01_data_audit/stage7a_turn_geometry_audit.json',{'status':'BEFORE_THRESHOLD_ADOPTION','connector_count':len(rows),
        'source':'arcline_path_3 first start_pose[2] and last end_pose[2] are analytic centerline entrance/exit tangents',
        'fine_discretization_resolution_m':0.2,'existing_lane_resolution_m_unchanged':2.0,
        'all_finite':bool(np.isfinite([[r['theta_in_rad'],r['theta_out_rad'],r['delta_theta_deg']] for r in rows]).all()),
        'fine_analytic_difference_max_deg':max(r['fine_analytic_difference_deg'] for r in rows),
        'coarse_analytic_difference_max_deg':max(r['coarse_analytic_difference_deg'] for r in rows),
        'connector_dangling_topology_references':sum(len(inventory[r['location']][r['token']]['unresolved_connectivity_references']) for r in rows),
        'no_threshold_adopted':True})
    print('TURN_GEOMETRY_DONE',len(rows),flush=True)
if __name__=='__main__':main()
