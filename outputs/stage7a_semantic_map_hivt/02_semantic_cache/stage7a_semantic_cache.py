"""Audit-only (location, token) semantics; never changes the old extractor."""
from pathlib import Path
import sys,csv,collections
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'00_manifest'),str(ROOT/'01_data_audit')]
from stage7a_common import *
from stage7a_map_inventory import load_maps
import numpy as np
from shapely.geometry import LineString
from shapely.strtree import STRtree

CONTROL=('none','traffic_light','stop_sign','yield','other_control','mixed_control')
def turn(d):return 'unknown' if abs(d)>=150 else ('left' if d>20 else 'right' if d<-20 else 'straight')
def canonical(s):return {'TRAFFIC_LIGHT':'traffic_light','STOP_SIGN':'stop_sign','YIELD':'yield'}.get(s,'other_control')
def main():
    review=read_json(ROOT/'01_data_audit/stage7a_turn_manual_review.json')
    assert review['status']=='PASS' and all(review['reviewed_counts'][t]>=20 for t in ('left','straight','right'))
    inventory=read_json(ROOT/'02_semantic_cache/stage7a_geometry_inventory.json');lines=np.load(ROOT/'02_semantic_cache/stage7a_centerlines.npz')
    with (ROOT/'06_tables/stage7a_connector_tangent_validation.csv').open() as f:angles={(r['location'],r['token']):r for r in csv.DictReader(f)}
    cache={};turnstats=[];controlstats=[];crossstats=[];stopassoc=[]
    for path,raw,api in load_maps():
        loc=path.stem;print('SEMANTIC_CACHE',loc,flush=True);cache[loc]={}
        stops=raw['stop_line'];stop_polys=[api.extract_polygon(r['polygon_token']) for r in stops];st=STRtree(stop_polys)
        crossings=raw['ped_crossing'];cross_polys=[api.extract_polygon(r['polygon_token']) for r in crossings];ct=STRtree(cross_polys)
        assoc=collections.defaultdict(list)
        for token,base in inventory[loc].items():
            p=lines[loc+'__'+token];geom=LineString(p);valid=np.linalg.norm(np.diff(p,axis=0),axis=-1)>1e-6;segments=[LineString(p[i:i+2]) for i in np.flatnonzero(valid)]
            si=st.query(geom,predicate='intersects');ci=ct.query(geom,predicate='intersects');c2=ct.query(geom,predicate='dwithin',distance=2);c5=ct.query(geom,predicate='dwithin',distance=5)
            rawtypes=sorted({stops[int(i)]['stop_line_type'] for i in si});canon=sorted({canonical(t) for t in rawtypes});control=canon[0] if len(canon)==1 else ('mixed_control' if canon else 'none')
            tlrefs=sorted({t for i in si for t in stops[int(i)].get('traffic_light_tokens',[])})
            typ=turn(float(angles[(loc,token)]['delta_theta_deg'])) if base['is_connector'] else 'unknown'
            direct={}
            for label,distance in [('intersects',0),('within2m',2),('within5m',5)]:
                pairs=ct.query(segments,predicate='intersects') if not distance else ct.query(segments,predicate='dwithin',distance=distance)
                direct[label]=sorted(set(map(int,pairs[0])))
            sm={'is_connector':base['is_connector'],'turn_type':typ,'turn_onehot':[int(typ==t) for t in TURN],
                'delta_theta_deg':float(angles[(loc,token)]['delta_theta_deg']) if base['is_connector'] else None,
                'theta_in_rad':float(angles[(loc,token)]['theta_in_rad']) if base['is_connector'] else None,
                'theta_out_rad':float(angles[(loc,token)]['theta_out_rad']) if base['is_connector'] else None,
                'traffic_control_type':control,'traffic_control_onehot':[int(control==t) for t in CONTROL],
                'raw_stop_line_types':rawtypes,'control_types_present':canon,'stop_line_tokens':[stops[int(i)]['token'] for i in si],
                'traffic_light_reference_tokens':tlrefs,'near_ped_crossing':bool(len(ci)),'near_ped_crossing_2m':bool(len(c2)),
                'near_ped_crossing_5m':bool(len(c5)),'crosswalk_intersects_tokens':[crossings[int(i)]['token'] for i in ci],
                'direct_crosswalk_segment_indices':direct,'layer':base['layer'],'segment_count':base['segment_count'],
                'centerline_length_m':base['length_m'],'has_source_topology_gap':bool(base['unresolved_connectivity_references']),
                'topology_used_as_feature':False,'control_association':'centerline_intersects_typed_stop_line_polygon',
                'turn_source':'analytic_arcline_entrance_exit_tangents' if base['is_connector'] else 'ordinary_lane_unknown'}
            assert sum(sm['turn_onehot'])==sum(sm['traffic_control_onehot'])==1
            cache[loc][token]=sm
            for i in si:assoc[int(i)].append(token)
        for i,r in enumerate(stops):stopassoc.append({'location':loc,'stop_line_token':r['token'],'stop_line_type':r['stop_line_type'],
            'associated_lane_count':len(assoc[i]),'associated_lane_tokens':'|'.join(assoc[i])})
        for typ in TURN:
            members=[v for v in cache[loc].values() if v['is_connector'] and v['turn_type']==typ];a=np.array([v['delta_theta_deg'] for v in members]);turnstats.append({'location':loc,'turn_type':typ,
                'connector_count':len(members),'segment_count':sum(v['segment_count'] for v in members),
                **{n:float(f(a)) if len(a) else None for n,f in [('angle_mean',np.mean),('angle_std',np.std),('angle_min',np.min),('angle_max',np.max)]}})
        for typ in sorted({r['stop_line_type'] for r in stops}):
            rec=[i for i,r in enumerate(stops) if r['stop_line_type']==typ];ts={t for i in rec for t in assoc[i]}
            controlstats.append({'location':loc,'control_type':typ,'record_count':len(rec),'associated_lane_count':len(ts),
                'segment_count':sum(cache[loc][t]['segment_count'] for t in ts),'associated_stop_line_records':sum(bool(assoc[i]) for i in rec)})
        for layer in ('lane','lane_connector'):
            members=[v for v in cache[loc].values() if v['layer']==layer]
            for label,field in [('intersects','near_ped_crossing'),('within2m','near_ped_crossing_2m'),('within5m','near_ped_crossing_5m')]:
                ns=sum(v['segment_count'] for v in members)
                crossstats.append({'location':loc,'layer':layer,'definition':label,'token_count':len(members),'associated_token_count':sum(v[field] for v in members),
                    'token_coverage_rate':sum(v[field] for v in members)/len(members),'segment_count':ns,
                    'broadcast_segment_coverage_rate':sum(v['segment_count'] for v in members if v[field])/ns,
                    'direct_segment_coverage_rate':sum(len(v['direct_crosswalk_segment_indices'][label]) for v in members)/ns})
    atomic_json(ROOT/'02_semantic_cache/stage7a_semantic_metadata.json',cache)
    write_csv(ROOT/'06_tables/stage7a_turn_semantic_statistics.csv',turnstats)
    write_csv(ROOT/'06_tables/stage7a_control_semantic_statistics.csv',controlstats)
    write_csv(ROOT/'06_tables/stage7a_crosswalk_coverage.csv',crossstats)
    write_csv(ROOT/'06_tables/stage7a_stop_line_lane_associations.csv',stopassoc)
    atomic_json(ROOT/'00_manifest/stage7a_semantic_candidate_definition.json',{'status':'AUDIT_CANDIDATE_ONLY','training_authorized':False,
        'key':['location','lane_token'],'is_connector_dimensions':1,'turn_classes':list(TURN),'turn_threshold_degrees':20,
        'turn_method':'analytic centerline entrance/exit tangents; finer/coarse sampling and map visual verification',
        'u_turn_unknown_absolute_degrees':150,'u_turn_reason':'Observed U-turn reverses wrapped sign near +/-180; outside left/straight/right V1 taxonomy, explicitly unknown.',
        'traffic_control_classes':list(CONTROL),'traffic_control_dimensions':len(CONTROL),
        'mixed_control_reason':'Preserve overlapping actual stop-line types; no invented priority rule or forced 5D labels.',
        'traffic_control_relation':'whole centerline intersects a source stop_line polygon; static map association, not direct authoritative lane-control annotation',
        'crosswalk_primary_relation':'whole centerline intersects source ped_crossing polygon',
        'crosswalk_alternatives_audited_m':[2,5],'token_semantics_broadcast_to_all_2m_segments':True,
        'actor_primary_relation':'existing Stage3 lane_actor_index (segment start <50m), then any matching token metadata',
        'signal_state_fields_created':[],'raw_topology_gap_records_excluded_from_topology_features':True,
        'yield_candidate':'raw audit retained; dedicated future feature not recommended with only 3 source records',
        'future_group_definitions_frozen':False,'VAL_performance_threshold_selection':False})
    atomic_json(ROOT/'02_semantic_cache/stage7a_cache_manifest.json',{'status':'AUDIT_ONLY','source_files':read_json(ROOT/'01_data_audit/stage7a_map_schema_audit.json')['locations'],
        'token_count':sum(len(v) for v in cache.values()),'metadata_sha256':sha256(ROOT/'02_semantic_cache/stage7a_semantic_metadata.json'),
        'centerlines_sha256':sha256(ROOT/'02_semantic_cache/stage7a_centerlines.npz'),'original_extractor_cache_overwritten':False})
    print('SEMANTIC_CACHE_DONE',sum(len(v) for v in cache.values()),flush=True)
if __name__=='__main__':main()
