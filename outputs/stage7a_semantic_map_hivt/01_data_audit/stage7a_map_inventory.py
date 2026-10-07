"""Actual raw JSON/API inventory, before adopting any turn thresholds."""
from pathlib import Path
import sys,collections,inspect
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'00_manifest'))
from stage7a_common import *
import numpy as np
from nuscenes.map_expansion.map_api import NuScenesMap
from shapely.geometry import LineString
from preprocessing.coordinates import wrap_angle

LAYERS=('lane','lane_connector','stop_line','traffic_light','ped_crossing','walkway','road_segment','road_block','lane_divider','road_divider','carpark_area')
BINS=[-180,-135,-90,-60,-30,-15,15,30,60,90,135,180]
def typed(v):
    if v is None:return 'null'
    if isinstance(v,bool):return 'bool'
    if isinstance(v,(int,float)):return 'number'
    if isinstance(v,list):return 'list'
    if isinstance(v,dict):return 'object'
    return 'string'
def schema(records):
    names=sorted({k for r in records for k in r})
    return {k:{'present':sum(k in r for r in records),'non_null':sum(r.get(k) is not None for r in records),
        'types':sorted({typed(r[k]) for r in records if k in r})} for k in names}
def load_maps():
    for path in sorted(MAP_JSON.glob('*.json')):
        raw=read_json(path)
        api=NuScenesMap(dataroot=str(MAP_ROOT),map_name=path.stem)
        yield path,raw,api
def polygon_summary(api,records):
    out=[]
    for r in records:
        p=api.extract_polygon(r['polygon_token']);coords=np.asarray(p.exterior.coords)
        out.append({'token':r['token'],'valid':bool(p.is_valid),'empty':bool(p.is_empty),
            'finite':bool(np.isfinite(coords).all()),'area_m2':float(p.area)})
    return out
def main():
    verify_v1();layer_rows=[];schema_all={};geo={};angles=[];hist=[];centerlines={};stopstats=[];quality=[];topology_issues=[]
    for path,raw,api in load_maps():
        loc=path.stem;print('INVENTORY',loc,flush=True)
        schema_all[loc]={'source_path':str(path.resolve()),'sha256':sha256(path),'version':raw['version'],
            'layers':{},'top_level_fields':sorted(raw)}
        for layer in LAYERS:
            records=raw[layer];assert len(records)==len(getattr(api,layer)),(loc,layer)
            layer_rows.append({'location':loc,'layer':layer,'record_count':len(records)})
            schema_all[loc]['layers'][layer]={'raw_record_fields':schema(records),
                'api_record_fields':schema(getattr(api,layer)), 'count':len(records)}
        schema_all[loc]['layers']['traffic_light']['items_fields']=schema([v for r in raw['traffic_light'] for v in r.get('items',[])])
        schema_all[loc]['layers']['traffic_light']['pose_fields']=schema([r.get('pose',{}) for r in raw['traffic_light']])
        tokens={r['token']:layer for layer in ('lane','lane_connector') for r in raw[layer]}
        assert len(tokens)==len(raw['lane'])+len(raw['lane_connector'])
        lines=api.discretize_lanes(list(tokens),2.0);meta={}
        for token,layer in tokens.items():
            line=np.asarray(lines.get(token,[]),dtype=np.float64)
            assert line.ndim==2 and len(line)>=2 and line.shape[1]>=2,(loc,token)
            line=line[:,:2];assert np.isfinite(line).all();centerlines[loc+'__'+token]=line
            delta=np.diff(line,axis=0);valid=np.linalg.norm(delta,axis=-1)>1e-6
            assert valid.any();vin=delta[valid][0];vout=delta[valid][-1]
            angle=float(np.degrees(wrap_angle(np.arctan2(vout[1],vout[0])-np.arctan2(vin[1],vin[0]))))
            arcs=raw['arcline_path_3'].get(token,[])
            exact=float(np.degrees(wrap_angle(arcs[-1]['end_pose'][2]-arcs[0]['start_pose'][2]))) if arcs else None
            topo=raw['connectivity'].get(token,{});refs=topo.get('incoming',[])+topo.get('outgoing',[])
            badrefs=[t for t in refs if t not in tokens]
            for t in badrefs:topology_issues.append({'location':loc,'token':token,'layer':layer,'unresolved_reference':t})
            meta[token]={'layer':layer,'is_connector':int(layer=='lane_connector'),'delta_theta_deg':angle if layer=='lane_connector' else None,
                'exact_arc_delta_deg':exact if layer=='lane_connector' else None,'segment_count':int(valid.sum()),
                'length_m':float(LineString(line).length),'incoming':topo.get('incoming',[]),'outgoing':topo.get('outgoing',[]),
                'centerline_finite':True,'connectivity_record_available':token in raw['connectivity'],'arcline_available':bool(arcs),
                'unresolved_connectivity_references':badrefs}
            if layer=='lane_connector':angles.append({'location':loc,'token':token,'delta_theta_deg':angle,'exact_arc_delta_deg':exact,
                'angle_difference_deg':float(abs(np.degrees(wrap_angle(np.radians(angle-exact))))) if exact is not None else None,
                'incoming_count':len(topo.get('incoming',[])),'outgoing_count':len(topo.get('outgoing',[])),
                'segment_count':int(valid.sum()),'length_m':float(LineString(line).length),'unresolved_connectivity_count':len(badrefs)})
        locangles=[r['delta_theta_deg'] for r in angles if r['location']==loc];counts=np.histogram(locangles,bins=BINS)[0]
        for i,n in enumerate(counts):hist.append({'location':loc,'bin_low_deg':BINS[i],'bin_high_deg':BINS[i+1],
            'right_closed':i==len(counts)-1,'connector_count':int(n)})
        # Preserve source geometry failures; never repair or fabricate associations.
        polys={layer:polygon_summary(api,raw[layer]) for layer in ('lane','lane_connector','stop_line','ped_crossing')}
        for layer,rs in polys.items():
            quality.append({'location':loc,'layer':layer,'record_count':len(rs),'invalid_polygons':sum(not r['valid'] for r in rs),
                'empty_polygons':sum(r['empty'] for r in rs),'nonfinite_polygons':sum(not r['finite'] for r in rs),'zero_area_polygons':sum(r['area_m2']<=0 for r in rs)})
        tl={r['token'] for r in raw['traffic_light']};pc={r['token'] for r in raw['ped_crossing']};rb={r['token'] for r in raw['road_block']}
        for typ in sorted({r['stop_line_type'] for r in raw['stop_line']}):
            records=[r for r in raw['stop_line'] if r['stop_line_type']==typ];areas=np.array([api.extract_polygon(r['polygon_token']).area for r in records]);trefs=[t for r in records for t in r.get('traffic_light_tokens',[])];prefs=[t for r in records for t in r.get('ped_crossing_tokens',[])]
            badt=sum(t not in tl for t in trefs);badp=sum(t not in pc for t in prefs);badr=sum(bool(r.get('road_block_token')) and r['road_block_token'] not in rb for r in records)
            stopstats.append({'location':loc,'stop_line_type':typ,'record_count':len(records),'area_min_m2':float(areas.min()),
                'area_median_m2':float(np.median(areas)),'area_max_m2':float(areas.max()),'area_mean_m2':float(areas.mean()),
                'traffic_light_reference_count':len(trefs),'unique_traffic_light_tokens':len(set(trefs)),
                'ped_crossing_reference_count':len(prefs),'unique_ped_crossing_tokens':len(set(prefs)),
                'invalid_traffic_light_references':badt,'invalid_ped_crossing_references':badp,'invalid_road_block_references':badr})
        schema_all[loc]['unavailable_fields']={'lane.turn_type':'NOT AVAILABLE','lane_connector.turn_type':'NOT AVAILABLE',
            'stop_line.lane_token':'NOT AVAILABLE','traffic_light.lane_token':'NOT AVAILABLE','traffic_light.current_signal_state':'NOT AVAILABLE',
            'traffic_light.future_signal_state':'NOT AVAILABLE'}
        schema_all[loc]['traffic_light_static_pose_quality']={'all_zero_xy_count':sum(r.get('pose',{}).get('tx')==0 and r.get('pose',{}).get('ty')==0 for r in raw['traffic_light']),
            'items_color_is_static_lens_descriptor':True,'dynamic_signal_state_available':False}
        geo[loc]=meta
    np.savez_compressed(ROOT/'02_semantic_cache/stage7a_centerlines.npz',**centerlines)
    atomic_json(ROOT/'02_semantic_cache/stage7a_geometry_inventory.json',geo)
    atomic_json(ROOT/'01_data_audit/stage7a_source_topology_gaps.json',{'unresolved_reference_count':len(topology_issues),'records':topology_issues,'repaired':False})
    atomic_json(ROOT/'01_data_audit/stage7a_map_schema_audit.json',{'status':'RAW_AND_API_READ','locations':schema_all,
        'source_map_api':{'path':inspect.getfile(NuScenesMap),'sha256':sha256(inspect.getfile(NuScenesMap))},
        'thresholds_adopted':False,'source_map_root':str(MAP_ROOT),'raw_trainval_mount_available':Path('/media/lrj/54926A1D926A0438/nuscenes-trainval').exists()})
    for filename,rows in [('stage7a_map_layer_statistics.csv',layer_rows),('stage7a_connector_angles.csv',angles),
        ('stage7a_connector_angle_histogram.csv',hist),('stage7a_stop_line_raw_statistics.csv',stopstats),('stage7a_map_geometry_quality.csv',quality)]:write_csv(ROOT/'06_tables'/filename,rows)
    print('INVENTORY_DONE',len(centerlines),len(angles),flush=True)
if __name__=='__main__':main()
