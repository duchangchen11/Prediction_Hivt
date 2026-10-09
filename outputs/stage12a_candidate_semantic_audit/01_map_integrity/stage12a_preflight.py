"""Fixed region scenes: exact selector replay, coordinate and GT-poison gates."""
from pathlib import Path
import sys,copy
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'00_manifest'),str(Path(__file__).resolve().parents[1]/'02_semantic_cache')]
from stage12a_common import *
from stage12a_features import from_observed_window
def geometry_checks():
    polygon=shapely.Polygon([(0,0),(2,0),(2,2),(0,2)])
    assert polygon.contains(shapely.Point(1,1)) and not polygon.contains(shapely.Point(0,1))
    crossing=shapely.LineString([(-1,1),(3,1)]);inside=shapely.LineString([(.5,1),(1.5,1)])
    outside=shapely.LineString([(-1,3),(3,3)])
    assert crossing.intersects(polygon) and not outside.intersects(polygon)
    assert crossing.crosses(polygon.boundary) and not inside.crosses(polygon.boundary)
    assert abs(outside.distance(polygon.boundary)-1)<1e-12 and abs(inside.distance(polygon.boundary)-.5)<1e-12
    assert abs(shapely.LineString([(0,0),(10,0)]).distance(shapely.Point(5,2))-2)<1e-12
    # Overlapping source components remain separate; strict containment is OR.
    a=shapely.Polygon([(0,0),(1,0),(1,1),(0,1)]);b=shapely.Polygon([(1,0),(2,0),(2,1),(1,1)])
    assert not any(part.contains(shapely.Point(1,.5)) for part in [a,b])
    return {'PolygonInside':'PASS','PolylineIntersection':'PASS','BoundaryDistance':'PASS','BoundaryCrossing':'PASS',
        'LineSegmentInteriorDistance':'PASS','OriginalComponentOR':'PASS'}
def main():
    os.environ['STAGE12A_PHASE']='extract';torch.set_num_threads(1);verify()
    manifest=read_json(ROOT/'01_map_integrity/cache/stage12a_frame_manifest.json');assert manifest['Status']=='PASS'
    index=SparseSemanticIndex();regions=sorted(index.regions)
    expected=read_json(S8C/'02_graph_cache/stage8a0c_entity_dictionary.json');assert index.dictionary==expected
    f=observed_identities();cache=ROOT/'01_map_integrity/cache';arrays={k:np.load(cache/f'stage12a_{k}.npy',mmap_mode='r') for k in ['origin','yaw','location','entity_ids','entity_types','geometry_distances']}
    scene_regions=f.assign(map_region=np.array(regions)[arrays['location']]).groupby('map_region').scene_token.min()
    chosen=[]
    for loc,scene in scene_regions.items():
        for t in ACTOR_TYPES:
            ii=np.flatnonzero((f.scene_token.to_numpy()==scene)&(f.agent_type.to_numpy()==t))
            chosen.extend(sorted(ii,key=lambda i:f.iloc[i].actor_id)[:16])
    chosen=np.array(sorted(set(chosen)),np.int64);assert len(chosen)>0
    roundtrip=0.;edge_max=0.;actor_count=0;selector_count=0;poison_count=0;details=[]
    node=np.load(TRAIN/'arg3.npy',mmap_mode='r');edge=np.load(TRAIN/'arg4.npy',mmap_mode='r')
    original=np.load(S11A/'01_identity_audit/cache/candidates.npy',mmap_mode='r')
    for path,within in f.iloc[chosen].groupby('cache_path',sort=False).indices.items():
        ids=chosen[within];payload=torch.load(S6/path,map_location='cpu',weights_only=False);windows={int(w['dataset_index']):w for w in payload['windows']}
        for di,inside in f.iloc[ids].groupby('dataset_index',sort=False).indices.items():
            ix=ids[inside];w=windows[int(di)];nodes=f.iloc[ix].node_in_graph.to_numpy();loc=regions[int(arrays['location'][ix[0]])]
            origin=arrays['origin'][ix[0]];yaw=float(arrays['yaw'][ix[0]])
            points,selection,feature,sem,valid=from_observed_window(index,w,origin,yaw,loc,nodes)
            local=original[f.iloc[ix].source_index.to_numpy()].reshape(-1,12,2)
            error=float(np.abs(global_to_ego(points,origin,yaw)-local).max());roundtrip=max(roundtrip,error)
            # Independent positive-forward / left rotation orientation checks.
            axes=ego_to_global(np.array([[1.,0.],[0.,1.]]),origin,yaw)-origin
            assert np.allclose(axes,[[np.cos(yaw),np.sin(yaw)],[-np.sin(yaw),np.cos(yaw)]],atol=1e-12,rtol=0)
            assert np.array_equal(selection['entity_ids'],arrays['entity_ids'][ix].reshape(-1,8))
            assert np.array_equal(selection['entity_types'],arrays['entity_types'][ix].reshape(-1,8))
            assert np.allclose(selection['geometry_distances'],arrays['geometry_distances'][ix].reshape(-1,8),atol=1e-8,rtol=0)
            src=f.iloc[ix].source_index.to_numpy();assert np.array_equal(feature['map_node'],node[src].reshape(-1,8,18))
            assert np.array_equal(feature['map_edge'],edge[src].reshape(-1,8,11))
            poisoned=copy.deepcopy(w);poisoned['GT'].fill_(float('nan'))
            poisoned['future_mask']=~poisoned['future_mask'];poisoned['target_mask']=~poisoned['target_mask']
            if 'motion_state' in poisoned:poisoned['motion_state']=['POISON']*len(poisoned['motion_state'])
            _,other,otherfeature,othersem,othervalid=from_observed_window(index,poisoned,origin,yaw,loc,nodes)
            for k in selection:assert np.array_equal(selection[k],other[k]),k
            for k in feature:assert np.array_equal(feature[k],otherfeature[k]),k
            assert np.array_equal(sem,othersem) and np.array_equal(valid,othervalid)
            # Added polygon OR features agree with direct original-component tests.
            for r in range(len(points)):
                for label,ty in [('drivable',2),('carpark',3),('crosswalk',4),('walkway',5)]:
                    entids=selection['entity_ids'][r,(selection['entity_types'][r]==ty)&selection['map_mask'][r]]
                    parts=[p for eid in entids for p in index.regions[loc]['entities'][index.regions[loc]['id_to_local'][int(eid)]]['parts']]
                    if not parts:continue
                    direct=np.mean([any(g.contains(shapely.Point(xy)) for g in parts) for xy in points[r]])
                    assert abs(sem[r,SEM_FIELDS.index(label+'_inside_fraction')]-direct)<1e-12
            actor_count+=len(ix);selector_count+=len(points);poison_count+=len(points)
            details.append({'Scene':w['scene_token'],'Sample':w['sample_token'],'Region':loc,'Actors':len(ix),'RoundtripMaxM':error})
    assert roundtrip<1e-5
    sources={p.name:sha256(p) for p in MAP_JSON.glob('*.json')}
    mapaudit={'Status':'PASS','MapIntegrity':'PASS','Regions':regions,'Entities':len(index.dictionary),'StaticTypes':MAP_TYPES,
        'MapJSONSHA256':sources,'MapJSONRealPath':str(MAP_JSON.resolve()),'OriginalHDMapSource':'unchanged local nuScenes expansion JSON, same Stage8 source hashes',
        'SelectorSourceSHA256':sha256(S8C/'01_selector/stage8a0c_selector.py'),'FrozenDictionaryIdentity':True,
        'InvalidComponentsIsolated':index.invalid_components,'GeometryRepairs':False,'NearestFallback':False,
        'SelectorCacheReplay':'bitwise entity IDs/types/map_node/map_edge; original FP32 values unchanged',
        'PreflightActors':actor_count,'PreflightCandidates':selector_count,'GeometryTests':geometry_checks()}
    atomic_json(ROOT/'01_map_integrity/stage12a_map_integrity.json',mapaudit)
    atomic_json(ROOT/'01_map_integrity/stage12a_coordinate_audit.json',{'Status':'PASS','CoordinateAudit':'PASS',
        'FixedScenePerRegion':scene_regions.to_dict(),'Actors':actor_count,'Candidates':selector_count,'RoundtripMaxDiffM':roundtrip,
        'ThresholdM':1e-5,'PositiveYawCounterclockwise':True,'OriginYawMapLocation':'frozen Stage8 scene/sample frame with original Stage6 identities',
        'Stage8HistoricalCoordinateAuditSHA256':sha256(S8/'01_cache_audit/stage8a_coordinate_audit.json'),'Details':details})
    atomic_json(ROOT/'01_map_integrity/stage12a_gt_poison_audit.json',{'Status':'PASS','GTLeakage':'PASS','Candidates':poison_count,
        'GTReplacedWithNaN':True,'FutureAndTargetMaskPoisoned':True,'MotionLabelsPoisoned':True,'SelectionsAndFeaturesBitwiseEqual':True,
        'ExtractorAPI':'observation allowlist and frozen actor nodes; no GT/FDE/ADE/future motion parameters','ExplicitFutureLabelFilesOpened':False})
    atomic_json(ROOT/'01_map_integrity/stage12a_preflight_gate.json',{'Status':'PASS','Map':'PASS','Coordinate':'PASS','Poison':'PASS',
        'FrozenSelectorAndFeatureReplay':'PASS','ProceedFull630':True,'SelectedHeadTrainIndices':chosen.tolist()})
    print('MAP_COORDINATE_GT_POISON_PREFLIGHT_PASS',actor_count,selector_count,roundtrip,flush=True)
if __name__=='__main__':main()
