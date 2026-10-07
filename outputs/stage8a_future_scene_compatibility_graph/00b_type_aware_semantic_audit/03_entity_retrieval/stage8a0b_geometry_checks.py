"""Independent full-map brute-force comparisons and non-endpoint geometry checks."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'00_manifest'))
from stage8a0b_common import *
from stage8a0b_entities import EntityIndex

def main():
    index=EntityIndex();manifest=read_json(STAGE8/'02_graph_cache/stage8a_graph_cache_manifest.json')
    checked=set();rows=[];rng=np.random.default_rng(2022)
    for b in manifest['batches']:
        if len(checked)==4:break
        with np.load(STAGE8/'02_graph_cache'/b['path']) as z:
            locations=z['window_location'].copy();origins=z['window_origin'].copy();yaws=z['window_yaw'].copy()
        unseen=[j for j,l in enumerate(locations) if list(index.regions)[int(l)] not in checked]
        if not unseen:continue
        payload=torch.load(STAGE6/b['source_cache_path'],map_location='cpu',weights_only=False)
        for j in unseen:
            location=list(index.regions)[int(locations[j])]
            if location in checked:continue
            c=payload['windows'][j];w=observable_window(c,location,origins[j],float(yaws[j]))
            selected=np.flatnonzero(~c['history_padding'][:,4].numpy());local=c['ego_prediction'][selected].numpy().reshape(-1,12,2)
            take=rng.choice(len(local),min(3,len(local)),replace=False);points=ego_to_global(local[take],w.origin,w.yaw)
            types=np.repeat(c['agent_type'][selected].numpy(),6)[take];result=index.retrieve(points,location,types)
            region=index.regions[location]
            for k,pts in enumerate(points):
                line=shapely.LineString(pts);hits=[];counts=np.zeros(7,dtype=np.int32)
                # No spatial index; evaluate every original usable geometry component.
                for e in region['entities']:
                    if e['type_id']<2:
                        d=line.distance(e['geometry']);match=d<=10.
                    else:
                        match=any(any(pg.contains(shapely.Point(p)) for p in pts) or line.intersects(pg)
                            or line.distance(pg.boundary)<=2. for pg in e['components'])
                        d=min(line.distance(pg) for pg in e['components'])
                    if match:
                        counts[e['type_id']]+=1
                        if e['type_id']<6:hits.append((d,e['type_id'],e['token'],e['id']))
                hits.sort();top=hits[:8];rel=[x for x in hits if RELEVANT[int(types[k]),x[1]]][:8]
                assert np.array_equal(counts,result['counts'][k])
                for name,items in [('all',top),('relevant',rel)]:
                    assert np.array_equal(result[name+'_ids'][k,:len(items)],np.array([x[3] for x in items]))
                    assert np.allclose(result[name+'_distance'][k,:len(items)],np.array([x[0] for x in items]),rtol=0,atol=1e-12)
                rows.append({'MapRegion':location,'DatasetIndex':c['dataset_index'],'CandidateIndex':int(take[k]),
                    'Counts':'PASS','Top8IDs':'PASS','Top8Distances':'PASS','EntitiesChecked':len(region['entities'])})
            checked.add(location)
    assert len(checked)==4 and len(rows)==12
    square=shapely.box(0,0,10,10)
    interior=shapely.LineString([(4,4),(6,6)])
    crossing=shapely.LineString([(-5,5),(15,5)])
    boundary2=shapely.LineString([(-2,1),(-2,9)])
    outside=shapely.LineString([(-2.000001,1),(-2.000001,9)])
    assert interior.distance(square.boundary)>2 and interior.intersects(square)
    assert not square.contains(shapely.Point(crossing.coords[0])) and not square.contains(shapely.Point(crossing.coords[-1])) and crossing.intersects(square)
    assert boundary2.distance(square.boundary)==2 and outside.distance(square.boundary)>2
    lane=shapely.LineString([(0,0),(100,0)]);candidate=shapely.LineString([(50,-1),(50,1)])
    assert candidate.distance(lane)==0 and min(candidate.distance(shapely.Point(p)) for p in lane.coords)>10
    write_csv(ROOT/'06_tables/stage8a0b_geometry_bruteforce.csv',rows)
    atomic_json(ROOT/'03_entity_retrieval/stage8a0b_geometry_audit.json',{'status':'PASS','regions':4,
        'bruteforce_candidates':12,'all_components_evaluated_without_STRtree':True,
        'inside_far_from_boundary':'PASS','intersection_with_both_endpoints_outside':'PASS',
        'exact2m_boundary_inclusion':'PASS','greater_than2m_exclusion':'PASS','complete_lane_segment_interior':'PASS',
        'distinct_entity_dedup_and_geometric_top8':'PASS','threshold_sweep':False})
    print('GEOMETRY PASS',flush=True)

if __name__=='__main__':main()
