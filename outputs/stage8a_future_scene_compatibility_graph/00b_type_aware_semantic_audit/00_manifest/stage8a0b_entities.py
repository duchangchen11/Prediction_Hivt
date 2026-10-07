"""Exact geometry retrieval, including conservative valid polygon components.

No labels, future masks or GT are accepted by this module's retrieval API.
"""
from stage8a0b_common import *
from nuscenes.map_expansion.map_api import NuScenesMap
from shapely.strtree import STRtree
from collections import Counter

class EntityIndex:
    def __init__(self, write_audit=False):
        self.lanes=MapIndex(); self.regions={}; self.dictionary=[]; schema=[]; component_rows=[]; reference_rows=[]
        self.semantic_pass=True
        for location, lane_region in self.lanes.regions.items():
            raw=read_json(MAP_JSON/(location+'.json'))
            api=NuScenesMap(dataroot=str(MAP_ROOT),map_name=location)
            tokens_by_layer={k:{r['token'] for r in v if isinstance(r,dict) and 'token' in r}
                             for k,v in raw.items() if isinstance(v,list)}
            polygons={r['token']:r for r in raw['polygon']}
            nodes={r['token']:r for r in raw['node']}
            entities=[]; components=[]; component_entity=[]; source_polygons=[]
            for type_id,layer in enumerate(TYPES):
                records=raw.get(layer,[]); counts=Counter(); geom_types=Counter(); fields=set()
                for r in sorted(records,key=lambda r:r['token']):
                    fields.update(r)
                    bad_refs=0
                    # Actual source association references: geometry and non-geometry.
                    for field,value in r.items():
                        if field=='token': continue
                        ref_layer={'polygon_token':'polygon','polygon_tokens':'polygon',
                            'from_edge_line_token':'line','to_edge_line_token':'line',
                            'road_block_token':'road_block','road_segment_token':'road_segment',
                            'ped_crossing_tokens':'ped_crossing','traffic_light_tokens':'traffic_light'}.get(field)
                        if ref_layer:
                            values=value if isinstance(value,list) else [value]
                            for token in values:
                                if token and token not in tokens_by_layer.get(ref_layer,set()):
                                    bad_refs+=1;reference_rows.append({'MapRegion':location,'Layer':layer,'Token':r['token'],
                                        'Field':field,'ReferencedToken':token,'ReferencedLayer':ref_layer})
                    counts['InvalidReference']+=bad_refs
                    if type_id<2:
                        i=lane_region['tokens'].index(r['token']) if r['token'] in lane_region['tokens'] else -1
                        if i<0: counts['InvalidGeometry']+=1;counts['EmptyGeometry']+=1;continue
                        geometry=lane_region['geometries'][i]
                        finite=np.isfinite(lane_region['coordinates'][i]).all()
                        valid=bool(shapely.is_valid(geometry)) and finite and not geometry.is_empty
                        counts['ValidGeometry' if valid else 'InvalidGeometry']+=1;counts['UsableGeometry']+=int(valid)
                        geom_types[geometry.geom_type]+=1
                        semantic=lane_region['semantic'][i]
                        self.semantic_pass &= np.array_equal(semantic,np.asarray(self.lanes.token_dictionary[int(lane_region['ids'][i])]['semantic'],dtype=np.float32))
                        entities.append({'token':r['token'],'type_id':type_id,'semantic':semantic,'geometry':geometry,'components':[geometry]})
                        continue
                    refs=r.get('polygon_tokens',[r['polygon_token']] if 'polygon_token' in r else [])
                    usable=[]; invalid=0; empty=0; nnan=0; ninf=0; closed=True
                    for pt in refs:
                        info=polygons.get(pt); issue='';geometry=None
                        if info is None: issue='missing_polygon';invalid+=1
                        else:
                            rings=[info.get('exterior_node_tokens',[])]+[h['node_tokens'] for h in info.get('holes',[])]
                            missing=[nt for ring in rings for nt in ring if nt not in nodes]
                            if missing:
                                counts['InvalidReference']+=len(missing);invalid+=1;issue='missing_node'
                                for nt in missing: reference_rows.append({'MapRegion':location,'Layer':layer,'Token':r['token'],
                                    'Field':'polygon_node','ReferencedToken':nt,'ReferencedLayer':'node'})
                            else:
                                xy=[np.asarray([[nodes[nt]['x'],nodes[nt]['y']] for nt in ring],dtype=np.float64).reshape(-1,2) for ring in rings]
                                nnan+=sum(int(np.isnan(x).sum()) for x in xy);ninf+=sum(int(np.isinf(x).sum()) for x in xy)
                                if any(not np.isfinite(x).all() for x in xy): invalid+=1;issue='nonfinite'
                                else:
                                    try:
                                        geometry=api.extract_polygon(pt)
                                        closed &= bool(geometry.exterior.is_closed) and all(h.is_closed for h in geometry.interiors)
                                        if geometry.is_empty: empty+=1;invalid+=1;issue='empty'
                                        elif not geometry.is_valid or geometry.area<=0:
                                            invalid+=1;issue=str(shapely.is_valid_reason(geometry))
                                        else: usable.append(geometry)
                                    except (ValueError,KeyError,TypeError) as e: invalid+=1;issue=type(e).__name__+': '+str(e)
                        component_rows.append({'MapRegion':location,'Layer':layer,'EntityToken':r['token'],
                            'PolygonToken':pt,'Usable':int(not issue),'Issue':issue})
                    counts['PolygonComponents']+=len(refs);counts['InvalidComponents']+=invalid
                    counts['NaN']+=nnan;counts['Inf']+=ninf;counts['EmptyGeometry']+=int(empty>0 or not refs)
                    valid=bool(refs) and invalid==0 and closed
                    counts['ValidGeometry' if valid else 'InvalidGeometry']+=1
                    counts['ClosedGeometry']+=int(closed and bool(usable));counts['UsableGeometry']+=int(bool(usable))
                    counts['PartiallyUsableGeometry']+=int(bool(usable) and not valid)
                    geom_types['Polygon' if len(refs)==1 else 'PolygonCollection']+=1
                    if usable:
                        # Preserve components exactly. No union, repair, buffer(0), centroid or endpoint approximation.
                        geometry=usable[0] if len(usable)==1 else shapely.GeometryCollection(usable)
                        entities.append({'token':r['token'],'type_id':type_id,'semantic':np.zeros(9,dtype=np.float32),
                                         'geometry':geometry,'components':usable})
                schema.append({'MapRegion':location,'Layer':layer,'Status':'AVAILABLE' if layer in raw else 'MISSING',
                    'Records':len(records), **{key:int(counts[key]) for key in ['ValidGeometry','InvalidGeometry','EmptyGeometry','NaN','Inf','InvalidReference','ClosedGeometry','UsableGeometry','PartiallyUsableGeometry','PolygonComponents','InvalidComponents']},
                    'GeometryType':json.dumps(dict(geom_types),sort_keys=True),'ActualFields':','.join(sorted(fields))})
            # Type then token gives the mandated deterministic tie-break order.
            entities.sort(key=lambda e:(e['type_id'],e['token']))
            for local_id,e in enumerate(entities):
                e['id']=len(self.dictionary)
                self.dictionary.append({'index':e['id'],'location':location,'token':e['token'],'type_id':e['type_id'],
                    'type':TYPES[e['type_id']],'semantic':e['semantic'].tolist()})
                if e['type_id']>=2:
                    for g in e['components']:components.append(g);component_entity.append(local_id)
            lane_entities=np.array([i for i,e in enumerate(entities) if e['type_id']<2],dtype=np.int32)
            polygon_entities=np.asarray(component_entity,dtype=np.int32)
            polygons_array=np.asarray(components,dtype=object)
            shapely.prepare(polygons_array)
            self.regions[location]={'entities':entities,'types':np.array([e['type_id'] for e in entities],dtype=np.int8),
                'ids':np.array([e['id'] for e in entities],dtype=np.int32),
                'geometries':np.asarray([e['geometry'] for e in entities],dtype=object),
                'lane_entities':lane_entities,'lane_tree':STRtree([entities[i]['geometry'] for i in lane_entities]),
                'polygon_entities':polygon_entities,'polygons':polygons_array,'polygon_tree':STRtree(polygons_array)}
        self.schema=schema;self.component_rows=component_rows;self.reference_rows=reference_rows
        assert self.semantic_pass
        assert all(not any(e['semantic']) for e in self.dictionary if e['type_id']>=2)
        if write_audit:
            write_csv(ROOT/'06_tables/stage8a0b_map_schema.csv',schema)
            write_csv(ROOT/'01_schema_audit/stage8a0b_polygon_components.csv',component_rows)
            if reference_rows:write_csv(ROOT/'01_schema_audit/stage8a0b_invalid_references.csv',reference_rows)
            atomic_json(ROOT/'03_entity_retrieval/stage8a0b_entity_dictionary.json',self.dictionary)
            atomic_json(ROOT/'01_schema_audit/stage8a0b_schema_audit.json',{'status':'PASS','regions':list(self.regions),
                'map_sources':{p.name:sha256(p) for p in sorted(MAP_JSON.glob('*.json'))},
                'layers':list(TYPES),'geometry_policy':'Exact unmodified valid polygon components; exclude invalid components, retaining and disclosing usable components of multipart records.',
                'entities':len(self.dictionary),'invalid_components':sum(x['InvalidComponents'] for x in schema),
                'invalid_references':sum(x['InvalidReference'] for x in schema),'partial_entities':sum(x['PartiallyUsableGeometry'] for x in schema),
                'semantic_attachment':'PASS','frozen_9d_source_sha256':sha256(SEMANTICS),'centerlines_sha256':sha256(CENTERLINES),
                'software':{'shapely':shapely.__version__,'GEOS':shapely.geos_version_string},'repairs':False})

    def retrieve(self, points, location, actor_types):
        """All matching tokens, exact distance, capped all/relevant selectors; no fallback."""
        points=np.asarray(points,dtype=np.float64); assert points.ndim==3 and points.shape[1:]==(12,2)
        assert np.isfinite(points).all()
        lines=shapely.linestrings(points); region=self.regions[location]; p=len(points)
        lane_pair=region['lane_tree'].query(lines,predicate='dwithin',distance=10.)
        poly_pair=region['polygon_tree'].query(lines,predicate='dwithin',distance=2.)
        rr=[];ee=[];dd=[]
        if lane_pair.size:
            e=region['lane_entities'][lane_pair[1]]
            rr.append(lane_pair[0]);ee.append(e);dd.append(shapely.distance(lines[lane_pair[0]],region['geometries'][e]))
        if poly_pair.size:
            pg=region['polygons'][poly_pair[1]];ll=lines[poly_pair[0]]
            hit=shapely.intersects(pg,ll)
            d=np.zeros(len(pg),dtype=np.float64)
            d[~hit]=shapely.distance(ll[~hit],pg[~hit])
            boundary=shapely.distance(ll[~hit],shapely.boundary(pg[~hit]))
            assert (boundary<=2.+1e-9).all() and np.isfinite(boundary).all()
            # dwithin on filled polygons implements the exact OR rule. Interior/line intersections
            # qualify independently of distance to boundary; outside polygon, d == boundary distance.
            assert np.allclose(d[~hit],boundary,atol=1e-8)
            rr.append(poly_pair[0]);ee.append(region['polygon_entities'][poly_pair[1]]);dd.append(d)
        counts=np.zeros((p,7),dtype=np.int32)
        all_ids=np.full((p,8),-1,dtype=np.int32);all_dist=np.zeros((p,8),dtype=np.float64)
        rel_ids=all_ids.copy();rel_dist=all_dist.copy()
        if rr:
            r=np.concatenate(rr);e=np.concatenate(ee);d=np.concatenate(dd)
            assert np.isfinite(d).all()
            # Collapse multiple polygon components to one entity with minimum exact distance.
            order=np.lexsort((d,e,r));r=r[order];e=e[order];d=d[order]
            first=np.r_[True,(r[1:]!=r[:-1])|(e[1:]!=e[:-1])];r=r[first];e=e[first];d=d[first]
            ty=region['types'][e];np.add.at(counts,(r,ty),1)
            order=np.lexsort((e,d,r));r=r[order];e=e[order];d=d[order];ty=ty[order]
            def cap(mask,ids,dist):
                ar=r[mask];ae=e[mask];ad=d[mask]
                c=np.bincount(ar,minlength=p);offsets=np.repeat(np.cumsum(np.r_[0,c[:-1]]),c)
                rank=np.arange(len(ar))-offsets;keep=rank<8
                ids[ar[keep],rank[keep]]=region['ids'][ae[keep]];dist[ar[keep],rank[keep]]=ad[keep]
            cap(ty<6,all_ids,all_dist)
            relevant=(ty<6)&RELEVANT[np.asarray(actor_types)[r],np.minimum(ty,5)]
            cap(relevant,rel_ids,rel_dist)
        any_coverage=(counts[:,:6]>0)&RELEVANT[np.asarray(actor_types)]
        return {'counts':counts,'all_ids':all_ids,'all_distance':all_dist,'relevant_ids':rel_ids,
            'relevant_distance':rel_dist,'relevant_counts':(counts[:,:6]*RELEVANT[np.asarray(actor_types)]).sum(-1),
            'type_aware_coverage':any_coverage.any(-1)}

    def build(self, w):
        selected=torch.where(~w.history_padding[:,4])[0].numpy()
        points=ego_to_global(w.predicted[selected].numpy().reshape(-1,12,2),w.origin,w.yaw)
        result=self.retrieve(points,w.map_location,np.repeat(w.actor_type[selected].numpy(),6))
        return selected,{k:v.reshape((len(selected),6)+v.shape[1:]) for k,v in result.items()}
