"""Deterministic sparse semantic selector: geometry, actor type and HD map only.

Candidates passed to selection are twelve predicted points in global coordinates.
Motion attributes, supervision and prediction errors are not accepted here.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'00_manifest'))
from stage8a0c_common import TYPES,QUOTAS,MAP_ROOT,MAP_JSON,CENTERLINES,SEMANTICS,read_json,rotation_matrix,MAP_NODE_FIELDS,MAP_EDGE_FIELDS
import numpy as np
import shapely
from shapely.strtree import STRtree
from nuscenes.map_expansion.map_api import NuScenesMap

def frozen_lane_semantic(meta):
    # Exact Stage7A semantic_vector logic, isolated from dataset/evaluation imports.
    connector=int(meta['is_connector']);controls=set(meta['control_types_present'])
    return np.array([connector,*[connector*int(meta['turn_type']==t) for t in ('left','straight','right','unknown')],
        int('traffic_light' in controls),int('stop_sign' in controls),int(bool(controls&{'other_control','yield'})),
        int(meta['near_ped_crossing'])],dtype=np.float32)

class SparseSemanticIndex:
    def __init__(self):
        metadata=read_json(SEMANTICS);self.regions={};self.dictionary=[];self.invalid_components=[]
        with np.load(CENTERLINES) as archive:
            for location,meta in sorted(metadata.items()):
                raw=read_json(MAP_JSON/(location+'.json'));api=NuScenesMap(dataroot=str(MAP_ROOT),map_name=location)
                entities=[]
                for ty,layer in enumerate(TYPES):
                    for record in sorted(raw.get(layer,[]),key=lambda x:x['token']):
                        token=record['token']
                        if ty<2:
                            assert token in meta and meta[token]['layer']==layer
                            coordinates=archive[location+'__'+token].copy();g=shapely.LineString(coordinates)
                            assert np.isfinite(coordinates).all() and g.is_valid and g.length>0
                            parts=[g];semantic=frozen_lane_semantic(meta[token])
                        else:
                            refs=record.get('polygon_tokens',[record['polygon_token']] if 'polygon_token' in record else [])
                            parts=[];coordinates=None;semantic=np.zeros(9,dtype=np.float32)
                            for pt in refs:
                                g=api.extract_polygon(pt)
                                if g.is_empty or not g.is_valid or g.area<=0 or not np.isfinite(shapely.get_coordinates(g)).all():
                                    self.invalid_components.append((location,layer,token,pt));continue
                                parts.append(g)
                            if not parts:continue
                            g=parts[0] if len(parts)==1 else shapely.GeometryCollection(parts)
                        entities.append({'type':ty,'token':token,'geometry':g,'parts':parts,'coordinates':coordinates,'semantic':semantic})
                entities.sort(key=lambda e:(e['type'],e['token']))
                for e in entities:
                    e['id']=len(self.dictionary)
                    self.dictionary.append({'index':e['id'],'location':location,'token':e['token'],'type_id':e['type'],
                        'entity_type':TYPES[e['type']],'semantic':e['semantic'].tolist()})
                lane_idx=np.array([i for i,e in enumerate(entities) if e['type']<2],dtype=np.int32)
                polygon_parts=[];part_entity=[]
                for i,e in enumerate(entities):
                    if e['type']>=2:
                        polygon_parts.extend(e['parts']);part_entity.extend([i]*len(e['parts']))
                pg=np.asarray(polygon_parts,dtype=object);shapely.prepare(pg)
                self.regions[location]={'entities':entities,'types':np.array([e['type'] for e in entities],dtype=np.int8),
                    'ids':np.array([e['id'] for e in entities],dtype=np.int32),'geometries':np.asarray([e['geometry'] for e in entities],dtype=object),
                    'lane_indices':lane_idx,'lane_tree':STRtree([entities[i]['geometry'] for i in lane_idx]),
                    'polygon_parts':pg,'part_entities':np.array(part_entity,dtype=np.int32),'polygon_tree':STRtree(pg),
                    'id_to_local':{e['id']:i for i,e in enumerate(entities)}}
        assert len(self.invalid_components)==1 and self.invalid_components[0][:2]==('boston-seaport','walkway')
        assert len(MAP_NODE_FIELDS)==18 and len(MAP_EDGE_FIELDS)==11

    def _matches(self,points,location):
        points=np.asarray(points,dtype=np.float64)
        assert points.ndim==3 and points.shape[1:]==(12,2) and np.isfinite(points).all()
        reg=self.regions[location];lines=shapely.linestrings(points);p=len(points)
        rr=[];ee=[];dd=[]
        pairs=reg['lane_tree'].query(lines,predicate='dwithin',distance=10.)
        if pairs.size:
            e=reg['lane_indices'][pairs[1]];d=shapely.distance(lines[pairs[0]],reg['geometries'][e])
            assert (d<=10.+1e-9).all();rr.append(pairs[0]);ee.append(e);dd.append(d)
        pairs=reg['polygon_tree'].query(lines,predicate='dwithin',distance=2.)
        if pairs.size:
            pg=reg['polygon_parts'][pairs[1]];ll=lines[pairs[0]];hit=shapely.intersects(pg,ll)
            d=np.zeros(len(pg),dtype=np.float64);d[~hit]=shapely.distance(ll[~hit],pg[~hit])
            assert (d<=2.+1e-9).all()
            rr.append(pairs[0]);ee.append(reg['part_entities'][pairs[1]]);dd.append(d)
        counts=np.zeros((p,6),dtype=np.int32)
        if not rr:return np.empty(0,dtype=np.int32),np.empty(0,dtype=np.int32),np.empty(0),counts
        r=np.concatenate(rr);e=np.concatenate(ee);d=np.concatenate(dd);assert np.isfinite(d).all()
        order=np.lexsort((d,e,r));r=r[order];e=e[order];d=d[order]
        first=np.r_[True,(r[1:]!=r[:-1])|(e[1:]!=e[:-1])];r=r[first];e=e[first];d=d[first]
        np.add.at(counts,(r,reg['types'][e]),1)
        return r,e,d,counts

    def select_batch(self,actor_types,predicted_candidates,map_location):
        actor_types=np.asarray(actor_types,dtype=np.int8);p=len(predicted_candidates)
        assert actor_types.shape==(p,) and np.isin(actor_types,[0,1,2]).all()
        r,e,d,counts=self._matches(predicted_candidates,map_location);reg=self.regions[map_location]
        def pack(ar,ae,ad):
            ids=np.full((p,8),-1,dtype=np.int32);ty=np.full((p,8),-1,dtype=np.int8);dist=np.zeros((p,8),dtype=np.float64)
            n=np.bincount(ar,minlength=p);offset=np.repeat(np.cumsum(np.r_[0,n[:-1]]),n);slot=np.arange(len(ar))-offset
            assert (n<=8).all()
            ids[ar,slot]=reg['ids'][ae];ty[ar,slot]=reg['types'][ae];dist[ar,slot]=ad
            return ids,ty,dist
        if len(r):
            ty=reg['types'][e];relevant=QUOTAS[actor_types[r],ty]>0
            order=np.lexsort((e,d,ty,r));qr=r[order];qe=e[order];qd=d[order];qt=ty[order]
            group=qr*6+qt;first=np.r_[True,group[1:]!=group[:-1]]
            rank=np.arange(len(qr))-np.maximum.accumulate(np.where(first,np.arange(len(qr)),0))
            keep=rank<QUOTAS[actor_types[qr],qt]
            quota=pack(qr[keep],qe[keep],qd[keep])
            order=np.lexsort((e,d,r));gr=r[order];ge=e[order];gd=d[order];gt=reg['types'][ge]
            def global8(primary):
                allow=QUOTAS[actor_types[gr],gt]>0 if primary else np.ones(len(gr),dtype=bool)
                ar=gr[allow];ae=ge[allow];ad=gd[allow];n=np.bincount(ar,minlength=p)
                offset=np.repeat(np.cumsum(np.r_[0,n[:-1]]),n);slot=np.arange(len(ar))-offset;keep=slot<8
                return pack(ar[keep],ae[keep],ad[keep])
            primary=global8(True);all_six=global8(False)
        else:
            empty=pack(np.empty(0,dtype=np.int32),np.empty(0,dtype=np.int32),np.empty(0));quota=primary=all_six=empty
        selected_counts=np.zeros((p,6),dtype=np.int32);mask=quota[0]>=0;ar,sl=np.where(mask)
        np.add.at(selected_counts,(ar,quota[1][ar,sl]),1)
        assert np.array_equal(selected_counts,np.minimum(counts,QUOTAS[actor_types]))
        assert np.array_equal((selected_counts>0),(counts>0)&(QUOTAS[actor_types]>0))
        assert np.array_equal(mask.any(-1),((counts>0)&(QUOTAS[actor_types]>0)).any(-1))
        assert (mask.sum(-1)<=QUOTAS[actor_types].sum(-1)).all()
        return {'entity_ids':quota[0],'entity_types':quota[1],'geometry_distances':quota[2],'map_mask':mask,
            'uncapped_counts':counts,'selected_counts':selected_counts,
            'global_top8_ids':primary[0],'global_top8_types':primary[1],
            'all_six_top8_ids':all_six[0],'all_six_top8_types':all_six[1]}

    def select_sparse_semantic_entities(self,actor_type,predicted_candidate,map_location):
        out=self.select_batch(np.array([actor_type]),np.asarray(predicted_candidate)[None],map_location)
        mask=out['map_mask'][0]
        return {'entity_ids':out['entity_ids'][0,mask],'entity_types':out['entity_types'][0,mask],
                'geometry_distances':out['geometry_distances'][0,mask]}

    def features(self,points,location,yaw,endpoint_heading,selection):
        points=np.asarray(points,dtype=np.float64);reg=self.regions[location];mask=selection['map_mask'];r,s=np.where(mask)
        node=np.zeros((len(points),8,18),dtype=np.float32);edge=np.zeros((len(points),8,11),dtype=np.float32)
        if not len(r):return {'map_node':node,'map_edge':edge}
        e=np.array([reg['id_to_local'][int(i)] for i in selection['entity_ids'][r,s]],dtype=np.int32)
        types=reg['types'][e];geometries=reg['geometries'][e];lines=shapely.linestrings(points[r])
        distance=shapely.distance(lines,geometries)
        assert np.allclose(distance,selection['geometry_distances'][r,s],rtol=0,atol=1e-8)
        point_distance=shapely.distance(shapely.points(points[r]),geometries[:,None])
        shortest=shapely.shortest_line(lines,geometries);nearest=shapely.get_point(shortest,0)
        projection=np.zeros(len(r),dtype=np.float64);moving=shapely.length(lines)>0
        projection[moving]=shapely.line_locate_point(lines[moving],nearest[moving])
        lengths=np.linalg.norm(np.diff(points[r],axis=1),axis=-1);ends=np.cumsum(lengths,axis=1)
        j=np.sum(ends<projection[:,None],axis=1).clip(0,10)
        before=np.where(j>0,ends[np.arange(len(j)),np.maximum(j-1,0)],0.)
        part=np.divide(projection-before,lengths[np.arange(len(j)),j],out=np.zeros(len(j)),where=lengths[np.arange(len(j)),j]>1e-12).clip(0,1)
        closest_time=(j+part+1.)/12.
        node[r,s,:6]=np.eye(6,dtype=np.float32)[types]
        node[r,s,6:15]=np.stack([reg['entities'][i]['semantic'] for i in e])
        edge[r,s,:6]=np.stack((distance/10.,point_distance.mean(-1)/10.,point_distance[:,-1]/10.,closest_time,
            (point_distance<=2.).mean(-1),(point_distance<=4.).mean(-1)),axis=-1)
        lane=types<2
        if lane.any():
            lane_projection=shapely.line_locate_point(geometries[lane],shapely.get_point(shortest[lane],-1))
            tangent=np.empty((lane.sum(),2),dtype=np.float64)
            for i,(li,position) in enumerate(zip(e[lane],lane_projection)):
                xy=reg['entities'][li]['coordinates'];delta=np.diff(xy,axis=0);norm=np.linalg.norm(delta,axis=-1)
                good=np.flatnonzero(norm>1e-9);assert len(good)
                k=min(int(np.searchsorted(np.cumsum(norm),position,side='right')),len(norm)-1)
                if norm[k]<=1e-9:k=int(good[np.argmin(np.abs(good-k))])
                tangent[i]=delta[k]/norm[k]
            local=tangent@rotation_matrix(yaw);theta=np.arctan2(local[:,1],local[:,0]);diff=np.asarray(endpoint_heading)[r[lane]]-theta
            node[r[lane],s[lane],15:18]=np.stack((np.sin(theta),np.cos(theta),np.ones(len(theta))),axis=-1)
            edge[r[lane],s[lane],6:9]=np.stack((np.sin(diff),np.cos(diff),np.ones(len(theta))),axis=-1)
        if (~lane).any():
            # Raw multipart records can contain touching/overlapping valid polygons.
            # Evaluate their original components independently, avoiding Collection
            # topology operations or repairs. Entity containment/intersection is OR.
            part_geometries=[];part_rows=[]
            for k,li in enumerate(e[~lane]):
                parts=reg['entities'][li]['parts'];part_geometries.extend(parts);part_rows.extend([k]*len(parts))
            part_geometries=np.asarray(part_geometries,dtype=object);part_rows=np.asarray(part_rows,dtype=np.int32)
            inside_points=np.zeros(((~lane).sum(),12),dtype=bool);intersects=np.zeros((~lane).sum(),dtype=bool)
            contains=shapely.contains(part_geometries[:,None],shapely.points(points[r[~lane]])[part_rows])
            np.logical_or.at(inside_points,part_rows,contains)
            np.logical_or.at(intersects,part_rows,shapely.intersects(part_geometries,lines[~lane][part_rows]))
            inside=inside_points.mean(-1)
            edge[r[~lane],s[~lane],9]=inside;edge[r[~lane],s[~lane],10]=intersects
            assert not node[r[~lane],s[~lane],6:18].any() and not edge[r[~lane],s[~lane],6:9].any()
        assert np.isfinite(node).all() and np.isfinite(edge).all()
        assert not node[~mask].any() and not edge[~mask].any()
        return {'map_node':node,'map_edge':edge}
