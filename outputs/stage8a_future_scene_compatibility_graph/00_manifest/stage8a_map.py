"""Exact future-polyline retrieval against frozen whole-region centerlines."""
from stage8a_common import *
from preprocessing.coordinates import ego_to_global,rotation_matrix
import shapely
from shapely.strtree import STRtree
sys.path.insert(0,str(STAGE7A/'00_manifest'))
from stage7a_dataset import semantic_vector,SEMANTIC_FIELDS

MAP_NODE_FIELDS=SEMANTIC_FIELDS+('local_tangent_sin','local_tangent_cos','is_lane','is_connector')
MAP_EDGE_FIELDS=('minimum_polyline_distance','mean_future_point_distance','endpoint_distance',
    'heading_difference_sin','heading_difference_cos','fraction_points_within2m','fraction_points_within4m',
    'closest_timestep_over_Tf')+SEMANTIC_FIELDS

class MapIndex:
    def __init__(self):
        self.metadata=read_json(SEMANTICS);self.regions={};self.token_dictionary=[]
        with np.load(CENTERLINES) as archive:
            for location,rows in sorted(self.metadata.items()):
                tokens=sorted(rows);lines=[archive[location+'__'+t].copy() for t in tokens]
                assert all(x.ndim==2 and x.shape[1]==2 and len(x)>=2 and np.isfinite(x).all() for x in lines)
                geometries=np.asarray([shapely.LineString(x) for x in lines],dtype=object)
                assert (shapely.length(geometries)>0).all()
                semantics=np.stack([semantic_vector(rows[t]) for t in tokens])
                assert semantics.shape==(len(tokens),9) and np.isin(semantics,[0,1]).all()
                ids=np.arange(len(tokens),dtype=np.int32)+len(self.token_dictionary)
                self.token_dictionary.extend({'index':int(i),'location':location,'token':t,'layer':rows[t]['layer'],
                    'semantic':s.tolist()} for i,t,s in zip(ids,tokens,semantics))
                self.regions[location]={'tokens':tokens,'coordinates':lines,'geometries':geometries,
                    'semantic':semantics,'tree':STRtree(geometries),'ids':ids}
        assert len(self.token_dictionary)==8251

    def retrieve(self,points,location):
        """points [P,12,2] global; token indices unique, exact distance then token tie-break."""
        region=self.regions[location];geometries=region['geometries'];tree=region['tree']
        points=np.asarray(points,dtype=np.float64);lines=shapely.linestrings(points);p=len(points)
        pairs=tree.query(lines,predicate='dwithin',distance=10.)
        counts=np.bincount(pairs[0],minlength=p)
        token=np.full((p,8),-1,dtype=np.int32);distance=np.zeros((p,8),dtype=np.float64)
        if pairs.size:
            d=shapely.distance(lines[pairs[0]],geometries[pairs[1]])
            assert np.isfinite(d).all() and (d<=10.+1e-9).all()
            order=np.lexsort((pairs[1],d,pairs[0]));r=pairs[0,order];l=pairs[1,order];d=d[order]
            offsets=np.repeat(np.cumsum(np.r_[0,counts[:-1]]),counts)
            rank=np.arange(len(r))-offsets;keep=rank<8
            token[r[keep],rank[keep]]=l[keep];distance[r[keep],rank[keep]]=d[keep]
        missing=np.flatnonzero(counts==0)
        if len(missing):
            pair,d=tree.query_nearest(lines[missing],all_matches=True,return_distance=True)
            order=np.lexsort((pair[1],d,pair[0]));r=pair[0,order];l=pair[1,order];d=d[order]
            first=np.r_[True,r[1:]!=r[:-1]]
            token[missing[r[first]],0]=l[first];distance[missing[r[first]],0]=d[first]
            assert len(np.unique(r))==len(missing) and (d>10.-1e-9).all()
        mask=token>=0
        assert mask.any(-1).all() and (mask.sum(-1)<=8).all()
        return {'region_token':token,'minimum_distance':distance,'mask':mask,'fallback':counts==0,
                'within_radius_token_count':counts,'candidate_lines':lines}

    def features(self,points,heading,location,yaw,retrieval):
        """Map direction at true closest centerline point; distance uses whole polyline."""
        region=self.regions[location];valid=retrieval['mask'];r,c=np.where(valid);token=retrieval['region_token'][r,c]
        candidate=retrieval['candidate_lines'][r];lane=region['geometries'][token]
        nearest=shapely.shortest_line(candidate,lane)
        candidate_point=shapely.get_point(nearest,0);lane_point=shapely.get_point(nearest,-1)
        lane_projection=shapely.line_locate_point(lane,lane_point)
        # Different tokens have different vertex counts; direction is candidate-specific.
        tangent=np.empty((len(r),2),dtype=np.float64)
        for i,t in enumerate(token):
            xy=region['coordinates'][t];d=np.diff(xy,axis=0);length=np.linalg.norm(d,axis=-1)
            good=np.flatnonzero(length>1e-9);assert len(good)
            ends=np.cumsum(length);j=min(int(np.searchsorted(ends,lane_projection[i],side='right')),len(length)-1)
            if length[j]<=1e-9:j=int(good[np.argmin(np.abs(good-j))])
            tangent[i]=d[j]/length[j]
        local_tangent=tangent@rotation_matrix(yaw)
        lane_heading=np.arctan2(local_tangent[:,1],local_tangent[:,0]);difference=heading[r]-lane_heading
        # Vectorized twelve-point distances retain full-centerline segment interiors.
        point_distance=shapely.distance(shapely.points(points[r]),lane[:,None])
        # A stationary predicted polyline has no arc-length coordinate; its earliest
        # future point is the fixed closest-time convention. Avoid GEOS 0/0 warnings.
        projection=np.zeros(len(candidate),dtype=np.float64)
        moving=shapely.length(candidate)>0
        projection[moving]=shapely.line_locate_point(candidate[moving],candidate_point[moving])
        lengths=np.linalg.norm(np.diff(points[r],axis=1),axis=-1);ends=np.cumsum(lengths,axis=1)
        j=np.sum(ends<projection[:,None],axis=1).clip(0,10)
        before=np.where(j>0,ends[np.arange(len(j)),np.maximum(j-1,0)],0.)
        part=np.divide(projection-before,lengths[np.arange(len(j)),j],out=np.zeros(len(j)),where=lengths[np.arange(len(j)),j]>1e-12).clip(0,1)
        timestep=(j+part+1.)/12.
        semantic=region['semantic'][token];is_connector=semantic[:,0]
        node=np.zeros((len(points),8,13),dtype=np.float32);edge=np.zeros((len(points),8,17),dtype=np.float32)
        node[r,c,:9]=semantic;node[r,c,9]=np.sin(lane_heading);node[r,c,10]=np.cos(lane_heading)
        node[r,c,11]=1-is_connector;node[r,c,12]=is_connector
        edge[r,c,:8]=np.stack((retrieval['minimum_distance'][r,c],point_distance.mean(-1),point_distance[:,-1],
            np.sin(difference),np.cos(difference),(point_distance<=2.).mean(-1),(point_distance<=4.).mean(-1),timestep),axis=-1)
        edge[r,c,8:]=semantic
        global_token=np.full_like(retrieval['region_token'],-1);global_token[r,c]=region['ids'][token]
        assert np.isfinite(node).all() and np.isfinite(edge).all()
        assert np.allclose(shapely.length(nearest),retrieval['minimum_distance'][r,c],atol=1e-8)
        return {'map_node':node,'map_edge':edge,'map_token_index':global_token,'map_mask':valid,
                'map_fallback':retrieval['fallback'],'within_radius_token_count':retrieval['within_radius_token_count']}

    def build(self,w):
        selected=torch.where(~w.history_padding[:,4])[0].numpy()
        local=w.predicted[selected].numpy().reshape(-1,12,2)
        points=ego_to_global(local,w.origin,w.yaw)
        d=local[:,-1]-local[:,-3];heading=np.arctan2(d[:,1],d[:,0])
        result=self.retrieve(points,w.map_location);features=self.features(points,heading,w.map_location,w.yaw,result)
        return selected,{key:value.reshape((len(selected),6)+value.shape[1:]) for key,value in features.items()}
