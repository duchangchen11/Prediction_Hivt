"""Aggregate only frozen quota entities; no future ground truth API."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'00_manifest'))
from stage12a_common import *
def semantic_features(index,points,location,selection,edges):
    """points [P,12,2] global; original entity IDs, distances and FP32 edges."""
    p=len(points);ids=selection['entity_ids'];types=selection['entity_types'];mask=selection['map_mask']
    assert ids.shape==types.shape==mask.shape==(p,8)
    values=np.zeros((p,len(SEM_FIELDS)),np.float64);valid=np.zeros_like(values,dtype=bool)
    col={s:i for i,s in enumerate(SEM_FIELDS)};reg=index.regions[location]
    def put(name,a,m):values[:,col[name]]=np.where(m,a,0);valid[:,col[name]]=m
    center=mask&(types<2);anycenter=center.any(-1)
    distances=np.where(center,selection['geometry_distances'],np.inf)
    slot=distances.argmin(-1);ii=np.arange(p)
    for name,field in [('centerline_distance',0),('centerline_mean_distance',1),('centerline_endpoint_distance',2)]:
        put(name,edges[ii,slot,field].astype(np.float64)*10,anycenter)
    heading_defined=np.linalg.norm(points[:,-1]-points[:,-3],axis=-1)>1e-6
    put('centerline_heading_error',np.arccos(np.clip(edges[ii,slot,7].astype(np.float64),-1,1)),anycenter&heading_defined)
    put('centerline_valid',anycenter,np.ones(p,bool));put('lane_count',(mask&(types==0)).sum(-1),np.ones(p,bool))
    put('connector_count',(mask&(types==1)).sum(-1),np.ones(p,bool))
    lines=shapely.linestrings(points);pointgeom=shapely.points(points)
    for label,ty in [('drivable',2),('carpark',3),('crosswalk',4),('walkway',5)]:
        m=mask&(types==ty);has=m.any(-1);nearest=np.where(m,selection['geometry_distances'],np.inf).min(-1)
        put(label+'_distance',nearest,has);put(label+'_valid',has,np.ones(p,bool))
        rows,slots=np.where(m);parts=[];pr=[]
        for row,sl in zip(rows,slots):
            ent=reg['entities'][reg['id_to_local'][int(ids[row,sl])]]
            parts.extend(ent['parts']);pr.extend([row]*len(ent['parts']))
        inside=np.zeros((p,12),bool);intersects=np.zeros(p,bool);crosses=np.zeros(p,bool);bd=np.full(p,np.inf)
        if parts:
            geom=np.asarray(parts,dtype=object);pr=np.asarray(pr,np.int64)
            assert shapely.is_valid(geom).all()
            np.logical_or.at(inside,pr,shapely.contains(geom[:,None],pointgeom[pr]))
            np.logical_or.at(intersects,pr,shapely.intersects(lines[pr],geom))
            if label=='drivable':
                boundary=shapely.boundary(geom)
                np.logical_or.at(crosses,pr,shapely.crosses(lines[pr],boundary))
                np.minimum.at(bd,pr,shapely.distance(lines[pr],boundary))
        put(label+'_inside_fraction',inside.mean(-1),has);put(label+'_intersects',intersects,has)
        if label=='drivable':put('drivable_boundary_crossing',crosses,has);put('drivable_boundary_distance',bd,has)
    put('map_valid_mask',mask.any(-1),np.ones(p,bool))
    assert np.isfinite(values).all() and not values[~valid].any()
    return values,valid
def from_observed_window(index,cache,origin,yaw,location,nodes):
    w=observable_window(cache,location,origin,yaw)
    local=w.predicted[nodes].numpy().reshape(-1,12,2);points=ego_to_global(local,origin,yaw)
    types=np.repeat(w.actor_type[nodes].numpy(),6);selection=index.select_batch(types,points,location)
    secant=local[:,-1]-local[:,-3];heading=np.arctan2(secant[:,1],secant[:,0])
    feature=index.features(points,location,yaw,heading,selection)
    sem,valid=semantic_features(index,points,location,selection,feature['map_edge'])
    return points,selection,feature,sem,valid
def geometric_features(candidates,current,history_heading,heading_valid,rows,node,edge,neighbor):
    """Observable motion and existing original G1 edge summaries, no labels."""
    n=len(candidates);out=np.zeros((n,6,len(GEO_FIELDS)),np.float64);col={k:i for i,k in enumerate(GEO_FIELDS)}
    def put(k,x):out[...,col[k]]=x
    steps=np.diff(np.concatenate((current[:,None,None,:].repeat(6,axis=1),candidates),axis=2),axis=2)
    lengths=np.linalg.norm(steps,axis=-1);secant=candidates[:,:,-1]-candidates[:,:,-3]
    angles=np.arctan2(steps[...,1],steps[...,0]);good=(lengths[...,1:]>1e-6)&(lengths[...,:-1]>1e-6)
    turn=np.abs(wrap_angle(np.diff(angles,axis=-1)));curvature=np.where(good,turn,0).sum(-1)
    put('predicted_displacement',node[:,0,:,10]);put('endpoint_x',candidates[:,:,-1,0]);put('endpoint_y',candidates[:,:,-1,1])
    put('predicted_heading',np.arctan2(secant[...,1],secant[...,0]));put('predicted_heading_valid',np.linalg.norm(secant,axis=-1)>1e-6)
    put('trajectory_length',node[:,0,:,11]);put('trajectory_curvature',curvature)
    for k,f in [('recent_speed','recent_speed'),('history_displacement','history_net'),('history_path_length','history_path'),('history_valid_count','history_valid_count')]:put(k,rows[f].to_numpy()[:,None])
    put('history_heading',history_heading[:,None]);put('history_heading_valid',heading_valid[:,None]);put('current_x',current[:,0,None]);put('current_y',current[:,1,None])
    put('original_probability',node[:,0,:,4]);put('original_logit',node[:,0,:,3]);put('neighbor_count',neighbor.sum(-1)[:,None]);put('interaction_valid',neighbor.any(-1)[:,None])
    mask=neighbor[:,None,:,None];denom=np.maximum(1,neighbor.sum(-1)*6)[:,None]
    for label,k,reduce in [('interaction_minimum_distance',0,'min'),('interaction_mean_distance',2,'mean'),
      ('interaction_endpoint_distance',3,'mean'),('interaction_closing_mean',8,'mean'),('interaction_relative_step_minimum',7,'min')]:
        a=edge[...,k].astype(np.float64)
        v=np.where(mask,a,np.inf).min(axis=(2,3)) if reduce=='min' else np.where(mask,a,0).sum(axis=(2,3))/denom
        v[~neighbor.any(-1)]=0;put(label,v)
    assert np.isfinite(out).all()
    return out
