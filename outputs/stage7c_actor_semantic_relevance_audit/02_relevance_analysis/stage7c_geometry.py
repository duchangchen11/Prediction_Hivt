"""Offline geometry only: true polyline/segment distances and fixed turn labels."""
import numpy as np
import shapely
from shapely.geometry import LineString,Point
from shapely.strtree import STRtree
from preprocessing.coordinates import ego_to_global,wrap_angle

def segments(positions,vectors):
    coordinates=np.stack([positions,positions+vectors],axis=1).astype(np.float64)
    return shapely.linestrings(coordinates)

def future_distances(future,lane_segments):
    # The 12 observed GT future points form a polyline. Distance includes segment
    # interiors, crossings and degenerate stationary polylines, never just starts.
    line=LineString(np.asarray(future,dtype=np.float64))
    return np.asarray(shapely.distance(line,lane_segments),dtype=np.float64)

def point_segment_distances(point,start,vector):
    p=np.asarray(point,dtype=np.float64);a=start.astype(np.float64);v=vector.astype(np.float64)
    den=np.sum(v*v,axis=-1);num=np.sum((p-a)*v,axis=-1)
    t=np.clip(np.divide(num,den,out=np.zeros_like(num),where=den>0),0,1)
    return np.linalg.norm(p-a-t[:,None]*v,axis=-1)

def turn_membership(trajectory,vehicle):
    t=np.asarray(trajectory,dtype=np.float64)
    vin=t[2]-t[0];vout=t[-1]-t[-3]
    eligible=vehicle and np.linalg.norm(t[-1]-t[0])>5 and np.linalg.norm(vin)>.5 and np.linalg.norm(vout)>.5
    delta=float(np.degrees(wrap_angle(np.arctan2(vout[1],vout[0])-np.arctan2(vin[1],vin[0])))) if eligible else 0.
    return int(eligible and abs(delta)>20),delta

class TurnAmbiguity:
    def __init__(self,metadata,centerlines):
        self.trees={}
        for location,rows in metadata.items():
            for turn in ('left','straight','right'):
                tokens=[t for t,v in rows.items() if v['is_connector'] and v['turn_type']==turn]
                self.trees[location,turn]=STRtree([LineString(centerlines[location+'__'+t]) for t in tokens])

    def graph(self,g):
        nodes=np.flatnonzero(g.full_horizon_mask.numpy()&(g.agent_type.numpy()==0))
        result=np.zeros((g.num_nodes,4),dtype=np.int32)
        if not len(nodes):return result
        xy=ego_to_global(g.positions[nodes,4].numpy(),g.origin.numpy(),float(g.ego_yaw))
        points=shapely.points(xy)
        for c,turn in enumerate(('left','straight','right')):
            tree=self.trees[g.map_location,turn]
            pairs=tree.query(points,predicate='dwithin',distance=20.)
            if pairs.size:
                distance=shapely.distance(points[pairs[0]],tree.geometries[pairs[1]])
                # Each tree row is one unique map token, so no segment duplication.
                count=np.bincount(pairs[0][distance<20.],minlength=len(nodes))
                result[nodes,c]=count
        result[:,3]=np.sum(result[:,:3]>0,axis=1)
        return result

def numerical_distance_checks():
    sg=segments(np.array([[0.,0.],[0.,0.],[2.,0.]]),np.array([[10.,0.],[0.,0.],[0.,2.]]))
    d=future_distances(np.array([[5.,-1.],[5.,1.]]),sg)
    assert np.allclose(d,[0.,5.,3.]),d
    stationary=future_distances(np.array([[3.,4.],[3.,4.]]),sg)
    assert np.allclose(stationary,[4.,5.,np.sqrt(5.)]),stationary
    assert np.allclose(point_segment_distances(np.array([[5.,2.]]),np.array([[0.,0.]]),np.array([[10.,0.]])),[2.])
    return {'interior_crossing':True,'degenerate_lane':True,'stationary_GT':True,'point_segment_interior':True}
