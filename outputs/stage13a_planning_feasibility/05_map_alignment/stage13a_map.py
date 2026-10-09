"""Existing HD Map geometry transformed with the existing t0 ego convention."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'00_manifest'))
from stage13a_common import *
import shapely
from stage8a0c_selector import SparseSemanticIndex
class MapInterface:
    def __init__(self):self.index=SparseSemanticIndex();self.roundtrip=0.
    def __call__(self,obs):
        radius=150.;localcorners=np.array([[-radius,-radius],[-radius,radius],[radius,-radius],[radius,radius]])
        globalcorners=ego_to_global(localcorners,obs['origin'],obs['yaw']);bounds=shapely.box(*np.r_[globalcorners.min(0),globalcorners.max(0)])
        region=self.index.regions[obs['location']];parts={2:[],4:[],5:[]};lanes=[];tokens=[]
        for i in region['polygon_tree'].query(bounds):
            entity=region['entities'][int(region['part_entities'][i])];ty=entity['type']
            if ty not in parts:continue
            source=region['polygon_parts'][i];local=shapely.transform(source,lambda xy:global_to_ego(xy,obs['origin'],obs['yaw']));parts[ty].append(local)
            xy=shapely.get_coordinates(source);self.roundtrip=max(self.roundtrip,float(np.abs(ego_to_global(global_to_ego(xy,obs['origin'],obs['yaw']),obs['origin'],obs['yaw'])-xy).max()))
        for i in region['lane_tree'].query(bounds):
            entity=region['entities'][int(region['lane_indices'][i])];lanes.append(global_to_ego(entity['coordinates'],obs['origin'],obs['yaw']));tokens.append(entity['token'])
        union=shapely.union_all(parts[2]) if parts[2] else shapely.GeometryCollection();coords=np.arange(-150,151,dtype=np.float64);xx,yy=np.meshgrid(coords,coords,indexing='xy')
        # Predicate acceleration only: prepare builds an index on the unchanged
        # polygon union. It does not buffer, simplify, repair or resample it.
        shapely.prepare(union)
        raster=shapely.covers(union,shapely.points(xx,yy));assert raster.shape==(301,301) and self.roundtrip<1e-5
        return dict(MapDrivableMask=raster,MapRasterX=coords,MapRasterY=coords,MapBoundaryGeometry=union.boundary,MapLaneGeometry=lanes,
            lane_tokens=tokens,drivable=union,drivable_parts=parts[2],crossing_parts=parts[4],walkway_parts=parts[5],
            patch_bounds=np.array([-150.,150.,-150.,150.]),available=not union.is_empty,location=obs['location'],coordinate_frame='t0 ego planar: +x forward, +y left')
def save_map(path,geometry):
    atomic_json(path,dict(Location=geometry['location'],CoordinateFrame=geometry['coordinate_frame'],DrivableWKB=shapely.to_wkb(geometry['drivable']).hex(),
        WalkwayWKB=[shapely.to_wkb(p).hex() for p in geometry['walkway_parts']],CrossingWKB=[shapely.to_wkb(p).hex() for p in geometry['crossing_parts']],
        LaneGeometry=[p.tolist() for p in geometry['MapLaneGeometry']],LaneTokens=geometry['lane_tokens'],PatchBounds=geometry['patch_bounds'].tolist()))
    atomic_npz(Path(path).with_suffix('.npz'),MapDrivableMask=geometry['MapDrivableMask'],MapRasterX=geometry['MapRasterX'],MapRasterY=geometry['MapRasterY'])
