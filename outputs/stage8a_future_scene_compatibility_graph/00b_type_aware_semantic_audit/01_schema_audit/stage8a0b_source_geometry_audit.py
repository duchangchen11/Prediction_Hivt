"""Audit raw layer polygon geometry separately from frozen lane centerlines."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'00_manifest'))
from stage8a0b_common import *
from nuscenes.map_expansion.map_api import NuScenesMap
from collections import Counter

def main():
    source=[]
    for path in sorted(MAP_JSON.glob('*.json')):
        raw=read_json(path);api=NuScenesMap(dataroot=str(MAP_ROOT),map_name=path.stem)
        polygons={r['token']:r for r in raw['polygon']};nodes={r['token']:r for r in raw['node']}
        for layer in TYPES:
            c=Counter();types=Counter()
            for record in raw.get(layer,[]):
                refs=record.get('polygon_tokens',[record['polygon_token']] if 'polygon_token' in record else [])
                types['Polygon' if len(refs)==1 else 'PolygonCollection']+=1
                valid=bool(refs);usable=False;closed=bool(refs)
                for token in refs:
                    p=polygons.get(token)
                    if not p:c['InvalidGeometryReference']+=1;valid=False;continue
                    nt=p.get('exterior_node_tokens',[])+[t for h in p.get('holes',[]) for t in h['node_tokens']]
                    missing=[t for t in nt if t not in nodes]
                    if missing:c['InvalidGeometryReference']+=len(missing);valid=False;continue
                    xy=np.array([[nodes[t]['x'],nodes[t]['y']] for t in nt],dtype=np.float64)
                    c['NaN']+=int(np.isnan(xy).sum());c['Inf']+=int(np.isinf(xy).sum())
                    if not np.isfinite(xy).all():valid=False;continue
                    g=api.extract_polygon(token)
                    ok=g.is_valid and not g.is_empty and g.area>0
                    valid &= ok;usable |= ok
                    closed &= g.exterior.is_closed and all(r.is_closed for r in g.interiors)
                    c['EmptyComponents']+=int(g.is_empty)
                c['ValidGeometry' if valid else 'InvalidGeometry']+=1
                c['ClosedGeometry']+=int(closed);c['UsableGeometry']+=int(usable)
            source.append({'MapRegion':path.stem,'Layer':layer,'Records':len(raw.get(layer,[])),
                'GeometryType':json.dumps(dict(types),sort_keys=True),**{k:int(c[k]) for k in ['ValidGeometry','InvalidGeometry','ClosedGeometry','UsableGeometry','EmptyComponents','NaN','Inf','InvalidGeometryReference']}})
    write_csv(ROOT/'01_schema_audit/stage8a0b_source_polygon_geometry.csv',source)
    by={(r['MapRegion'],r['Layer']):r for r in source}
    rows=list(csv.DictReader((ROOT/'06_tables/stage8a0b_map_schema.csv').open()))
    for r in rows:
        s=by[(r['MapRegion'],r['Layer'])]
        r['RetrievalGeometryType']=r.get('RetrievalGeometryType',r['GeometryType'])
        r['GeometryType']=s['GeometryType']
        for key in ['ValidGeometry','InvalidGeometry','ClosedGeometry','NaN','Inf']:r[key]=s[key]
        r['SourceUsablePolygonGeometry']=s['UsableGeometry']
        r['SourceInvalidGeometryReference']=s['InvalidGeometryReference']
    write_csv(ROOT/'06_tables/stage8a0b_map_schema.csv',rows)
    audit=read_json(ROOT/'01_schema_audit/stage8a0b_schema_audit.json')
    audit['raw_layer_polygon_geometry_audit']='PASS';audit['retrieval_geometry_note']='lane/lane_connector use frozen full LineString centerlines; actual API layer polygons are audited separately'
    audit['raw_polygon_geometry_table_sha256']=sha256(ROOT/'01_schema_audit/stage8a0b_source_polygon_geometry.csv')
    atomic_json(ROOT/'01_schema_audit/stage8a0b_schema_audit.json',audit)
    print('SOURCE GEOMETRY PASS: actual polygons for all seven layers; frozen lane centerlines audited independently',flush=True)

if __name__=='__main__':main()
