"""Prepare only committed Stage7A artifacts for the authorized GitHub push."""
from pathlib import Path
import sys,base64
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'00_manifest'))
from stage7a_common import *
def main():
    assert git('branch','--show-current')=='stage7a/semantic-map-enhancement'
    assert git('rev-parse','HEAD^')==BASE
    assert not git('diff','--name-only') and not git('diff','--cached','--name-only')
    names=git('diff','--name-only',BASE,'HEAD').splitlines();assert names and all(n.startswith(str(ROOT.relative_to(PROJECT))+'/') for n in names)
    rows=[]
    for name in names:
        path=PROJECT/name;rel=path.relative_to(ROOT);assert path.is_file() and path.suffix not in ('.pt','.pth','.npz','.log','.tmp')
        assert not path.name.startswith('stage7a_git_upload_')
        assert not (rel.parts[0]=='01_data_audit' and path.name.startswith('stage7a_actor_'))
        assert not (rel.parts[0]=='02_semantic_cache' and path.suffix=='.json')
        mode,kind,digest=git('ls-tree','HEAD','--',name).split('\t')[0].split();assert mode=='100644' and kind=='blob'
        data=path.read_bytes();rows.append({'path':name,'mode':mode,'type':kind,'sha':digest,'content':base64.b64encode(data).decode(),'encoding':'base64','bytes':len(data)})
    payload={'rows':rows,'local_commit':git('rev-parse','HEAD'),'tree':git('rev-parse','HEAD^{tree}'),'parent':BASE,'base_tree':git('rev-parse',BASE+'^{tree}')}
    target=ROOT/'00_manifest/stage7a_git_upload_final.json';atomic_json(target,payload)
    print(json.dumps({'payload':str(target.relative_to(PROJECT)),'chars':target.stat().st_size,'files':len(rows),'bytes_to_upload':sum(r['bytes'] for r in rows),'tree':payload['tree'],'parent':BASE}))
if __name__=='__main__':main()
