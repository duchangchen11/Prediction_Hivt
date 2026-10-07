"""Prepare a scoped commit for equivalent GitHub Git-object publication."""
from pathlib import Path
import subprocess,json,base64
PROJECT=Path(__file__).resolve().parents[3]
ROOT=PROJECT/'outputs/stage8a_future_scene_compatibility_graph'
def git(*a):return subprocess.check_output(['git',*a],cwd=PROJECT,text=True).strip()
def main():
    head=git('rev-parse','HEAD');parent=git('rev-parse','HEAD^')
    names=git('diff','--name-only',parent,head).splitlines()
    assert names and all(n.startswith('outputs/stage8a_future_scene_compatibility_graph/') for n in names)
    files=[]
    for name in names:
        assert not name.endswith(('.pt','.npz','.log'))
        data=(PROJECT/name).read_bytes();assert len(data)<10_000_000
        files.append({'path':name,'sha':git('rev-parse',head+':'+name),'content':base64.b64encode(data).decode()})
    payload={'parent':parent,'local_commit':head,'base_tree':git('rev-parse',parent+'^{tree}'),
        'tree':git('rev-parse',head+'^{tree}'),'message':git('log','-1','--format=%B'),'files':files}
    p=ROOT/'00_manifest/stage8a_git_upload_payload.json';p.write_text(json.dumps(payload,separators=(',',':')))
    print(json.dumps({'files':len(files),'payload_bytes':p.stat().st_size,'parent':parent,'tree':payload['tree']}))
if __name__=='__main__':main()
