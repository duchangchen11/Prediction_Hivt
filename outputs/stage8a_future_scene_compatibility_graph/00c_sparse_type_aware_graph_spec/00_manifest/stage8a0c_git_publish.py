"""Prepare publication restricted to the new Stage8A-0C subtree."""
from pathlib import Path
import subprocess,json,base64
PROJECT=Path(__file__).resolve().parents[4];ROOT=Path(__file__).resolve().parents[1]
def git(*args):return subprocess.check_output(['git',*args],cwd=PROJECT,text=True).strip()
def main():
    head=git('rev-parse','HEAD');parent=git('rev-parse','HEAD^')
    assert parent=='dbb743ab79d3d0309419cd051624ff99c7a6a0ee'
    names=git('diff','--name-only',parent,head).splitlines();prefix=str(ROOT.relative_to(PROJECT))+'/'
    assert names and all(n.startswith(prefix) for n in names)
    files=[]
    for name in names:
        assert not name.endswith(('.pt','.npz','.log'))
        data=(PROJECT/name).read_bytes();assert len(data)<10_000_000
        files.append({'path':name,'sha':git('rev-parse',head+':'+name),'content':base64.b64encode(data).decode()})
    payload={'parent':parent,'local_commit':head,'base_tree':git('rev-parse',parent+'^{tree}'),
        'tree':git('rev-parse',head+'^{tree}'),'message':git('log','-1','--format=%B'),'files':files}
    p=ROOT/'00_manifest/stage8a0c_git_upload_payload.json';p.write_text(json.dumps(payload,separators=(',',':')))
    print(json.dumps({'files':len(files),'payload_bytes':p.stat().st_size,'parent':parent,'tree':payload['tree']}))
if __name__=='__main__':main()
