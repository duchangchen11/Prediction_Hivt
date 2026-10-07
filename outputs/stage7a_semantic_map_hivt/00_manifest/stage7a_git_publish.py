"""Prepare one scoped native commit for equivalent GitHub Git-object publication."""
from pathlib import Path
import json,subprocess,base64,sys
PROJECT=Path(__file__).resolve().parents[3]
ROOT=PROJECT/'outputs/stage7a_semantic_map_hivt'
def git(*args):return subprocess.check_output(['git',*args],cwd=PROJECT,text=True).strip()
def main():
    parent=git('rev-parse','HEAD^');head=git('rev-parse','HEAD')
    names=git('diff','--name-only',parent,head).splitlines()
    assert names and all(n.startswith('outputs/stage7a_semantic_map_hivt/') for n in names)
    assert not any(n.endswith(('.pt','.npz','.log')) or '/stage7a_actor_' in n for n in names)
    files=[]
    for n in names:
        data=(PROJECT/n).read_bytes();assert len(data)<10_000_000
        files.append({'path':n,'sha':git('rev-parse',head+':'+n),'content':base64.b64encode(data).decode()})
    payload={'parent':parent,'local_commit':head,'base_tree':git('rev-parse',parent+'^{tree}'),
             'tree':git('rev-parse',head+'^{tree}'),'message':git('log','-1','--format=%B'),'files':files}
    path=ROOT/'00_manifest/stage7a_git_upload_formal.json';path.write_text(json.dumps(payload,separators=(',',':')))
    print(json.dumps({'files':len(files),'payload_bytes':path.stat().st_size,'parent':parent,'tree':payload['tree']}))
if __name__=='__main__':main()
