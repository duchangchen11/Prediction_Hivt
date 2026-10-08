"""Prepare ordered native commit metadata for authenticated Git data publication."""
from pathlib import Path
import subprocess,json
ROOT=Path(__file__).resolve().parents[1]; PROJECT=ROOT.parents[1]
BASE='b4dd350e3f9488e284943e0eed9c93fd87adb8d8'
def git(*args): return subprocess.check_output(['git',*args],cwd=PROJECT,text=True).strip()
def main():
    commits=git('rev-list','--reverse',BASE+'..HEAD').splitlines(); prefix=str(ROOT.relative_to(PROJECT))+'/'; payload=[]
    for commit in commits:
        parent=git('rev-parse',commit+'^'); files=[]
        for line in git('diff','--name-status',parent,commit).splitlines():
            status,path=line.split('\t'); assert status=='A' and path.startswith(prefix),(status,path)
            assert '/cache/' not in path and not path.endswith(('.pt','.npy','.npz','.log')) and not path.endswith('stage10a_actor_results.csv'),path
            data=subprocess.check_output(['git','show',commit+':'+path],cwd=PROJECT); assert len(data)<10_000_000
            files.append({'path':path,'sha':git('rev-parse',commit+':'+path),'bytes':len(data)})
        payload.append({'parent':parent,'local_commit':commit,'base_tree':git('rev-parse',parent+'^{tree}'),
            'tree':git('rev-parse',commit+'^{tree}'),'message':git('log','-1','--format=%B',commit),'files':files})
    out=ROOT/'00_manifest/stage10a_git_upload_payload.json'
    out.write_text(json.dumps({'base':BASE,'commits':payload},separators=(',',':')))
    print(json.dumps({'commits':len(payload),'files':sum(len(c['files']) for c in payload),'metadata_bytes':out.stat().st_size}))
if __name__=='__main__': main()
