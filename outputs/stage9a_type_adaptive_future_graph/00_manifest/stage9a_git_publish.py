"""Ordered commit publication payload; preserve the Stage8 base and ancestry."""
from pathlib import Path
import subprocess,json,base64
ROOT=Path(__file__).resolve().parents[1];PROJECT=ROOT.parents[1]
BASE='31aa16dff59f393d6e2e3337e5861755ab069aec'
def git(*args):return subprocess.check_output(['git',*args],cwd=PROJECT,text=True).strip()
def main():
    commits=git('rev-list','--reverse',BASE+'..HEAD').splitlines();prefix=str(ROOT.relative_to(PROJECT))+'/';payload=[]
    for commit in commits:
        parent=git('rev-parse',commit+'^');files=[]
        for row in git('diff','--name-status',parent,commit).splitlines():
            status,path=row.split('\t');assert status=='A' and path.startswith(prefix),(status,path)
            assert '/cache/' not in path and not path.endswith(('.pt','.npz','.npy','.log')) and not path.endswith('stage9a_actor_results.csv'),path
            data=subprocess.check_output(['git','show',commit+':'+path],cwd=PROJECT);assert len(data)<10_000_000
            files.append({'path':path,'sha':git('rev-parse',commit+':'+path),'content':base64.b64encode(data).decode()})
        payload.append({'parent':parent,'local_commit':commit,'base_tree':git('rev-parse',parent+'^{tree}'),'tree':git('rev-parse',commit+'^{tree}'),'message':git('log','-1','--format=%B',commit),'files':files})
    out=ROOT/'00_manifest/stage9a_git_upload_payload.json';out.write_text(json.dumps({'base':BASE,'commits':payload},separators=(',',':')))
    print(json.dumps({'commits':len(payload),'files':sum(len(c['files']) for c in payload),'bytes':out.stat().st_size}))
if __name__=='__main__':main()
