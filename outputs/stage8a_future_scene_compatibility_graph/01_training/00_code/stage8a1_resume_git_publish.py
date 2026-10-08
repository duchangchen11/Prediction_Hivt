"""Publish an ordered native commit chain; preserve every historical parent."""
from pathlib import Path
import subprocess,json,base64,argparse
PROJECT=Path(__file__).resolve().parents[4];ROOT=Path(__file__).resolve().parents[2]
BASE='2a79b0d687192002537073147d93658bf4c6423e'
def git(*a):return subprocess.check_output(['git',*a],cwd=PROJECT,text=True).strip()
def main():
    opt=argparse.ArgumentParser();opt.add_argument('--base',default=BASE);args=opt.parse_args()
    commits=git('rev-list','--reverse',args.base+'..HEAD').splitlines();assert commits
    prefix=str(ROOT.relative_to(PROJECT))+'/';payload=[]
    for commit in commits:
        parent=git('rev-parse',commit+'^');files=[]
        for name in git('diff','--name-only',parent,commit).splitlines():
            assert name.startswith(prefix),name
            assert not name.endswith(('.pt','.npy','.npz','.log')) and '/cache/' not in name,name
            assert not name.startswith(prefix+'00c_') and not name.startswith(prefix+'00b_')
            data=subprocess.check_output(['git','show',commit+':'+name],cwd=PROJECT);assert len(data)<10_000_000
            files.append({'path':name,'sha':git('rev-parse',commit+':'+name),'content':base64.b64encode(data).decode()})
        payload.append({'parent':parent,'local_commit':commit,'base_tree':git('rev-parse',parent+'^{tree}'),
            'tree':git('rev-parse',commit+'^{tree}'),'message':git('log','-1','--format=%B',commit),'files':files})
    out={'base':args.base,'commits':payload};p=Path(__file__).parent/'stage8a1_resume_git_upload_payload.json'
    p.write_text(json.dumps(out,separators=(',',':')));print(json.dumps({'commits':len(payload),'files':sum(len(p['files']) for p in payload),'bytes':p.stat().st_size}))
if __name__=='__main__':main()
