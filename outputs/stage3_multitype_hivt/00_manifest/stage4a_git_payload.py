"""Exact Stage4A-only Git tree payload; guards the frozen branch and large data."""
import argparse
import base64
import json
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[1];PROJECT=ROOT.parents[1]

def git(*args):return subprocess.check_output(['git',*args],cwd=PROJECT)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--base',required=True);parser.add_argument('--label',required=True)
    args=parser.parse_args();assert args.label.replace('_','').isalnum()
    assert git('branch','--show-current').decode().strip()=='stage4a/type-conditioned-interaction'
    parent=args.base;commits=[]
    for commit in git('rev-list','--reverse',args.base+'..HEAD').decode().splitlines():
        entries=[]
        for name in git('diff','--name-only',parent,commit).decode().splitlines():
            path=Path(name)
            assert name.startswith(str(ROOT.relative_to(PROJECT))+'/') and path.name.startswith('stage4a_'),name
            assert 'stage3_cache' not in path.parts and '__pycache__' not in path.parts,name
            assert path.suffix not in ('.pt','.pth','.ckpt','.pyc'),name
            assert path.name!='stage4a_actor_errors.csv'
            raw=git('show',commit+':'+name);sha=git('rev-parse',commit+':'+name).decode().strip()
            assert len(raw)<=5*1024*1024,'Large artifact should stay local: '+name
            try:content=raw.decode('utf-8');encoding='utf-8'
            except UnicodeDecodeError:content=base64.b64encode(raw).decode();encoding='base64'
            entries.append({'path':name,'sha':sha,'mode':'100644','type':'blob','content':content,'encoding':encoding})
        commits.append({'local_commit':commit,'parent_local':parent,
            'base_tree':git('rev-parse',parent+'^{tree}').decode().strip(),
            'expected_tree':git('rev-parse',commit+'^{tree}').decode().strip(),
            'message':git('log','-1','--format=%B',commit).decode().strip(),'entries':entries})
        parent=commit
    payload={'repository':'duchangchen11/Prediction_Hivt','base_commit':args.base,
        'branch':'stage4a/type-conditioned-interaction','commits':commits}
    path=ROOT/'02_preprocessed/stage3_cache'/('stage4a_git_payload_'+args.label+'.json')
    path.write_text(json.dumps(payload,ensure_ascii=True,separators=(',',':')))
    print(json.dumps({'path':str(path),'bytes':path.stat().st_size,'commits':len(commits),'entries':[len(c['entries']) for c in commits]}))

if __name__=='__main__':main()
