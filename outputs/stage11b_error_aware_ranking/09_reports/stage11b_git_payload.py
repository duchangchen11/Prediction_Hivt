"""Describe exact committed Git objects for connector publication; no credentials."""
from pathlib import Path
import subprocess,json
ROOT=Path(__file__).resolve().parents[1];PROJECT=ROOT.parents[1]
BASE='5617f9463b3f5baa4d941f52c157c25a341fd5e1'
def git(*args):return subprocess.check_output(['git',*args],cwd=PROJECT).decode().strip()
def main():
    commits=git('rev-list','--reverse',f'{BASE}..HEAD').splitlines();rows=[];blobs={}
    for commit in commits:
        parents=git('show','-s','--format=%P',commit).split();assert len(parents)==1
        parent=parents[0];entries=[]
        paths=git('diff-tree','--no-commit-id','--name-only','-r',commit).splitlines()
        for path in paths:
            assert path.startswith('outputs/stage11b_error_aware_ranking/'),path
            entry=git('ls-tree',commit,'--',path);assert entry
            mode,kind,sha=entry.split('\t')[0].split();assert mode=='100644' and kind=='blob'
            size=int(git('cat-file','-s',sha));assert size<1500000 and not path.endswith(('.pt','.npy','.npz','.log'))
            entries.append({'path':path,'mode':mode,'type':kind,'sha':sha})
            blobs[sha]={'sha':sha,'size':size}
        rows.append({'native_sha':commit,'native_parent':parent,'tree':git('show','-s','--format=%T',commit),
            'message':git('show','-s','--format=%B',commit),'entries':entries})
    payload={'base_commit':BASE,'base_tree':git('show','-s','--format=%T',BASE),'native_head':git('rev-parse','HEAD'),
        'branch':'stage11b/controlled-ranking-objectives','repository':'duchangchen11/Prediction_Hivt','commits':rows,'blobs':list(blobs.values())}
    (ROOT/'09_reports/stage11b_git_upload_payload.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'commits':len(rows),'distinct_blobs':len(blobs),'bytes':sum(b['size'] for b in blobs.values()),'native_head':payload['native_head']}))
if __name__=='__main__':main()
