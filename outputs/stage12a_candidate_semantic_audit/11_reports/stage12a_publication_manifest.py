"""Create a local Git-object publication manifest with a strict stage scope."""
from pathlib import Path
import subprocess,json
PROJECT=Path('/home/lrj/Prediction_Hivt');ROOT=PROJECT/'outputs/stage12a_candidate_semantic_audit'
BASE='165c497961afe9f3a196a0987128f9eed48da144';PREFIX=str(ROOT.relative_to(PROJECT))+'/'
def git(*args):return subprocess.check_output(['git',*args],cwd=PROJECT,text=True).strip()
def main():
    commits=git('rev-list','--reverse',BASE+'..HEAD').splitlines();rows=[];blobs={}
    assert git('branch','--show-current')=='stage12a/candidate-semantic-consistency-audit'
    assert not git('diff','--name-only') and not git('diff','--cached','--name-only')
    for commit in commits:
        changes=[]
        for line in git('diff-tree','--no-commit-id','--name-status','-r',commit).splitlines():
            status,path=line.split('\t');assert status in ['A','M'] and path.startswith(PREFIX),line
            assert '/cache/' not in path and Path(path).suffix not in ['.npy','.npz','.pt','.pth','.log'],path
            fields=git('ls-tree',commit,'--',path).split();mode,kind,sha=fields[:3];assert mode=='100644' and kind=='blob'
            size=int(git('cat-file','-s',sha));assert size<10*1024*1024,(path,size)
            blobs[sha]={'SHA':sha,'Size':size,'ExamplePath':path}
            changes.append({'path':path,'mode':mode,'type':kind,'sha':sha})
        rows.append({'NativeCommit':commit,'Tree':git('rev-parse',commit+'^{tree}'),
            'Message':git('show','-s','--format=%B',commit),'Changes':changes})
    dest=ROOT/'11_reports/cache';dest.mkdir(exist_ok=True)
    out={'Repository':'duchangchen11/Prediction_Hivt','Branch':'stage12a/candidate-semantic-consistency-audit',
        'Base':BASE,'BaseTree':git('rev-parse',BASE+'^{tree}'),'NativeHEAD':git('rev-parse','HEAD'),
        'Commits':rows,'Blobs':list(blobs.values()),'TotalDistinctBlobBytes':sum(r['Size'] for r in blobs.values()),
        'StageScopeOnly':True,'LargeLocalArraysExcluded':True}
    (dest/'stage12a_publication_manifest.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print('PUBLICATION_SCOPE_PASS',len(commits),'commits',len(blobs),'blobs',out['TotalDistinctBlobBytes'],'bytes')
if __name__=='__main__':main()
