"""Prepare only committed, stage-isolated small artifacts for the authorized push."""
from pathlib import Path
import subprocess
import base64
import json
ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT.parents[1]
PARENT='1b15908ab6daa8a800a45dcf243ab406e4869542'


def command(*args):
    return subprocess.check_output(['git',*args],cwd=PROJECT,text=True).strip()


def main():
    assert command('branch','--show-current')=='stage5a/motion-aware-residual-decoder'
    assert command('rev-parse','HEAD^')==PARENT
    assert not command('diff','--name-only') and not command('diff','--cached','--name-only')
    names=command('diff','--name-only',PARENT,'HEAD').splitlines()
    assert names and all(name.startswith(str(ROOT.relative_to(PROJECT))+'/') for name in names)
    rows=[]
    for name in names:
        path=PROJECT/name
        assert path.is_file() and path.suffix not in ('.pt','.pth','.ckpt','.log','.tmp')
        assert not path.name.startswith(('stage5a_actor_','stage5a_val_step_','stage5a_git_upload_'))
        assert path.name!='stage5a_router_actor_records.csv'
        mode,kind,sha=command('ls-tree','HEAD','--',name).split('\t')[0].split()
        assert kind=='blob' and mode=='100644'
        data=path.read_bytes()
        rows.append({'path':name,'mode':mode,'type':'blob','sha':sha,
                     'content':base64.b64encode(data).decode(),'encoding':'base64','bytes':len(data)})
    metadata={'rows':rows,'local_commit':command('rev-parse','HEAD'),
              'tree':command('rev-parse','HEAD^{tree}'),'parent':PARENT,
              'base_tree':command('rev-parse',PARENT+'^{tree}')}
    target=ROOT/'00_manifest/stage5a_git_upload_final.json'
    target.write_text(json.dumps(metadata))
    print(json.dumps({'payload':str(target.relative_to(PROJECT)),'chars':target.stat().st_size,
        'files':len(rows),'bytes_to_upload':sum(r['bytes'] for r in rows),'tree':metadata['tree'],'parent':PARENT}))


if __name__=='__main__':main()
