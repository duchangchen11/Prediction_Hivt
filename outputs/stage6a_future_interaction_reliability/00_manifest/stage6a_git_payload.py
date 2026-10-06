"""Prepare committed Stage6A artifacts for the authorized branch push."""
from pathlib import Path
import subprocess,base64,json
ROOT=Path(__file__).resolve().parents[1];PROJECT=ROOT.parents[1]
PARENT='e5cec8f44d27608eddb7820a8f873b53d0b8ddab'
def command(*args):return subprocess.check_output(['git',*args],cwd=PROJECT,text=True).strip()
def main():
    assert command('branch','--show-current')=='stage6a/future-interaction-reliability'
    assert command('rev-parse','HEAD^')==PARENT
    assert not command('diff','--name-only') and not command('diff','--cached','--name-only')
    names=command('diff','--name-only',PARENT,'HEAD').splitlines()
    assert names and all(n.startswith(str(ROOT.relative_to(PROJECT))+'/') for n in names)
    rows=[]
    for name in names:
        path=PROJECT/name;assert path.is_file() and path.suffix not in ('.pt','.pth','.ckpt','.log','.tmp')
        assert not path.name.startswith(('stage6a_actor_','stage6a_git_upload_'))
        mode,kind,sha=command('ls-tree','HEAD','--',name).split('\t')[0].split();assert kind=='blob' and mode=='100644'
        data=path.read_bytes();rows.append({'path':name,'mode':mode,'type':'blob','sha':sha,
            'content':base64.b64encode(data).decode(),'encoding':'base64','bytes':len(data)})
    payload={'rows':rows,'local_commit':command('rev-parse','HEAD'),'tree':command('rev-parse','HEAD^{tree}'),
        'parent':PARENT,'base_tree':command('rev-parse',PARENT+'^{tree}')}
    target=ROOT/'00_manifest/stage6a_git_upload_final.json';target.write_text(json.dumps(payload))
    print(json.dumps({'payload':str(target.relative_to(PROJECT)),'chars':target.stat().st_size,'files':len(rows),
        'bytes_to_upload':sum(r['bytes'] for r in rows),'tree':payload['tree'],'parent':PARENT}))
if __name__=='__main__':main()
