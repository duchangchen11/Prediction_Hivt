"""Small-artifact publication, restricted to new Stage8A-1 paths."""
from pathlib import Path
import subprocess,json,base64
PROJECT=Path(__file__).resolve().parents[4];ROOT=Path(__file__).resolve().parents[2]
def git(*a):return subprocess.check_output(['git',*a],cwd=PROJECT,text=True).strip()
def main():
    head=git('rev-parse','HEAD');parent=git('rev-parse','HEAD^');assert parent=='df28e8638a1901422d7cded42430a32ea580f02c'
    prefix=str(ROOT.relative_to(PROJECT))+'/'
    files=[]
    for name in git('diff','--name-only',parent,head).splitlines():
        assert name.startswith(prefix),name
        local=name[len(prefix):]
        assert local.startswith(('01_training/','03_evaluation/','04_bootstrap/','05_diagnostics/','07_cases/','08_efficiency/')) or local.startswith(('00_manifest/stage8a1_','06_tables/stage8a1_','09_reports/stage8a1_')),name
        assert not name.endswith(('.pt','.npy','.npz','.log')) and '/cache/' not in name
        data=(PROJECT/name).read_bytes();assert len(data)<2_000_000
        files.append({'path':name,'sha':git('rev-parse',head+':'+name),'content':base64.b64encode(data).decode()})
    payload={'parent':parent,'local_commit':head,'base_tree':git('rev-parse',parent+'^{tree}'),'tree':git('rev-parse',head+'^{tree}'),
        'message':git('log','-1','--format=%B'),'files':files}
    dest=Path(__file__).parent/'stage8a1_git_upload_payload.json';dest.write_text(json.dumps(payload,separators=(',',':')))
    print(json.dumps({'files':len(files),'bytes':dest.stat().st_size,'parent':parent,'tree':payload['tree']}))
if __name__=='__main__':main()
