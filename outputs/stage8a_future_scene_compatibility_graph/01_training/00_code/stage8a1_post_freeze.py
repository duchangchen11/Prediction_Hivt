"""Sequential frozen-checkpoint evaluation and delivery; no training API."""
from stage8a1_common import *
import subprocess
PYTHON='/home/lrj/anaconda3/envs/ped_intent/bin/python'

def frozen_check():
    f=read_json(FROZEN);assert f['status']=='FROZEN_ALL_COMPLETE'
    for row in f['checkpoints']:assert sha256(PROJECT/row['Path'])==row['SHA256']
    assert f['normalization_sha256']==sha256(NORM);verify()

def main():
    frozen_check();steps=[
        ('VAL',ROOT/'03_evaluation/stage8a1_resume_evaluate.py',ROOT/'03_evaluation/stage8a1_complete.json'),
        ('BOOTSTRAP',ROOT/'04_bootstrap/stage8a1_analyze.py',ROOT/'04_bootstrap/stage8a1_bootstrap_audit.json'),
        ('CASES',ROOT/'07_cases/stage8a1_resume_cases.py',ROOT/'07_cases/stage8a1_case_manifest.json'),
        ('EFFICIENCY',ROOT/'08_efficiency/stage8a1_resume_efficiency.py',ROOT/'08_efficiency/stage8a1_efficiency_audit.json'),
        ('FINAL_REPORT',ROOT/'09_reports/stage8a1_finalize_resumed.py',ROOT/'09_reports/stage8a1_resumed_final_audit.json')]
    env=dict(os.environ);env['PYTHONDONTWRITEBYTECODE']='1'
    for label,script,complete in steps:
        frozen_check()
        if complete.exists():
            assert read_json(complete)['status'] in ('PASS','COMPLETE');print('PRESERVE_COMPLETED_PHASE',label,flush=True);continue
        print('START_PHASE',label,flush=True)
        subprocess.run([PYTHON,'-u',str(script)],cwd=PROJECT,env=env,check=True)
    frozen_check();print('STAGE8A1_COMPLETE_STOP',flush=True)

if __name__=='__main__':main()
