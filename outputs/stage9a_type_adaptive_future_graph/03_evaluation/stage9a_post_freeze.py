"""Frozen-checkpoint secondary evaluation, bootstrap, diagnostics, report, STOP."""
from pathlib import Path
import os,sys,subprocess,json
ROOT=Path(__file__).resolve().parents[1]

def main():
    freeze=ROOT/'02_checkpoints/stage9a_checkpoint_manifest.json';assert json.loads(freeze.read_text())['status']=='FROZEN_ALL_COMPLETE'
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1');steps=[('03_evaluation/stage9a_evaluate.py',ROOT/'03_evaluation/stage9a_complete.json'),('04_bootstrap/stage9a_analyze.py',ROOT/'09_reports/stage9a_scientific_decision.json'),('07_cases/stage9a_cases.py',ROOT/'07_cases/stage9a_case_manifest.json'),('08_efficiency/stage9a_efficiency.py',ROOT/'08_efficiency/stage9a_efficiency_audit.json'),('09_reports/stage9a_finalize.py',ROOT/'09_reports/stage9a_final_audit.json')]
    for relative,done in steps:
        if done.exists():print('PRESERVE_COMPLETED',relative,flush=True);continue
        print('RUN',relative,flush=True);subprocess.run([sys.executable,'-u',str(ROOT/relative)],cwd=ROOT.parents[1],env=env,check=True)
    print('STAGE9_POST_FREEZE_COMPLETE_STOP',flush=True)

if __name__=='__main__':main()
