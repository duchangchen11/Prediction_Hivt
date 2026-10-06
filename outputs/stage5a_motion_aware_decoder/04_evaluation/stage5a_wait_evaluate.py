"""Run the locked evaluation package once formal training is complete."""
from pathlib import Path
import sys
import os
import subprocess
import time
import json
ROOT=Path(__file__).resolve().parents[1]
PYTHON='/home/lrj/anaconda3/envs/ped_intent/bin/python'
STATE=ROOT/'00_manifest/stage5a_pipeline_state.json'


def state(phase,status='RUNNING'):
    temp=STATE.with_suffix('.json.tmp')
    temp.write_text(json.dumps({'phase':phase,'status':status,'training_completed_before_evaluation':phase!='WAIT_TRAINING',
                               'pipeline_PID':os.getpid()},indent=2)+'\n')
    os.replace(temp,STATE)


def main():
    state('WAIT_TRAINING')
    summary=ROOT/'03_training/stage5a_training_summary.json'
    while not summary.exists() or json.loads(summary.read_text()).get('status')!='COMPLETE':
        time.sleep(40)
    steps=(('FRESH_VAL','04_evaluation/stage5a_evaluate.py','stage5a_evaluation.log'),
           ('EFFICIENCY','04_evaluation/stage5a_efficiency.py','stage5a_efficiency.log'),
           ('QUANTITATIVE_FIGURES','05_figures/stage5a_figures.py','stage5a_quantitative.log'),
           ('MATCHED_CASES','05_figures/stage5a_cases.py','stage5a_cases.log'))
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
    for phase,script,log in steps:
        state(phase);print('PIPELINE_START',phase,flush=True)
        with (ROOT/'08_logs'/log).open('w') as out:
            result=subprocess.run([PYTHON,'-u',str(ROOT/script)],stdout=out,stderr=subprocess.STDOUT,env=env)
        if result.returncode:
            state(phase,'FAIL');raise RuntimeError('Pipeline failed: '+phase+'; inspect '+log)
        print('PIPELINE_PASS',phase,flush=True)
    state('READY_FOR_MANUAL_VISUAL_QA','READY')
    print('STAGE5A_READY_FOR_MANUAL_VISUAL_QA',flush=True)


if __name__=='__main__':main()
