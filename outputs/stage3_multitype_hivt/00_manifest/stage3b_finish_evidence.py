"""Wait for the registered training process, then run only evaluation and audits."""
import argparse
from datetime import datetime,timezone
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'00_manifest'))
from stage3b_common import read_json,atomic_json,SUMMARY

def now():return datetime.now(timezone.utc).isoformat()

def alive(pid):
    try:
        os.kill(pid,0)
        return (Path('/proc')/str(pid)/'stat').read_text().split()[2]!='Z'
    except (ProcessLookupError,FileNotFoundError):return False

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--training-pid',required=True,type=int);args=parser.parse_args()
    state_path=ROOT/'00_manifest/stage3b_evidence_pipeline_state.json'
    if state_path.exists():
        state=read_json(state_path);assert state['training_pid']==args.training_pid
        assert not state['training_started_by_pipeline'] and not state['Stage4_executed']
        state['resumed_UTC']=now()
    else:
        state={'status':'WAITING_FOR_FORMAL_TRAINING','training_pid':args.training_pid,'started_UTC':now(),
            'training_started_by_pipeline':False,'Stage4_executed':False,'steps':[]}
    atomic_json(state_path,state)
    while alive(args.training_pid):time.sleep(45)
    assert SUMMARY.exists() and read_json(SUMMARY)['status']=='COMPLETE','Training exited without successful completed summary; no evaluation or retraining started.'
    scripts=(('fresh_evaluation','04_evaluation/stage3b_evaluate.py'),
        ('qualitative','05_figures/stage3b_plot_cases.py'),('quantitative','05_figures/stage3b_plot_quantitative.py'),
        ('export_QA','05_figures/stage3b_check_exports.py'),('final_audit','00_manifest/stage3b_final_audit.py'))
    for name,relative in scripts:
        if any(e['task']==name and e['status']=='PASS' for e in state['steps']):
            print('EVIDENCE_TASK_RETAIN_PASS',name,flush=True);continue
        attempt=sum(e['task']==name for e in state['steps'])+1
        suffix='' if attempt==1 else '_retry'+str(attempt)
        log=ROOT/'08_logs'/('stage3b_'+name+suffix+'.log');entry={'task':name,'script':relative,'attempt':attempt,
            'status':'RUNNING','started_UTC':now(),'log':str(log.relative_to(ROOT))}
        state['status']='RUNNING';state['steps'].append(entry);atomic_json(state_path,state)
        print('EVIDENCE_TASK_START',name,flush=True)
        with log.open('w') as handle:result=subprocess.run([sys.executable,'-u',str(ROOT/relative)],stdout=handle,stderr=subprocess.STDOUT,cwd=ROOT.parents[1])
        entry.update(status='PASS' if result.returncode==0 else 'FAIL',exit_code=result.returncode,finished_UTC=now())
        atomic_json(state_path,state)
        if result.returncode:
            state['status']='FAIL';atomic_json(state_path,state)
            raise RuntimeError('Evidence task failed: '+name+'; inspect '+str(log))
        print('EVIDENCE_TASK_PASS',name,flush=True)
    state.update(status='PASS',finished_UTC=now(),manual_visual_review_and_final_report_still_required=True)
    atomic_json(state_path,state);print('STAGE3B_EVIDENCE_PIPELINE=PASS',flush=True)

if __name__=='__main__':main()
