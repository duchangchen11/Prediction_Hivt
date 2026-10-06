"""Proceed through cache audit, TRAIN features and the two heads after cache PASS."""
from pathlib import Path
import json,os,subprocess,time
ROOT=Path(__file__).resolve().parents[1]
PYTHON='/home/lrj/anaconda3/envs/ped_intent/bin/python'
STATE=ROOT/'00_manifest/stage6a_pipeline_state.json'

def state(phase,status='RUNNING'):
    temp=STATE.with_suffix('.json.tmp');temp.write_text(json.dumps({'phase':phase,'status':status,'PID':os.getpid()},indent=2)+'\n');temp.replace(STATE)

def main():
    state('WAIT_PREDICTION_CACHE')
    path=ROOT/'01_cache/stage6a_cache_manifest.json'
    while not path.exists() or json.loads(path.read_text()).get('status')!='PASS':time.sleep(30)
    steps=(('CACHE_AUDIT','01_cache/stage6a_cache.py',['--audit'],'stage6a_cache_audit.log'),
           ('TRAIN_FEATURES','02_features/stage6a_build_features.py',[],'stage6a_train_features.log'),
           ('TRAIN_R1_R2','03_training/stage6a_train.py',[],'stage6a_training.log'))
    for phase,script,args,name in steps:
        state(phase);print('PIPELINE_START',phase,flush=True)
        with (ROOT/'08_logs'/name).open('w') as log:
            result=subprocess.run([PYTHON,'-u',str(ROOT/script),*args],stdout=log,stderr=subprocess.STDOUT,
                                  env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
        if result.returncode:state(phase,'FAIL');raise RuntimeError(phase+' failed: '+name)
        print('PIPELINE_PASS',phase,flush=True)
    state('HEADS_COMPLETE_WAIT_FINAL_VAL','READY');print('STAGE6A_HEADS_READY',flush=True)

if __name__=='__main__':main()
