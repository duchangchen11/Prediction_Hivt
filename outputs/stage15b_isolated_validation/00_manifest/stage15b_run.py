"""Sequential one-GPU stage supervisor; stop and preserve evidence on any failure."""
from pathlib import Path
import os,sys,subprocess,time,json,hashlib,traceback
ROOT=Path(__file__).resolve().parents[1];PROJECT=ROOT.parents[1]
PYTHON='/home/lrj/anaconda3/envs/ped_intent/bin/python'
ENV={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OPENBLAS_NUM_THREADS':'1','CUBLAS_WORKSPACE_CONFIG':':4096:8'}

def save(path,x):
 path=ROOT/path;tmp=path.with_suffix('.json.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');tmp.replace(path)

def read(path):return json.loads((ROOT/path).read_text())

def run(tag,script,*args):
 print('START',tag,flush=True);start=time.monotonic();log=ROOT/f'10_logs/stage15b_{tag}.log'
 save('00_manifest/stage15b_live_status.json',{'Status':'RUNNING','Phase':tag,'StartedUTC':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'Log':str(log.relative_to(ROOT))})
 with log.open('a') as f:
  p=subprocess.Popen([PYTHON,'-u',str(ROOT/'00_manifest/stage15b_infrastructure.py'),str(ROOT/script),*map(str,args)],env=ENV,cwd=PROJECT,stdout=f,stderr=subprocess.STDOUT)
  save(f'10_logs/stage15b_{tag}_process.json',{'PID':p.pid,'Command':[PYTHON,str(ROOT/script),*map(str,args)],'StartedUTC':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
  code=p.wait()
 receipt={'Phase':tag,'ExitCode':code,'Seconds':time.monotonic()-start,'Log':str(log.relative_to(ROOT))}
 save(f'10_logs/stage15b_{tag}_receipt.json',receipt)
 assert code==0,'Stop after failed '+tag+'; preserve logs/checkpoints, no protocol change'
 print('COMPLETE',tag,receipt['Seconds'],flush=True)

def main():
 assert read('00_manifest/stage15b_registration.json')['PredictorFullTrainingAuthorized']
 # Fold1 was started only after the preregistration commit reached GitHub.
 external=read('00_manifest/stage15b_fold1_resume_process.json');pid=external['PID']
 print('WAIT_FOR_ALREADY_REGISTERED_FOLD1',pid,flush=True)
 while not (ROOT/'04_predictor_checkpoints/fold1/stage15b_nll_summary.json').exists():
  assert not (ROOT/'04_predictor_checkpoints/fold1/stage15b_failure.json').exists(),'Fold1 FAILED_STOP'
  try:os.kill(pid,0)
  except ProcessLookupError:raise RuntimeError('Fold1 exited without qualified final summary')
  time.sleep(15)
 for fold in (2,3):
  seed=2022+100*(fold-1)
  run(f'predictor_fold{fold}','02_training/stage15b_train.py','--fold',fold,'--seed',seed,
      '--training-scenes',ROOT/f'01_data_isolation/stage15b_fold{fold}_InnerTrain.json',
      '--development-scenes',ROOT/f'01_data_isolation/stage15b_fold{fold}_InnerDev.json',
      '--output-dir',ROOT/f'04_predictor_checkpoints/fold{fold}','--mode','formal')
 for fold in (1,2,3):run(f'checkpoint_check_fold{fold}','03_checks/stage15b_checkpoint_check.py','--fold',fold)
 run('freeze_predictors','05_candidate_interface/stage15b_cache.py','--freeze')
 for fold in (1,2,3):run(f'candidate_fold{fold}','05_candidate_interface/stage15b_cache.py','--fold',fold)
 for fold in (1,2,3):
  run(f'head_check_fold{fold}','03_checks/stage15b_head_check.py','--fold',fold)
  run(f'ranking_fold{fold}','06_rank_training/stage15b_heads.py','--fold',fold)
 run('freeze_heads','06_rank_training/stage15b_heads.py','--freeze')
 run('outer_evaluation','08_evaluation/stage15b_evaluate.py')
 run('bootstrap','09_statistics/stage15b_statistics.py')
 run('extra_analysis','09_statistics/stage15b_extra.py')
 run('finalize','00_manifest/stage15b_finalize.py')
 save('00_manifest/stage15b_live_status.json',{'Status':'COMPLETE_PENDING_GIT_PUBLICATION','AllFormalStagesComplete':True})
 print('STAGE15B_COMPLETE_PENDING_GIT_PUBLICATION',flush=True)

if __name__=='__main__':
 try:main()
 except BaseException as e:
  save('00_manifest/stage15b_live_status.json',{'Status':'FAILED_STOP','Exception':repr(e),'Traceback':traceback.format_exc(),'NoProtocolChangePermitted':True})
  print(traceback.format_exc(),flush=True);sys.exit(1)
