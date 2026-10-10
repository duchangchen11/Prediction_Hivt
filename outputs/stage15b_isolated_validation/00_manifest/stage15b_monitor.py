"""Small read-only progress snapshot, no candidate or Outer performance reads."""
from pathlib import Path
import json,subprocess,time
ROOT=Path(__file__).resolve().parents[1]

def read(p):return json.loads(p.read_text())

def snapshot():
 x={'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'predictors':[]}
 live=ROOT/'00_manifest/stage15b_live_status.json'
 x['stage_status']=read(live) if live.exists() else {'Status':'RUNNING'}
 for fold in (1,2,3):
  folder=ROOT/f'04_predictor_checkpoints/fold{fold}';record={'fold':fold,'started':(folder/'stage15b_initialization.json').exists()}
  witness=folder/'stage15b_batch_sources.log'
  if witness.exists() and witness.stat().st_size:
   with witness.open('rb') as f:f.seek(max(0,witness.stat().st_size-16000));lines=f.read().splitlines()
   try:
    row=json.loads(lines[-1]);record.update(step=row['step'],phase=row['phase'])
   except (ValueError,IndexError):pass
  monitors=sorted(folder.glob('stage15b_dev_step_*.json'))
  if monitors:
   row=read(monitors[-1]);record.update(last_dev_step=row['global_step'],dev_FDE=row['measured']['full_horizon_Overall_minFDE6'],best_step=row['best_step'],bad=row['bad_validations'])
  for phase in ('warmup','nll'):
   p=folder/f'stage15b_{phase}_summary.json'
   if p.exists():s=read(p);record[phase]={'status':s['status'],'steps':s['phase_steps'],'best_FDE':s['best_overall_FDE'],'best_step':s['best_global_step']}
  p=folder/'stage15b_failure.json'
  if p.exists():record['failure']=read(p)['exception']
  x['predictors'].append(record)
 for p in (ROOT/'04_predictor_checkpoints/stage15b_all_frozen.json',ROOT/'07_rank_checkpoints/stage15b_all_frozen.json'):
  x[p.parent.name]=p.exists()
 print(json.dumps(x,ensure_ascii=False))

if __name__=='__main__':snapshot()
