"""Finish authorized analyses after the separate training process freezes all24 heads."""
from pathlib import Path
import subprocess,sys,time,json,os
ROOT=Path(__file__).resolve().parents[1]
def main():
 marker=ROOT/'03_seed_stability/stage16_all_frozen.json';log=ROOT/'08_logs/stage16_train_heads.log'
 while not marker.exists():
  tail=log.read_text()[-5000:]
  if 'Traceback (most recent call last)' in tail:raise RuntimeError('Head fitting failed; no automatic retry permitted')
  time.sleep(5)
 assert json.loads(marker.read_text())['Status']=='FROZEN_ALL_COMPLETE'
 scripts=['03_seed_stability/stage16_evaluate_seeds.py','09_reports/stage16_seed_report.py','05_figures/stage16_seed_figure.py','05_figures/stage16_figure_qa.py','09_reports/stage16_finalize.py','00_manifest/stage16_final_audit.py']
 for script in scripts:
  print('PIPELINE_START',script,flush=True);subprocess.run([sys.executable,str(ROOT/script)],check=True)
 print('ALL_STAGE16_LOCAL_DELIVERABLES_COMPLETE_AWAIT_GIT_PUSH',flush=True)
if __name__=='__main__':main()
