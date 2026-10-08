"""One locked HeadDev pass, analysis, small timing audit and final STOP. No training."""
from pathlib import Path
import sys,subprocess,json
ROOT=Path(__file__).resolve().parents[1]
def main():
    frozen=json.loads((ROOT/'02_checkpoints/stage10a_checkpoint_manifest.json').read_text())
    assert frozen['status']=='FROZEN_ALL_COMPLETE' and frozen['training_prohibited']
    for p in ('05_diagnostics/stage10a_all_actor_inference.py','03_evaluation/stage10a_evaluate.py','04_bootstrap/stage10a_analyze.py',
        '08_efficiency/stage10a_efficiency.py','09_reports/stage10a_finalize.py'):
        subprocess.run([sys.executable,'-u',str(ROOT/p)],check=True)
    print('STOP: HeadDev feasibility phase complete; no official VAL or Stage10B',flush=True)
if __name__=='__main__': main()
