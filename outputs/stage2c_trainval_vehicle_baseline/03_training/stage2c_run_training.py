"""Run the two authorized phases sequentially; abort on any training error."""
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
for stage,filename in (("warmup","stage2c_warmup_train.log"),("nll","stage2c_nll_train.log")):
    print("START_PHASE",stage,flush=True)
    with open(ROOT/"08_logs"/filename,"a") as handle:
        subprocess.run([sys.executable,str(ROOT/"03_training/stage2c_train.py"),"--stage",stage],stdout=handle,stderr=subprocess.STDOUT,check=True)
    print("COMPLETED_PHASE",stage,flush=True)
