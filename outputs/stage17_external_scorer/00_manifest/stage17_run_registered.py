"""Sequential single-GPU runner; every gate must pass, no automatic retries."""
from pathlib import Path
import sys,subprocess
sys.path.insert(0,str(Path(__file__).resolve().parent))
from stage17_common import *
def run(script,*args):
 subprocess.run([sys.executable,str(ROOT/script),*map(str,args)],check=True)
def main():
 verify_registration()
 for fold in (1,2,3):
  for role in ('InnerTrain','InnerDev'):run('03_context/stage17_export.py','--fold',fold,'--role',role)
 run('04_checks/stage17_preflight.py')
 records=[read_json(ROOT/f'03_context/stage17_fold{k}_{role}.json') for k in (1,2,3) for role in ('InnerTrain','InnerDev')]
 assert all(r['Status']=='PASS' and r['CachedCandidatesBitwiseEqual'] and r['FuturePerturbationInputAndOutputBitwiseEqual'] for r in records)
 n=sum(r['PairedIntegrityBatches'] for r in records);assert n>=100
 assert read_json(ROOT/'04_checks/stage17_preflight.json')['Status']=='PASS'
 atomic_json(ROOT/'00_manifest/stage17_pretraining_gate.json',{'Status':'PASS','IntegrityBatches':n,'RegistrationSHA256':sha256(REG),
 'PassedBeforeFormalFitting':True,'OuterMetricsViewed':False,'IntegrityRecordsSHA256':{f'{r["Fold"]}/{r["Role"]}':sha256(ROOT/f'03_context/stage17_fold{r["Fold"]}_{r["Role"]}.json') for r in records}})
 for fold in (1,2,3):run('05_training/stage17_train.py','--fold',fold)
 run('05_training/stage17_train.py','--freeze')
 for fold in (1,2,3):run('03_context/stage17_export.py','--fold',fold,'--role','OuterTest')
 print('STAGE17_ALL_FITS_AND_EXPORTS_COMPLETE',flush=True)
if __name__=='__main__':main()
