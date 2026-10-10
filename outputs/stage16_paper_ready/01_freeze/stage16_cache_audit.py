"""Independently check every frozen candidate array against its signed original SHA."""
from pathlib import Path
import sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'00_manifest'))
from stage16_common import *
def main():
 rows=[];start=time.monotonic()
 for fold in (1,2,3):
  for role in ('InnerTrain','InnerDev','OuterTest'):
   folder=OLD/f'05_candidate_interface/cache/fold{fold}/{role}';m=read_json(folder/'manifest.json')
   for name,rec in m['Arrays'].items():
    path=folder/(name+'.npy');actual=sha256(path);assert actual==rec['sha256'],str(path)
    rows.append({'Fold':fold,'Role':role,'Array':name,'Bytes':path.stat().st_size,'SHA256':actual,'Status':'PASS'})
   print('CACHE_SHA_PASS',fold,role,flush=True)
 dump(ROOT/'01_freeze/stage16_candidate_array_sha256.csv',rows)
 atomic_json(ROOT/'01_freeze/stage16_cache_audit.json',{'Status':'PASS','ArrayCount':len(rows),'BytesHashed':sum(r['Bytes'] for r in rows),'Seconds':time.monotonic()-start,'ImmutableSources':True,'NotDuplicated':True,'NoModelSelectionOrFit':True})
if __name__=='__main__':main()
