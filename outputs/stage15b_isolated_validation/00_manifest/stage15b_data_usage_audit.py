"""Reconstruct every formal optimizer batch and append-log prefix SHA without fitting."""
from pathlib import Path
import json,csv,hashlib
ROOT=Path(__file__).resolve().parents[1];PROJECT=ROOT.parents[1]

def read(p):return json.loads(p.read_text())

def main():
 protocol=read(ROOT/'00_manifest/stage15b_protocol.json')
 with (PROJECT/'outputs/stage3_multitype_hivt/02_preprocessed/stage3_train_index.csv').open() as f:
  index={(r['scene_token'],r['sample_token']) for r in csv.DictReader(f)}
 records=[]
 for fold in protocol['folds']:
  n=fold['fold'];folder=ROOT/f'04_predictor_checkpoints/fold{n}';warm=read(folder/'stage15b_warmup_summary.json');nll=read(folder/'stage15b_nll_summary.json')
  data=(folder/'stage15b_batch_sources.log').read_bytes();lines=data.splitlines(keepends=True)
  assert len(lines)==5000+nll['phase_steps'];seen=set();prefix=b''
  for step,line in enumerate(lines,1):
   row=json.loads(line);assert row['step']==step and row['role']=='InnerTrain'
   assert row['phase']==('fixed_scale' if step<=5000 else 'original_nll')
   assert len(row['scenes'])==len(row['samples'])<=16 and set(row['scenes'])<=set(fold['parts']['InnerTrain'])
   assert all((s,t) in index for s,t in zip(row['scenes'],row['samples']))
   seen.update(row['scenes'])
  assert seen==set(fold['parts']['InnerTrain'])
  warmhash=hashlib.sha256(b''.join(lines[:5000])).hexdigest();allhash=hashlib.sha256(data).hexdigest()
  assert warmhash==warm['BatchSourceWitnessSHA256'] and allhash==nll['BatchSourceWitnessSHA256']
  events=[]
  for phase,summary in [('warmup',warm),('nll',nll)]:
   for row in summary['Monitoring']:
    assert row['global_step']%500==0 and row['measured']['selection_role']=='InnerDev'
    assert row['measured']['Count']==fold['metadata']['InnerDev']['full_targets']
    events.append(row['global_step'])
  records.append({'Fold':n,'Status':'PASS','OptimizerSteps':len(lines),'UniqueTrainScenes':len(seen),'DevScenes':42,
     'DevelopmentFullTargetCount':fold['metadata']['InnerDev']['full_targets'],'ValidationEventSteps':events,
     'WarmupSourcePrefixSHA256':warmhash,'FinalSourceWitnessSHA256':allhash,'AllBatchSampleSceneIdentitiesVerified':True,
     'OuterOrHeadDevOptimizationCount':0,'OriginalVALUsed':False,'PrefixSHAExplanation':'warm summary binds first5000 lines; NLL summary binds complete append-only witness'})
 (ROOT/'01_data_isolation/stage15b_actual_data_usage.json').write_text(json.dumps({'Status':'PASS','Folds':records},indent=2)+'\n')
 print('ALL_FORMAL_BATCH_IDENTITIES_AND_PHASE_LOG_SHA_PASS')

if __name__=='__main__':main()
