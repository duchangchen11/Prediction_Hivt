"""Selected new predictor readback and exact next-step recovery; discard diagnostic updates."""
from pathlib import Path
import sys,json,copy,time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'00_manifest'),str(ROOT/'02_training'),str(ROOT/'05_candidate_interface')]
from stage15b_common import *
from stage15b_train import restore,optimize_step
from stage15b_cache import context
from preprocessing.coordinates import ego_to_global
import sqlite3


def verify(fold):
 ctx=context(fold,'predictor');folder=ctx.output;n=read_json(folder/'stage15b_nll_summary.json');cp=ROOT/n['selected_checkpoint']
 assert sha256(cp)==n['selected_checkpoint_sha256']
 ds=FoldDataset(ctx,'InnerTrain');m,o,saved=restore(cp,ctx,'FORMAL_SCENE_ISOLATED')
 iterator=copy.deepcopy(saved['iterator']);batch,ids=next_training_batch(ds,copy.deepcopy(iterator),'original_nll');data=batch.cuda();m.eval()
 with torch.no_grad():
  out=m(model_input(data));ref={k:out[k].clone() for k in ('raw_prediction','mode_logits','mode_prob')};pred=m.ego_predictions(out,model_input(data))
  assert out['raw_prediction'].shape==(6,data.num_nodes,12,4) and pred.shape==(data.num_nodes,6,12,2)
  poisoned=data.clone();poisoned.positions[:,5:]=float('nan');poisoned.y.fill_(float('nan'));poisoned.future_mask=~poisoned.future_mask
  poisoned.target_mask=~poisoned.target_mask
  out2=m(model_input(poisoned));diff={k:float((ref[k]-out2[k]).abs().max()) for k in ref};assert max(diff.values())<1e-6
 del m,o;torch.cuda.empty_cache()
 results=[]
 for rep in range(2):
  ds.clear();m,o,s=restore(cp,ctx,'FORMAL_SCENE_ISOLATED');cursor=copy.deepcopy(s['iterator'])
  row=optimize_step(m,o,ds,cursor,'original_nll')
  rng={'CPU':state_digest({'rng':torch.get_rng_state()}),'CUDA':[state_digest({'rng':x}) for x in torch.cuda.get_rng_state_all()],
       'Python':repr(__import__('random').getstate()),'NumPy':repr(np.random.get_state())}
  opt_digest=state_digest({str(i)+'/'+k:v for i,r in o.state_dict()['state'].items() for k,v in r.items() if torch.is_tensor(v)})
  results.append({'loss':row['loss'],'ids':row['dataset_indices'],'state':state_digest(m.state_dict()),'optimizer':opt_digest,'rng':rng,'cursor':cursor})
  del m,o;torch.cuda.empty_cache()
 assert results[0]==results[1],'Next-step exact replay failed'
 # Compare raw stored timestamps and coordinates to source metadata for these Train-only windows.
 db=PROJECT/'outputs/stage2c_trainval_vehicle_baseline/02_preprocessed/stage2c_metadata_cache/stage2c_trajectory_metadata.sqlite'
 con=sqlite3.connect('file:'+str(db)+'?mode=ro',uri=True);coordinate_error=0.;timestamp_error=0.
 for idx in ids:
  g=ds[int(idx)];t0=json.loads(con.execute('SELECT payload FROM samples WHERE token=?',(g.sample_token,)).fetchone()[0])['timestamp']
  for i,annotations in enumerate(g.annotation_tokens):
   for t,token in enumerate(annotations):
    if not token:continue
    annotation=json.loads(con.execute('SELECT payload FROM annotations WHERE token=?',(token,)).fetchone()[0])
    assert annotation['instance_token']==g.instance_tokens[i]
    sample=json.loads(con.execute('SELECT payload FROM samples WHERE token=?',(annotation['sample_token'],)).fetchone()[0]);assert sample['scene_token']==g.scene_token
    actual=(sample['timestamp']-t0)/1e6;stored=float(g.history_times[t] if t<5 else g.future_times[t-5])
    timestamp_error=max(timestamp_error,abs(actual-stored))
    xy=ego_to_global(g.positions[i,t].numpy()[None],g.origin.numpy(),float(g.ego_yaw))[0]
    coordinate_error=max(coordinate_error,float(np.linalg.norm(xy-np.asarray(annotation['translation'][:2]))))
 con.close();assert timestamp_error<1e-5 and coordinate_error<1e-4
 atomic_json(ROOT/f'03_checks/stage15b_fold{fold}_checkpoint_checks.json',{'Status':'PASS','Fold':fold,'CheckpointSHA256':sha256(cp),
    'OutputK':6,'Parameters':650403,'GTInputPoisonMaxDiff':diff,'NextStepReplay':'BITWISE_EXACT',
    'DiagnosticOptimizerUpdates':2,'DiagnosticWeightsRetainedAsFormal':False,'OptimizerStateAndAllRNGEqual':True,
    'RawCoordinateMaxErrorMeters':coordinate_error,'RawTimestampMaxErrorSeconds':timestamp_error,
    'TimestampPolicy':'actual original source-clock values, no nominal-spacing threshold',
    'CoordinateChecksOnlyOn':'InnerTrain','OuterTestEvaluated':False})
 print('CHECKPOINT_RECOVERY_AND_GT_PASS',fold,flush=True)

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--fold',type=int,required=True,choices=(1,2,3));a=p.parse_args()
 torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
 verify(a.fold)
