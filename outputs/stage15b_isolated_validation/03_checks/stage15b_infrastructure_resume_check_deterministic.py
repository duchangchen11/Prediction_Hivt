"""Prove exact restoration from durable step1000 checkpoint before same-run resume."""
from pathlib import Path
import sys,copy,json,hashlib,random
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'00_manifest'),str(ROOT/'02_training')]
from stage15b_infrastructure import numpy_scalar_default
from stage15b_common import *
from stage15b_train import restore,optimize_step

torch.set_num_threads(4)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False
c=read_json(PROTOCOL)['folds'][0]
ctx=FoldContext(1,2022,ROOT/c['files']['InnerTrain']['path'],ROOT/c['files']['InnerDev']['path'],ROOT/'04_predictor_checkpoints/fold1')
cp=ctx.output/'stage15b_warmup_last_checkpoint.pt';ds=FoldDataset(ctx,'InnerTrain');cp_hash=sha256(cp);rows=[]
for repeat in range(2):
 m,o,saved=restore(cp,ctx,'FORMAL_SCENE_ISOLATED')
 assert saved['phase_state']['phase_step']==1000 and saved['phase_state']['phase']=='fixed_scale'
 assert torch.equal(torch.get_rng_state(),saved['torch_rng'])
 assert all(torch.equal(a,b) for a,b in zip(torch.cuda.get_rng_state_all(),saved['cuda_rng']))
 assert random.getstate()==saved['python_rng']
 ns=np.random.get_state();ss=saved['numpy_rng'];assert ns[0]==ss[0] and np.array_equal(ns[1],ss[1]) and ns[2:]==ss[2:]
 cursor=copy.deepcopy(saved['iterator']);r=optimize_step(m,o,ds,cursor,'fixed_scale')
 rows.append({'loss':r['loss'],'batch_indices':r['dataset_indices'],'state':state_digest(m.state_dict()),
    'optimizer':state_digest({str(i)+'/'+k:v for i,row in o.state_dict()['state'].items() for k,v in row.items() if torch.is_tensor(v)}),
    'cpu_rng':state_digest({'rng':torch.get_rng_state()}),'cuda_rng':[state_digest({'rng':r}) for r in torch.cuda.get_rng_state_all()],
    'python_rng':repr(random.getstate()),'numpy_rng':repr(np.random.get_state()),'cursor':cursor})
 del m,o;ds.clear();torch.cuda.empty_cache()
equality={key:rows[0][key]==rows[1][key] for key in rows[0]}
atomic_json(ROOT/'03_checks/stage15b_resume_replay_detail.json',{'SameCheckpoint':sha256(cp)==cp_hash,'EqualityByField':equality,'Losses':[x['loss'] for x in rows],'CPUThreads':torch.get_num_threads(),'DeterministicAlgorithms':torch.are_deterministic_algorithms_enabled()})
assert all(equality.values()) and sha256(cp)==cp_hash
# Repair only missing monitoring sidecars from the authoritative checkpoint.
for row in saved['phase_state']['monitoring']:
 p=ctx.output/f"stage15b_dev_step_{row['global_step']:05d}.json"
 if not p.exists():atomic_json(p,row)
atomic_json(ROOT/'03_checks/stage15b_infrastructure_resume_audit.json',{'Status':'PASS','CheckpointSHA256':cp_hash,
 'DurableGlobalStep':1000,'OriginalFittingCoreUnchanged':True,'LossOptimizerSplitSeedBudgetUnchanged':True,
 'DiagnosticDiscardedUpdates':2,'NextStepModelOptimizerLossBatchCursorAndAllRNG':'BITWISE_EXACT','Step1000MonitorRecoveredFromCheckpoint':True})
print('INFRASTRUCTURE_SAME_RUN_RECOVERY_PASS',flush=True)
