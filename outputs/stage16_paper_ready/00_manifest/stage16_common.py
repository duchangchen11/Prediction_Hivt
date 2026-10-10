"""Stage16-only outputs and read-only frozen Stage15B interfaces."""
from pathlib import Path
import sys,os,json,hashlib
ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT.parents[1]
OLD=PROJECT/'outputs/stage15b_isolated_validation'
sys.path[:0]=[str(OLD/d) for d in ('00_manifest','05_candidate_interface','06_rank_training','08_evaluation')]
import stage15b_heads as original
from stage15b_heads import RoleStore,assess,gradient,cpu_state,PARAMS
from stage15b_ranking import NoGraphReranker,MatchedNoGraphReranker,SparseGraphReranker,ReliabilityHead,objective
import torch,numpy as np,pandas as pd
from stage15b_common import sha256,read_json,seed_all,state_digest
def atomic_json(path,value):
 path=Path(path);assert path.resolve().is_relative_to(ROOT);path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)
def atomic_torch(path,value):
 path=Path(path);assert path.resolve().is_relative_to(ROOT);path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_suffix('.pt.tmp');torch.save(value,tmp);os.replace(tmp,path)
def dump(path,rows):
 path=Path(path);assert path.resolve().is_relative_to(ROOT);path.parent.mkdir(parents=True,exist_ok=True)
 pd.DataFrame(rows).to_csv(path,index=False,float_format='%.12g')
def runtime():
 torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
 torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
def groups(f):
 t=f.agent_type.to_numpy();s=f.motion_state.to_numpy()
 return {'Overall':np.ones(len(f),bool),**{k:t==k for k in ('Vehicle','Pedestrian','Bicycle')},'MovingVehicle':(t=='Vehicle')&(s=='vehicle.moving')}
MODELS=('R0','R2','NG-A','NG-C','G-A','G-C','Matched-NG-C')
FIELDS=('Top1FDE','Top1ADE','OracleGap','HitRate','MRR','SoftCE','ExpectedRegret','NormalizedExpectedRegret','PredictionEntropy','Top1Probability','Top1Top2Margin','OracleModeProbability','minADEOracle6','minFDE6','MR6')
def frozen_eval():
 folder=OLD/'08_evaluation/cache';f=pd.read_csv(folder/'actor_records.csv')
 m=np.load(folder/'metrics.npy',mmap_mode='r');p=np.load(folder/'probabilities.npy',mmap_mode='r');z=np.load(folder/'logits.npy',mmap_mode='r')
 assert len(f)==260151 and f.scene_token.nunique()==630
 return f,m,p,z
def protect_fit():
 def hook(event,args):
  if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
   p=Path(os.fsdecode(args[0])).resolve();mode=args[1] if len(args)>1 else '';flags=args[2] if len(args)>2 else 0
   writing=bool(isinstance(mode,str) and any(c in mode for c in 'wax+')) or bool(isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
   if writing and p.is_relative_to(PROJECT):assert p.is_relative_to(ROOT),'Historical project write forbidden: '+str(p)
   if p.is_relative_to(OLD/'05_candidate_interface/cache'):assert 'OuterTest' not in p.parts,'Outer candidate read forbidden during seed fitting'
   if p.is_relative_to(PROJECT/'outputs') and not p.is_relative_to(ROOT):
    assert not any(x in p.parts for x in ('val','test','HeadDev')),'VAL/test/HeadDev data forbidden during fitting'
   if p==OLD/'08_evaluation/cache/metrics.npy' or p==OLD/'08_evaluation/cache/probabilities.npy':raise AssertionError('Frozen Outer performance forbidden during fitting')
 sys.addaudithook(hook)
