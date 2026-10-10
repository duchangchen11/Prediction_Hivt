"""Stage17-scoped writes; immutable Stage15B predictor and candidate interfaces."""
from pathlib import Path
import sys,os,json,hashlib,random
ROOT=Path(__file__).resolve().parents[1];PROJECT=ROOT.parents[1]
OLD=PROJECT/'outputs/stage15b_isolated_validation';S16=PROJECT/'outputs/stage16_paper_ready'
sys.path[:0]=[str(OLD/d) for d in ('00_manifest','05_candidate_interface','06_rank_training')]
import torch,numpy as np,pandas as pd
import stage15b_common as pc
import stage15b_heads as heads
from stage15b_common import sha256,read_json,seed_all,state_digest
from stage15b_heads import RoleStore,cpu_state,gradient
REG=ROOT/'00_manifest/stage17_registration.json'
def atomic_json(path,value):
 path=Path(path);assert path.resolve().is_relative_to(ROOT);path.parent.mkdir(parents=True,exist_ok=True)
 q=path.with_suffix(path.suffix+'.tmp');q.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');q.replace(path)
def atomic_torch(path,value):
 path=Path(path);assert path.resolve().is_relative_to(ROOT);path.parent.mkdir(parents=True,exist_ok=True)
 q=path.with_suffix('.pt.tmp');torch.save(value,q);q.replace(path)
def dump(path,rows):
 path=Path(path);assert path.resolve().is_relative_to(ROOT);path.parent.mkdir(parents=True,exist_ok=True)
 pd.DataFrame(rows).to_csv(path,index=False,float_format='%.12g')
def runtime():
 torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
 torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
def protect(fitting=False):
 def hook(event,args):
  if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
  p=Path(os.fsdecode(args[0])).resolve();mode=args[1] if len(args)>1 else '';flags=args[2] if len(args)>2 else 0
  write=isinstance(mode,str) and any(c in mode for c in 'wax+') or isinstance(flags,int) and bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
  if write and p.is_relative_to(PROJECT):assert p.is_relative_to(ROOT),'Historical write forbidden: '+str(p)
  if fitting and p.is_relative_to(PROJECT/'outputs'):
   assert 'OuterTest' not in p.parts and not any(k in p.parts for k in ('val','test','HeadDev')),'Forbidden fitting read: '+str(p)
   assert not p.is_relative_to(OLD/'08_evaluation/cache'),'Historical Outer performance forbidden during fitting'
 sys.addaudithook(hook)
def verify_registration():
 r=read_json(REG)
 for p,h in r['FrozenSources'].items():assert sha256(ROOT/p)==h,'Registered source changed '+p
 assert sha256(pc.PROTOCOL)==r['Stage15BProtocolSHA256']
 return r
def scorer(seed,device='cuda'):
 """Execute the unmodified pinned public classes, never a replacement MLP."""
 import ast
 audit=read_json(S16/'02_literature/TNT-Trajectory-Prediction_code_audit.json')
 scope={'torch':torch,'nn':torch.nn,'F':torch.nn.functional}
 for name,names in [('basic_module.py',{'MLP'}),('scoring_and_selection.py',{'distance_metric','TrajScoreSelection'})]:
  key='core/model/layers/'+name;p=S16/'02_literature/cache/TNT-Trajectory-Prediction'/key
  assert sha256(p)==audit['code'][key]['sha256']
  tree=__import__('ast').parse(p.read_text());nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
  assert len(nodes)==len(names)
  exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(p),'exec'),scope)
 seed_all(seed);m=scope['TrajScoreSelection'](64,horizon=12,hidden_dim=64,temper=.01,device=torch.device(device)).to(device)
 assert sum(p.numel() for p in m.parameters())==16001
 return m
def r2_head(fold):
 m=heads.fresh('R2',fold);path=heads.cp_path(fold,'R2')
 m.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['state_dict'],strict=True)
 m.eval().requires_grad_(False);return m
def predictor(fold):
 r=read_json(OLD/'04_predictor_checkpoints/stage15b_all_frozen.json')['Checkpoints'][fold-1]
 p=OLD/r['path'];assert sha256(p)==r['sha256']
 m=pc.model_new(r['seed']);s=torch.load(p,map_location='cpu',weights_only=False)
 m.load_state_dict(s['state_dict'],strict=True);m.eval().requires_grad_(False)
 assert state_digest(m.state_dict())==r['state_sha256'];return m,r
def groups(frame):
 t=frame.agent_type.to_numpy();s=frame.motion_state.to_numpy()
 return {'Overall':np.ones(len(frame),bool),**{k:t==k for k in ('Vehicle','Pedestrian','Bicycle')},'MovingVehicle':(t=='Vehicle')&(s=='vehicle.moving')}
class TNTStore:
 def __init__(self,fold,role):
  if role=='OuterTest':assert read_json(ROOT/'05_training/stage17_all_frozen.json')['Status']=='FROZEN_ALL_COMPLETE'
  self.base=RoleStore(fold,role);self.fold=fold;self.role=role;self.frame=self.base.frame;self.ids=self.base.ids;self.types=self.base.types
  self.folder=ROOT/f'03_context/cache/fold{fold}/{role}';self.manifest=read_json(self.folder/'manifest.json')
  assert self.manifest['Status']=='PASS' and self.manifest['IdentityCSV_SHA256']==sha256(self.base.folder/'targets.csv')
  self.arrays={k:np.load(self.folder/(k+'.npy'),mmap_mode='r') for k in ('context','trajectory','gt')}
 def inputs(self,ids):
  # Labels are deliberately excluded from this callable.
  return tuple(torch.from_numpy(np.array(self.arrays[k][ids],copy=True)).cuda() for k in ('context','trajectory'))
 def labels(self,ids):return torch.from_numpy(np.array(self.arrays['gt'][ids],copy=True)).cuda()
def cp_path(fold):return ROOT/f'05_training/fold{fold}/stage17_best.pt'
