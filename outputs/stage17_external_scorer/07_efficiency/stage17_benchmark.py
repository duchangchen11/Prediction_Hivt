"""Comparable cached head and observed-only whole engineering path timings."""
from pathlib import Path
import sys,time,shutil
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'00_manifest'))
from stage17_common import *
from stage15b_ranking import pack_window,normalized,head_forward
from torch_geometric.data import Batch
@torch.no_grad()
def benchmark():
 runtime();protect();verify_registration();assert read_json(ROOT/'05_training/stage17_all_frozen.json')['Status']=='FROZEN_ALL_COMPLETE'
 headrows=[];whole=[]
 for fold in (1,2,3):
  store=TNTStore(fold,'InnerDev');models={}
  for name in heads.NAMES:
   net=heads.fresh(name,fold);net.load_state_dict(torch.load(heads.cp_path(fold,name),map_location='cpu',weights_only=False)['state_dict']);net.eval().requires_grad_(False);models[name]=net
  tnt=scorer(2022+100*(fold-1));tnt.load_state_dict(torch.load(cp_path(fold),map_location='cpu',weights_only=False)['state_dict']);tnt.eval().requires_grad_(False)
  names=['R0',*heads.NAMES,'Adapted TNT Scoring'];pieces=[]
  for s in range(0,1024,128):
   ix=store.ids[s:s+128];args,_,_=store.base.batch(ix);pieces.append((args,store.base.r2(ix),store.inputs(ix)))
  for name in names:
   def forward_piece(piece):
    args,rf,(x,tr)=piece
    if name=='Adapted TNT Scoring':return tnt(x,tr)
    if name=='R0':return args[6].softmax(-1)
    return head_forward(models[name],name,args,rf)
   for _ in range(10):
    for piece in pieces:forward_piece(piece)
   torch.cuda.synchronize();ts=[];torch.cuda.reset_peak_memory_stats()
   for _ in range(50):
    torch.cuda.synchronize();t=time.perf_counter()
    for piece in pieces:forward_piece(piece)
    torch.cuda.synchronize();ts.append(time.perf_counter()-t)
   params=16001 if name=='Adapted TNT Scoring' else 0 if name=='R0' else heads.PARAMS[name]
   headrows.append({'Fold':fold,'Model':name,'Parameters':params,'BatchCycleTargets':1024,'Microbatch':128,'WarmupCycles':10,'RepeatedCycles':50,
    'MeanMSPer1024':1000*float(np.mean(ts)),'MedianMSPer1024':1000*float(np.median(ts)),'PeakAllocatedBytes':torch.cuda.max_memory_allocated(),
    'Scope':'preloaded CUDA inputs, head forward only; excludes H2D, feature/context export and Bicycle routing; R0 softmax only'})
  spec=read_json(pc.PROTOCOL)['folds'][fold-1];ctx=pc.FoldContext(fold,spec['seed'],OLD/spec['files']['InnerTrain']['path'],OLD/spec['files']['InnerDev']['path'],OLD/f'04_predictor_checkpoints/fold{fold}',install=False,stage='candidate')
  ds=pc.FoldDataset(ctx,'InnerDev');graphs=[ds[i] for i in range(16)];batch=Batch.from_data_list(graphs);observed=pc.model_input(batch.cuda());model,rec=predictor(fold)
  emb=[]
  def full(name):
   emb.clear()
   hook=model.decoder.register_forward_pre_hook(lambda mod,args:emb.append(args[0].detach().clone())) if name=='Adapted TNT Scoring' else None
   out=model(observed);pred=model.ego_predictions(out,observed)
   if hook:hook.remove()
   ptr=batch.ptr.tolist();targets=0
   for j,g in enumerate(graphs):
    lo,hi=ptr[j:j+2];t=torch.where((~g.padding_mask[:,:5]).sum(-1)>=2)[0];targets+=len(t);ti=t.cuda()
    if name=='Adapted TNT Scoring':
     center=observed.positions[lo:hi,4][ti,None,None];rot=out['rotation'][lo:hi][ti]
     tr=torch.matmul(pred[lo:hi][ti]-center,rot[:,None]);p=tnt(emb[0][lo:hi][ti,None],tr.reshape(len(t),6,24));p.argmax(-1)
    elif name=='R0':out['mode_prob'][lo:hi][ti].argmax(-1)
    else:
     w={'history':g.positions[:,:5],'history_padding':g.padding_mask[:,:5],'agent_type':g.agent_type,
      'ego_prediction':pred[lo:hi].cpu(),'mode_logits':out['mode_logits'][lo:hi].cpu(),'mode_prob':out['mode_prob'][lo:hi].cpu(),
      'scene_token':g.scene_token,'sample_token':g.sample_token,'instance_tokens':g.instance_tokens,'map_location':g.map_location,'origin':g.origin.numpy(),'yaw':float(g.ego_yaw),'source_role':'InnerDev'}
     pack=pack_window(w);assert torch.equal(pack['targets'],t);args,rf=normalized(pack,store.base.norm)
     args=tuple(a.cuda() if a is not None else None for a in args);rf=rf.cuda();r=head_forward(models[name],name,args,rf)
     if name!='R2':
      r2=models['R2'](rf,args[6]);bike=g.agent_type[t].cuda()==2;r['mode_prob'][bike]=r2['mode_prob'][bike]
     r['mode_prob'].argmax(-1)
   return targets
  for name in names:
   for _ in range(3):full(name)
   ts=[];torch.cuda.reset_peak_memory_stats()
   for _ in range(10):
    torch.cuda.synchronize();t=time.perf_counter();n=full(name);torch.cuda.synchronize();ts.append(time.perf_counter()-t)
   whole.append({'Fold':fold,'Model':name,'Windows':16,'CurrentActors':batch.num_nodes,'PastEligibleTargets':n,'MeanSeconds':float(np.mean(ts)),
    'MedianSeconds':float(np.median(ts)),'WarmupRepeats':3,'TimedRepeats':10,'PeakAllocatedBytes':torch.cuda.max_memory_allocated(),
    'Scope':'identical preloaded 16 InnerDev windows; frozen predictor + ego conversion + required context/features + head + actual routing + argmax; excludes raw I/O and SHA checks',
    'WindowIdentitiesSHA256':hashlib.sha256('|'.join(g.scene_token+'/'+g.sample_token for g in graphs).encode()).hexdigest()})
  del models,tnt,model,pieces,store;ds.clear();torch.cuda.empty_cache();print('EFFICIENCY_FOLD_COMPLETE',fold,flush=True)
 dump(ROOT/'07_efficiency/stage17_head_compute.csv',headrows);dump(ROOT/'07_efficiency/stage17_whole_compute.csv',whole)
 atomic_json(ROOT/'07_efficiency/stage17_resource_receipt.json',{'GPU':torch.cuda.get_device_name(),'Torch':torch.__version__,'DiskFreeBytes':shutil.disk_usage(ROOT).free,
  'AddedContextCacheBytes':sum(p.stat().st_size for p in (ROOT/'03_context/cache').rglob('*.npy')),
  'SelectedCheckpointBytes':{str(k):cp_path(k).stat().st_size for k in (1,2,3)},'HeadAndWholeContexts':'independent frozen HiVT local context used by TNT; separate CPU graph pipeline used by G-C',
  'NoConcurrentStage17GPUJobsDuringBenchmark':True})
if __name__=='__main__':benchmark()
