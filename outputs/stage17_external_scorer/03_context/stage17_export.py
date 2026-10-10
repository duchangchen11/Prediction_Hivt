"""Read-only decoder pre-hook captures observed fused local context; no forward edits."""
from pathlib import Path
import sys,time,argparse
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'00_manifest'))
from stage17_common import *
from torch_geometric.data import Batch
def export(fold,role):
 runtime();protect(fitting=role!='OuterTest');verify_registration()
 if role=='OuterTest':assert read_json(ROOT/'05_training/stage17_all_frozen.json')['Status']=='FROZEN_ALL_COMPLETE'
 spec=read_json(pc.PROTOCOL)['folds'][fold-1]
 ctx=pc.FoldContext(fold,spec['seed'],OLD/spec['files']['InnerTrain']['path'],OLD/spec['files']['InnerDev']['path'],OLD/f'04_predictor_checkpoints/fold{fold}',install=False,stage='candidate')
 ds=pc.FoldDataset(ctx,role);old=RoleStore(fold,role);model,rec=predictor(fold)
 folder=ROOT/f'03_context/cache/fold{fold}/{role}';folder.mkdir(parents=True,exist_ok=True)
 if (folder/'manifest.json').exists():
  done=read_json(folder/'manifest.json');assert done['Status']=='PASS';return
 assert not (folder/'context.npy').exists(),'Partial export retained; explicit diagnosis required'
 n=len(old.ids);arrays={k:np.lib.format.open_memmap(folder/(k+'.npy'),mode='w+',dtype=np.float32,shape=s) for k,s in
  [('context',(n,1,64)),('trajectory',(n,6,24)),('gt',(n,24))]}
 got=[];hook=model.decoder.register_forward_pre_hook(lambda mod,args:got.append(args[0].detach().clone()))
 pos=batches=integrity=0;maxd={'raw_prediction':0.,'mode_logits':0.,'mode_prob':0.};cachemax=0.;start=time.monotonic();torch.cuda.reset_peak_memory_stats()
 contexts={r['scene_token']:r for r in old.manifest['Contexts']}
 for scene,ids in ds.scene_indices.items():
  savedpath=OLD/contexts[scene]['path'];assert sha256(savedpath)==contexts[scene]['sha256']
  frozen=torch.load(savedpath,map_location='cpu',weights_only=False);assert len(frozen)==len(ids)
  for off in range(0,len(ids),16):
   graphs=[ds[int(i)] for i in ids[off:off+16]];batch=Batch.from_data_list(graphs).cuda()
   observed=pc.model_input(batch);got.clear()
   with torch.no_grad():out=model(observed);emb=got[-1];pred=model.ego_predictions(out,observed)
   # First 18 batches per Train/Dev role/fold =108 paired batches before fitting.
   if batches<18:
    hook.remove()
    poisoned=batch.clone();poisoned.positions[:,5:]=12345.;poisoned.padding_mask[:,5:]=True
    if 'y' in poisoned:poisoned.y=-poisoned.y
    cleaned=pc.model_input(poisoned)
    assert set(cleaned.keys())==set(observed.keys())
    for key in observed.keys():
     if torch.is_tensor(observed[key]):assert torch.equal(cleaned[key],observed[key]),'Future changed observable input'
    with torch.no_grad():normal=model(cleaned)
    for k in maxd:
     diff=float((normal[k]-out[k]).abs().max());maxd[k]=max(maxd[k],diff);assert torch.equal(normal[k],out[k]),'Instrumentation output changed'
    hook=model.decoder.register_forward_pre_hook(lambda mod,args:got.append(args[0].detach().clone()));integrity+=1
   ptr=batch.ptr.tolist()
   for j,g in enumerate(graphs):
    lo,hi=ptr[j:j+2];w=frozen[off+j];obs=w['observable'];labels=w['labels']
    assert obs['scene_token']==scene and obs['sample_token']==g.sample_token and tuple(g.instance_tokens)==tuple(obs['instance_tokens'])
    assert torch.equal(g.history_times.cpu(),obs['history_times']) and torch.equal(g.future_times.cpu(),obs['future_sample_times_metadata'])
    assert torch.equal(g.positions[:,:5].cpu(),obs['history']) and torch.equal(g.padding_mask[:,:5].cpu(),obs['history_padding'])
    actual=pred[lo:hi].cpu();z=out['mode_logits'][lo:hi].cpu();prob=out['mode_prob'][lo:hi].cpu()
    # Same model, same 16-window batching and identity order as the original exporter.
    assert torch.equal(actual,obs['ego_prediction']),f'Cached candidate changed fold{fold} {scene}'
    assert torch.equal(z,obs['mode_logits']) and torch.equal(prob,obs['mode_prob'])
    full=g.target_mask & g.future_mask.all(-1);t=torch.where(full)[0];nt=len(t)
    assert torch.equal(full.cpu(),labels['full_horizon_mask']) and w['target_cache_start']==pos and w['target_count']==nt
    rows=old.frame.iloc[pos:pos+nt];assert rows.node_index.tolist()==t.tolist() and rows.instance_token.tolist()==[g.instance_tokens[k] for k in t]
    rot=out['rotation'][lo:hi].cpu()[t];center=obs['history'][t,4,None,None]
    # Meter-preserving rigid transform of exact cached candidates to actor t0 frame.
    traj=torch.matmul(obs['ego_prediction'][t]-center,rot[:,None])
    gt=torch.matmul(labels['future_xy'][t]-center[:,0],rot)
    raw=out['raw_prediction'][: ,lo:hi,:,:2].permute(1,0,2,3).cpu()[t]
    d=float((traj-raw).abs().max()) if nt else 0.;cachemax=max(cachemax,d)
    assert torch.allclose(traj,raw,atol=5e-5,rtol=2e-6)
    if nt:
     fd=(traj-gt[:,None]).norm(dim=-1)[:,:,-1];assert torch.allclose(fd,torch.from_numpy(np.array(old.fde[pos:pos+nt],copy=True)),atol=5e-5,rtol=2e-6)
    sl=slice(pos,pos+nt);arrays['context'][sl]=emb[lo:hi][t.cuda()].cpu().numpy()[:,None]
    arrays['trajectory'][sl]=traj.reshape(nt,6,24).numpy();arrays['gt'][sl]=gt.reshape(nt,24).numpy();pos+=nt
   batches+=1
  if len(ctx.loaded_shards)%25==0:print('EXPORT',fold,role,'scenes',len(ctx.loaded_shards),'targets',pos,flush=True)
 assert pos==n;hook.remove()
 for a in arrays.values():a.flush();assert np.isfinite(a).all()
 assert state_digest(model.state_dict())==rec['state_sha256'] and sha256(OLD/rec['path'])==rec['sha256']
 record={'Status':'PASS','Fold':fold,'Role':role,'Targets':n,'Scenes':len(ds.scene_indices),'Batches':batches,'PairedIntegrityBatches':integrity,
 'InstrumentationMaxDiff':maxd,'CachedCandidatesBitwiseEqual':True,'CachedLogitsBitwiseEqual':True,'CachedProbabilitiesBitwiseEqual':True,
 'ActorFrameTransformMaxDiff':cachemax,'IdentityCSV_SHA256':sha256(old.folder/'targets.csv'),'OriginalCacheManifestSHA256':sha256(old.folder/'manifest.json'),
 'PredictorCheckpointSHA256':rec['sha256'],'FrozenRegistrationSHA256':sha256(REG),'Context':'decoder input local: history+lanes+local actors+type embedding,64D; detached',
 'FuturePerturbationInputAndOutputBitwiseEqual':True,
 'LabelUse':'separate GT array, never predictor/scorer.forward','Seconds':time.monotonic()-start,'PeakGPUMemoryBytes':torch.cuda.max_memory_allocated(),
 'Arrays':{k:{'shape':list(a.shape),'sha256':sha256(folder/(k+'.npy')),'bytes':(folder/(k+'.npy')).stat().st_size} for k,a in arrays.items()}}
 atomic_json(folder/'manifest.json',record);atomic_json(ROOT/f'03_context/stage17_fold{fold}_{role}.json',record)
 print('EXPORT_PASS',fold,role,n,record['Seconds'],flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--fold',type=int,required=True);p.add_argument('--role',choices=['InnerTrain','InnerDev','OuterTest'],required=True);a=p.parse_args();export(a.fold,a.role)
