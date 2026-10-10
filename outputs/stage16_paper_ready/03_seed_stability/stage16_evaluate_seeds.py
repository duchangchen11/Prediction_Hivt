"""Evaluate seed supplements only after all24 selected heads freeze."""
from pathlib import Path
import sys,time,itertools
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'00_manifest'))
from stage16_common import *
@torch.no_grad()
def main():
 runtime();gate=read_json(ROOT/'03_seed_stability/stage16_all_frozen.json');reg=read_json(ROOT/'00_manifest/stage16_supplement_registration.json')
 assert gate['Status']=='FROZEN_ALL_COMPLETE' and len(gate['Results'])==24
 assert gate['RegistrationSHA256']==sha256(ROOT/'00_manifest/stage16_supplement_registration.json')
 assert gate['TrainingSourceSHA256']==sha256(ROOT/'03_seed_stability/stage16_train_heads.py')
 for path,h in reg['ImplementationSHA256'].items():assert sha256(PROJECT/path)==h
 assert read_json(ROOT/'01_freeze/stage16_cache_audit.json')['Status']=='PASS'
 for c in gate['Results']:
  path=ROOT/f"03_seed_stability/checkpoints/fold{c['fold']}/seed{c['initialization_seed']}/{c['model']}_best.pt"
  assert sha256(path)==c['CheckpointSHA256']
 f,oldmetrics,oldprob,oldlogits=frozen_eval();names=reg['Models'];n=len(f);dest=ROOT/'03_seed_stability/cache';dest.mkdir(exist_ok=True)
 values=np.lib.format.open_memmap(dest/'seed_metrics.npy',mode='w+',dtype=np.float64,shape=(n,2,4,3))
 frames=[];checks=[];offset=0;started=time.monotonic()
 for fold in (1,2,3):
  store=RoleStore(fold,'OuterTest');frame=f.iloc[offset:offset+len(store.ids)].reset_index(drop=True)
  fields=['scene_token','sample_token','instance_token','node_index','fold','predictor_checkpoint_sha256'];assert frame[fields].equals(store.frame[fields])
  r2=original.fresh('R2',fold);r2.load_state_dict(torch.load(original.cp_path(fold,'R2'),map_location='cpu',weights_only=False)['state_dict']);r2.eval().requires_grad_(False)
  heads={}
  for c in gate['Results']:
   if c['fold']!=fold:continue
   name=c['model'];seed=c['initialization_seed'];cls=NoGraphReranker if name.startswith('NG-') else MatchedNoGraphReranker if name=='Matched-NG-C' else None
   m=(cls(seed) if cls else SparseGraphReranker('G1',seed)).cuda();path=ROOT/f'03_seed_stability/checkpoints/fold{fold}/seed{seed}/{name}_best.pt';cp=torch.load(path,map_location='cpu',weights_only=False)
   assert cp['Fold']==fold and cp['Model']==name and cp['NormalizationSHA256']==sha256(store.normpath) and cp['PredictorCheckpointSHA256']==store.manifest['PredictorCheckpointSHA256']
   assert cp['Epoch']==c['SelectedEpoch'];m.load_state_dict(cp['state_dict'],strict=True);m.eval().requires_grad_(False);heads[(c['replicate'],name)]=m
  for pos in range(0,len(store.ids),128):
   ids=store.ids[pos:pos+128];args,fd,ad=store.batch(ids);bike=torch.as_tensor(store.types[ids]==2,device='cuda');route=r2(store.r2(ids),args[6]);idx=torch.arange(len(ids),device='cuda')
   for (rep,name),m in heads.items():
    out=m(*args);p=out['mode_prob'].clone();p[bike]=route['mode_prob'][bike];top=p.argmax(-1)
    assert torch.isfinite(p).all() and torch.allclose(p.sum(-1),torch.ones(len(ids),device='cuda'),atol=1e-6,rtol=0)
    val=torch.stack((fd[idx,top],ad[idx,top],(top==fd.argmin(-1)).float()),-1).double().cpu().numpy()
    assert np.array_equal(val[store.types[ids]==2],oldmetrics[offset+ids[store.types[ids]==2],MODELS.index('R2')][:,[0,1,3]])
    values[offset+ids,rep-1,names.index(name)]=val
  for rep in (0,1,2):
   for j,name in enumerate(names):
    seed=2022+100*(fold-1)+rep*1000
    a=oldmetrics[offset:offset+len(store.ids),MODELS.index(name)][:,[0,1,3]] if rep==0 else values[offset:offset+len(store.ids),rep-1,j]
    for group,mask in groups(store.frame).items():
     frames.append({'Fold':fold,'Group':group,'Model':name,'InitializationReplicate':rep,'InitializationSeed':seed,'FixedPredictorSeed':2022+100*(fold-1),'FixedSamplerSeed':2022+100*(fold-1),'Count':int(mask.sum()),'Scenes':int(store.frame.loc[mask,'scene_token'].nunique()),'Top1FDE':float(a[mask,0].mean()),'Top1ADE':float(a[mask,1].mean()),'HitRate':float(a[mask,2].mean()),'Origin':'frozen Stage15B' if rep==0 else 'Stage16 initialization-only supplement'})
  checks.append({'Fold':fold,'Identity':'PASS','Heads':8,'BicycleR2Exact':True,'OuterReadAfter24Freeze':True});offset+=len(store.ids)
  del store,r2,heads;torch.cuda.empty_cache();print('SEED_OUTER_EVALUATED_FOLD',fold,flush=True)
 values.flush();assert offset==n
 for rep in (0,1,2):
  for j,name in enumerate(names):
   a=oldmetrics[:,MODELS.index(name)][:,[0,1,3]] if rep==0 else values[:,rep-1,j]
   for group,mask in groups(f).items():frames.append({'Fold':0,'Group':group,'Model':name,'InitializationReplicate':rep,'InitializationSeed':'fold-specific; see fold rows','FixedPredictorSeed':'three frozen fold seeds','FixedSamplerSeed':'original per-fold ordering','Count':int(mask.sum()),'Scenes':int(f.loc[mask,'scene_token'].nunique()),'Top1FDE':float(a[mask,0].mean()),'Top1ADE':float(a[mask,1].mean()),'HitRate':float(a[mask,2].mean()),'Origin':'frozen Stage15B' if rep==0 else 'Stage16 initialization-only supplement'})
 df=pd.DataFrame(frames);dump(ROOT/'06_source_data/stage16_seed_metrics.csv',frames)
 summaries=[];contrasts=[]
 for (fold,group,name),d in df.groupby(['Fold','Group','Model'],sort=False):
  assert len(d)==3;summaries.append({'Fold':fold,'Group':group,'Model':name,'Initializations':3,'CountPerInitialization':int(d.Count.iloc[0]),'MeanTop1FDE':float(d.Top1FDE.mean()),'SDTop1FDE':float(d.Top1FDE.std(ddof=1)),'MinTop1FDE':float(d.Top1FDE.min()),'MaxTop1FDE':float(d.Top1FDE.max())})
 for (fold,group,rep),d in df.groupby(['Fold','Group','InitializationReplicate'],sort=False):
  a=d.set_index('Model')
  for left,right in [('G-C','NG-C'),('G-C','Matched-NG-C'),('NG-C','NG-A')]:
   delta=float(a.loc[left].Top1FDE-a.loc[right].Top1FDE);contrasts.append({'Fold':fold,'Group':group,'InitializationReplicate':rep,'Comparison':left+'-'+right,'DeltaTop1FDE':delta,'Direction':'Improved' if delta<0 else 'Worsened' if delta>0 else 'Equal'})
 dump(ROOT/'06_source_data/stage16_seed_summary.csv',summaries);dump(ROOT/'06_source_data/stage16_seed_contrasts.csv',contrasts)
 # Enumerate all preregistered per-fold choices; no seed is selected from them.
 combinations=[]
 for group in groups(f):
  for seeds in itertools.product(range(3),repeat=3):
   for name in names:
    a=df[(df.Group==group)&(df.Model==name)&(df.Fold!=0)]
    chosen=pd.concat([a[(a.Fold==k)&(a.InitializationReplicate==rep)] for k,rep in enumerate(seeds,1)])
    assert len(chosen)==3;combinations.append({'Group':group,'Model':name,'Fold1Replicate':seeds[0],'Fold2Replicate':seeds[1],'Fold3Replicate':seeds[2],'PooledTop1FDE':float(np.average(chosen.Top1FDE,weights=chosen.Count))})
 dump(ROOT/'06_source_data/stage16_seed_all_fold_combinations.csv',combinations)
 atomic_json(ROOT/'03_seed_stability/stage16_seed_evaluation_integrity.json',{'Status':'PASS','FoldChecks':checks,'Rows':n,'SeedMetricsSHA256':sha256(dest/'seed_metrics.npy'),'Seconds':time.monotonic()-started,'SourceAllFrozenSHA256':sha256(ROOT/'03_seed_stability/stage16_all_frozen.json'),'FurtherTrainingOrSeedSelection':False,'NoHiVTFit':True,'MetricOrder':['Top1FDE','Top1ADE','HitRate'],'CubeAxes':'actors,new initialization replicate[1,2],models[NG-A,NG-C,G-C,Matched-NG-C],metric'})
 print(df[(df.Fold==0)&(df.Group=='Overall')][['Model','InitializationReplicate','Top1FDE']].to_string(index=False))
if __name__=='__main__':main()
