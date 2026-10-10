"""Supplementary evaluation and paired descriptive bootstrap, after global freeze."""
from pathlib import Path
import sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'00_manifest'))
from stage17_common import *
MODELS=('R0','R2','NG-A','NG-C','G-A','G-C','Matched-NG-C','Adapted TNT Scoring')
FIELDS=('Top1FDE','Top1ADE','HitRate','minFDE6')
def main():
 runtime();protect();verify_registration();gate=read_json(ROOT/'05_training/stage17_all_frozen.json')
 assert gate['Status']=='FROZEN_ALL_COMPLETE'
 for i,s in enumerate(gate['Checkpoints'],1):assert sha256(cp_path(i))==s['CheckpointSHA256']
 folder=ROOT/'06_evaluation/cache';assert not (folder/'complete.json').exists(),'Frozen evaluation must not be overwritten'
 historical=OLD/'08_evaluation/cache';frame=pd.read_csv(historical/'actor_records.csv');old=np.load(historical/'metrics.npy',mmap_mode='r')
 originalp=np.load(historical/'probabilities.npy',mmap_mode='r');n=len(frame);assert n==260151 and frame.scene_token.nunique()==630
 vals=np.lib.format.open_memmap(folder/'metrics.npy',mode='w+',dtype=np.float64,shape=(n,8,4))
 ps=np.lib.format.open_memmap(folder/'probabilities.npy',mode='w+',dtype=np.float32,shape=(n,6))
 raw=np.lib.format.open_memmap(folder/'raw_scores.npy',mode='w+',dtype=np.float32,shape=(n,6))
 secondary=np.lib.format.open_memmap(folder/'r2_bicycle_sensitivity.npy',mode='w+',dtype=np.float64,shape=(n,4))
 vals[:,:7]=old[:,:,[0,1,3,13]];offset=0
 for fold in (1,2,3):
  st=TNTStore(fold,'OuterTest');m=scorer(2022+100*(fold-1));saved=torch.load(cp_path(fold),map_location='cpu',weights_only=False)
  assert saved['Fold']==fold and saved['PredictorCheckpointSHA256']==st.base.manifest['PredictorCheckpointSHA256']
  m.load_state_dict(saved['state_dict'],strict=True);m.eval().requires_grad_(False)
  for field in ('source_index','fold','scene_token','sample_token','instance_token','node_index','agent_type','motion_state'):
   assert np.array_equal(st.frame[field].to_numpy(),frame.iloc[offset:offset+len(st.ids)][field].to_numpy()),field
  with torch.no_grad():
   for s in range(0,len(st.ids),1024):
    ix=st.ids[s:s+1024];x,tr=st.inputs(ix);p=m(x,tr);z=m.score_mlp(torch.cat((x.repeat(1,6,1),tr),dim=2)).squeeze(-1)
    assert torch.equal(z.softmax(-1),p) and torch.isfinite(p).all()
    assert torch.allclose(p.sum(-1),torch.ones(len(ix),device='cuda'),atol=1e-6,rtol=0)
    p=p.cpu().numpy();mode=p.argmax(-1);fd=np.array(st.base.fde[ix],copy=True);ad=np.array(st.base.ade[ix],copy=True)
    v=np.stack((fd[np.arange(len(ix)),mode],ad[np.arange(len(ix)),mode],(mode==fd.argmin(-1)).astype(np.float32),fd.min(-1)),axis=-1).astype(np.float64)
    rows=offset+ix;vals[rows,7]=v;ps[rows]=p;raw[rows]=z.cpu().numpy()
    assert np.array_equal(v[:,3],vals[rows,0,3]),'Candidate oracle changed'
    sensitivity=v.copy();bike=st.types[ix]==2;sensitivity[bike]=vals[rows[bike],1];secondary[rows]=sensitivity
  offset+=len(st.ids);print('TNT_OUTER_FOLD_COMPLETE',fold,offset,flush=True);del m,st;torch.cuda.empty_cache()
 assert offset==n
 for a in (vals,ps,raw,secondary):a.flush()
 rows=[]
 for fold in (1,2,3,'Pooled'):
  for group,keep in groups(frame).items():
   if fold!='Pooled':keep=keep&(frame.fold.to_numpy()==fold)
   assert keep.any()
   for a,name in enumerate(MODELS):rows.append({'Fold':fold,'Group':group,'Model':name,'Count':int(keep.sum()),'Scenes':int(frame.loc[keep,'scene_token'].nunique()),
    **dict(zip(FIELDS,vals[keep,a].mean(0))),'Protocol':'nuScenes custom internal scene-isolated CV; supplementary external baseline',
    'FDE_ADE_Unit':'m','HitRateDefinition':'argmax score equals earliest argmin endpoint FDE candidate; ties preserve original mode order'})
 dump(ROOT/'stage17_external_comparison.csv',rows)
 sr=[]
 for group,k in groups(frame).items():sr.append({'Group':group,'Model':'Adapted TNT + frozen R2 Bicycle route (sensitivity only)','Count':int(k.sum()),**dict(zip(FIELDS,secondary[k].mean(0)))})
 dump(ROOT/'06_evaluation/stage17_bicycle_route_sensitivity.csv',sr)
 boot(frame,vals)
 atomic_json(folder/'complete.json',{'Status':'PASS','Models':MODELS,'Fields':FIELDS,'ActorWindows':n,'Scenes':630,'IdentitySource':str((historical/'actor_records.csv').relative_to(PROJECT)),
  'IdentitySHA256':sha256(historical/'actor_records.csv'),'CandidateOracleBitwiseEqual':True,'ExistingSevenMetricsBitwiseEqual':np.array_equal(vals[:,:7],old[:,:,[0,1,3,13]]),
  'AllThreeFrozenBeforeOuter':True,'GlobalFreezeSHA256':sha256(ROOT/'05_training/stage17_all_frozen.json'),'SupplementaryOutsideFamily3':True,
  'UTC':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
  'SHA256':{p.name:sha256(p) for p in folder.glob('*.npy')}})
 print('STAGE17_EXTERNAL_EVALUATION_COMPLETE',flush=True)
def boot(frame,values):
 scene_order=[s for f in read_json(pc.PROTOCOL)['folds'] for s in sorted(f['parts']['OuterTest'])];assert len(set(scene_order))==630
 code=frame.scene_token.map({s:i for i,s in enumerate(scene_order)}).to_numpy();weights=np.load(OLD/'09_statistics/cache/bootstrap_scene_weights.npy',mmap_mode='r')
 rng=np.random.default_rng(2022);check=np.zeros((2000,630),np.int32)
 for b in range(2000):
  for k in range(3):check[b,k*210:(k+1)*210]=np.bincount(rng.integers(0,210,210),minlength=210)
 assert np.array_equal(weights,check)
 rows=[];foldrows=[]
 for group,keep in groups(frame).items():
  den=np.bincount(code[keep],minlength=630).astype(np.float64);bd=weights@den;assert (bd>0).all()
  for metric in FIELDS:
   j=FIELDS.index(metric)
   for name in ('G-C','NG-C'):
    a=MODELS.index(name);d=values[:,a,j]-values[:,7,j];sums=np.bincount(code[keep],weights=d[keep],minlength=630)
    draw=weights@sums/bd;ci=np.quantile(draw,[.025,.975]);row={'Group':group,'Comparison':name+' - Adapted TNT Scoring','Metric':metric,'Count':int(keep.sum()),
     'Scenes':int((den>0).sum()),'Delta':float(d[keep].mean()),'CI95Lower':float(ci[0]),'CI95Upper':float(ci[1]),'Replicates':2000,'Seed':2022,
     'CIType':'95% descriptive paired scene-cluster; stratified fold; supplementary; no confirmatory family expansion','Unit':'proportion' if metric=='HitRate' else 'm'}
    for fold in (1,2,3):
     k=keep&(frame.fold.to_numpy()==fold);point=float(d[k].mean());row[f'Fold{fold}Delta']=point;sl=slice((fold-1)*210,fold*210)
     fc=np.quantile(weights[:,sl]@sums[sl]/(weights[:,sl]@den[sl]),[.025,.975])
     foldrows.append({'Fold':fold,'Group':group,'Comparison':row['Comparison'],'Metric':metric,'Count':int(k.sum()),'Delta':point,'CI95Lower':float(fc[0]),'CI95Upper':float(fc[1])})
    row['NegativeFolds']=sum(row[f'Fold{k}Delta']<0 for k in (1,2,3));rows.append(row)
 dump(ROOT/'stage17_bootstrap_comparison.csv',rows);dump(ROOT/'06_evaluation/stage17_fold_bootstrap.csv',foldrows)
 atomic_json(ROOT/'06_evaluation/stage17_bootstrap_receipt.json',{'Status':'PASS','PairedSceneIdentity':True,'SceneOrder':scene_order,'WeightsSHA256':sha256(OLD/'09_statistics/cache/bootstrap_scene_weights.npy'),
  'HistoricalFamily3Unchanged':True,'SupplementaryOnly':True,'MetricSelection':'frozen before new external Outer results','Uncertainty':'conditional on the three fixed predictors and three fitted scorers; not across predictor seeds'})
if __name__=='__main__':main()
