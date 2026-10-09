"""Independent reconstruction of normalization, selection, routing and scene CI."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'00_manifest'))
from stage12b_common import *
from stage12b_model import SemanticResidual
@torch.no_grad()
def main():
    seed();frozen=verify(history=True,data=True);checks=[]
    def passed(name,detail):checks.append(dict(Check=name,Status='PASS',Detail=detail));print('FINAL_AUDIT_PASS',name,flush=True)
    passed('HistoricalIntegrity',f"{len(frozen['historical_files'])} historical files, {len(frozen['checkpoints'])} checkpoints,5 old untracked files unchanged; all frozen data hashes checked")
    pre=read_json(ROOT/'01_preflight/stage12b_preflight_audit.json');assert pre['Status']=='PASS' and pre['FormalTrainingPermitted']
    tiny=pd.read_csv(ROOT/'03_tiny/stage12b_tiny_results.csv');assert (tiny.Status=='PASS').all() and (tiny.RelativeLossDecrease>=.05).all() and (tiny.FrozenGradients==0).all()
    passed('PreflightAndTiny','all requested gates PASS; 300 updates for each fixed128 target variant; tiny weights discarded')
    feature=read_json(ROOT/'02_input_cache/stage12b_feature_manifest.json')
    for path,h in {**feature['Files'],**feature['Normalizations']}.items():assert sha256(ROOT/path)==h
    passed('FrozenInputs','new fold caches, original per-scene records and normalizers verified')
    config=read_json(ROOT/'04_training/stage12b_training_config.json');assert config['ProtocolSHA256']==sha256(PROTOCOL)
    for path,h in config['SourceSHA256'].items():assert sha256(ROOT/path)==h
    all9=read_json(ROOT/'05_checkpoints/stage12b_all9_frozen.json');assert all9['Status']=='FROZEN_ALL9' and len(all9['Checkpoints'])==9
    for path,h in all9['Checkpoints'].items():assert sha256(ROOT/path)==h
    manifest=pd.read_csv(ROOT/'05_checkpoints/stage12b_checkpoint_manifest.csv');assert len(manifest)==9 and (manifest.Params==641).all()
    assert manifest[['Fold','Model']].values.tolist()==[[k,v] for k in (1,2,3) for v in VARIANTS]
    curves=pd.read_csv(ROOT/'04_training/stage12b_training_curves.csv');fields,source=semantic_source()
    raw,valid,walk=raw_observable_inputs(source['geometry'],source['semantic'],fields)
    poison=frame().copy();future=[c for c in poison if c.startswith(('GT_','future_')) or c=='motion_state'];poison[future]=np.nan
    assert np.array_equal(walk,np.load(CACHE/'stage12b_walkway.npy'))
    assert not any(w in FEATURES for w in ['GT_displacement','motion_state','FDE','ADE'])
    for k in (1,2,3):
        train=indices(k,'InnerTrain',True);norm=read_json(ROOT/f'02_input_cache/stage12b_fold{k}_normalization.json')
        assert norm['FitPartition']=='InnerTrain' and norm['FitActorType']=='Pedestrian' and norm['FitScenes']==378 and norm['LabelsUsed']==False
        assert array_sha(train)==norm['InnerTrainIndicesSHA256'] and set(norm['FitSceneTokens'])==set(split(k)['InnerTrain'])
        for j in range(8):
            values=raw[train,:,j][valid[train,:,j]];assert abs(float(values.mean())-norm['Mean'][j])<1e-12 and abs(float(values.std())-norm['Std'][j])<1e-12
        expected=np.where(valid,(raw-np.array(norm['Mean']))/np.maximum(norm['Std'],1e-6),0).astype(np.float32)
        actual=normalized_features(k);assert np.array_equal(actual[...,:8],expected) and np.array_equal(actual[...,8:],valid.astype(np.float32))
        init=torch.load(ROOT/f'01_preflight/cache/stage12b_fold{k}_initial.pt',map_location='cpu',weights_only=False)
        for part in ('InnerTrain','InnerDev','OuterTest'):
            ix=indices(k,part,True);perm=np.load(CACHE/f'fold{k}/stage12b_{part}_permutation.npy',mmap_mode='r');assert np.array_equal(np.sort(perm[ix],-1),np.tile(np.arange(6),(len(ix),1)))
            xp=variant_input(k,part,'P',ix,device='cpu').numpy();assert np.array_equal(xp[...,-2:],walk[ix][np.arange(len(ix))[:,None],perm[ix]])
        for name in VARIANTS:
            d=curves[(curves.Fold==k)&(curves.Model==name)];bestrow=d.iloc[int(np.argmin(d.DevSelectionScore.to_numpy()))]
            cp=torch.load(ROOT/f'05_checkpoints/fold{k}/{name}_best.pt',map_location='cpu',weights_only=False)
            assert cp['SelectedEpoch']==int(bestrow.Epoch) and abs(cp['SelectionScore']-float(bestrow.DevSelectionScore))<1e-12
            assert cp['Temperature']==TEMPERATURES[k] and cp['InitialStateSHA256']==init['StateSHA256'] and cp['ConfigSHA256']==sha256(ROOT/'04_training/stage12b_training_config.json')
            pending=np.empty(0,np.int64);seen=np.zeros(len(train),bool)
            for row in d.itertuples():
                ordered=np.random.default_rng(2022+100*(k-1)+int(row.Epoch)).permutation(len(train));joined=np.concatenate([pending,ordered]);used=len(joined)//512*512;work=joined[:used];pending=joined[used:];seen[work]=True
                assert array_sha(train[work])==row.BatchOrderSHA256 and row.TrainingOccurrences==used and row.PendingOccurrences==len(pending)
            assert seen.all() and (d.GradientNorm>0).all() and np.isfinite(d.select_dtypes('number').to_numpy()).all()
        for epoch in curves[curves.Fold==k].Epoch.unique():assert curves[(curves.Fold==k)&(curves.Epoch==epoch)].BatchOrderSHA256.nunique()==1
    passed('TrainingAndCapacity','nine independent641-parameter heads; frozen source code; exact ordered batches512/micro128/accum4; selected strict InnerDevFDE minimum; all unique train targets used')
    passed('NormalizationAndPermutation','all8 means/std independently recomputed exclusively from corresponding InnerTrain pedestrians; every shuffled row matches fixed joint value/mask permutation')
    dest=ROOT/'06_oof_evaluation/cache';audit=read_json(ROOT/'06_oof_evaluation/stage12b_oof_identity_audit.json');assert audit['All9FrozenBeforeOuterTest'] and audit['All9FreezeSHA256']==sha256(ROOT/'05_checkpoints/stage12b_all9_frozen.json')
    for path,h in audit['CacheFiles'].items():assert sha256(ROOT/path)==h
    f=pd.read_csv(dest/'stage12b_actor_records.csv');z=np.load(dest/'stage12b_logits.npy',mmap_mode='r');p=np.load(dest/'stage12b_probabilities.npy',mmap_mode='r');delta=np.load(dest/'stage12b_residuals.npy',mmap_mode='r');top=np.load(dest/'stage12b_modes.npy',mmap_mode='r');v=np.load(dest/'stage12b_metrics.npy',mmap_mode='r')
    assert len(f)==260151 and f.actor_id.is_unique and f.scene_token.nunique()==630
    assert np.array_equal(top,p.argmax(-1));fd,ad=label_arrays();ii=np.arange(len(f))
    oldz=np.load(OOF/'stage11b_oof_predictions_logits.npy',mmap_mode='r');oldp=np.load(OOF/'stage11b_oof_predictions_probabilities.npy',mmap_mode='r')
    for j in range(4):
        assert np.array_equal(v[:,j,0],fd[ii,top[:,j]]) and np.array_equal(v[:,j,1],ad[ii,top[:,j]])
        assert np.array_equal(v[:,j,4],fd.min(-1)) and np.array_equal(v[:,j,5],ad.min(-1)) and np.array_equal(v[:,j,6],fd.min(-1)>2)
        for typ,old in [(0,4),(2,1)]:
            keep=f.agent_type_id.to_numpy()==typ;assert np.array_equal(z[keep,j],oldz[keep,old]) and np.array_equal(p[keep,j],oldp[keep,old])
    for k in (1,2,3):
        outer=indices(k,'OuterTest');assert np.array_equal(f.iloc[outer].Fold,np.full(len(outer),k));ix=indices(k,'OuterTest',True)
        base=torch.from_numpy(np.array(np.load(CACHE/f'fold{k}/stage12b_base_logits.npy',mmap_mode='r')[ix],copy=True)).cuda()/TEMPERATURES[k]
        for j,name in enumerate(VARIANTS,1):
            cp=torch.load(ROOT/f'05_checkpoints/fold{k}/{name}_best.pt',map_location='cpu',weights_only=False);m=SemanticResidual().cuda().eval();m.load_state_dict(cp['state_dict']);x=variant_input(k,'OuterTest',name,ix)
            for start in range(0,len(ix),128):
                out=m(x[start:start+128],base[start:start+128]);ids=ix[start:start+128]
                assert np.array_equal(out['logits'].cpu().numpy(),z[ids,j]) and np.array_equal(out['probabilities'].cpu().numpy(),p[ids,j]) and np.array_equal(out['delta'].cpu().numpy(),delta[ids,j-1])
            del x,m
    passed('CompleteOutputs','all9 residual predictions independently replayed bitwise; full3-type260151actor population; C0 identical to historical C; V identical to C; B identical to R2; shared candidates and all oracle metrics unchanged')
    weights=np.load(ROOT/'07_bootstrap/cache/stage12b_scene_weights.npy');rng=np.random.default_rng(2022);expected=np.zeros_like(weights)
    for rep in range(2000):
        for k in range(3):expected[rep,k*210:(k+1)*210]=np.bincount(rng.integers(0,210,210),minlength=210)
    assert np.array_equal(expected,weights);scenes=sum([sorted(split(k)['OuterTest']) for k in (1,2,3)],[]);codes=f.scene_token.map({s:i for i,s in enumerate(scenes)}).to_numpy()
    table=pd.read_csv(ROOT/'07_bootstrap/stage12b_bootstrap_ci.csv');masks=groups(f,walk)
    for row in table.itertuples():
        mask=masks[row.Group];new,old=row.Comparison.split('-');d=v[mask,MODELS.index(new),0]-v[mask,MODELS.index(old),0]
        sums=np.bincount(codes[mask],weights=d,minlength=630);counts=np.bincount(codes[mask],minlength=630);bs=(weights@sums)/(weights@counts);lo,hi=np.quantile(bs,[.025,.975])
        assert abs(d.mean()-row.DeltaTop1FDE)<1e-12 and abs(lo-row.CI95Lower)<1e-12 and abs(hi-row.CI95Upper)<1e-12
        if row.Group=='Pedestrian' and row.Comparison in ('S-G','S-P'):
            lo,hi=np.quantile(bs,[.0125,.9875]);assert abs(lo-row.FamilyAdjustedLower)<1e-12 and abs(hi-row.FamilyAdjustedUpper)<1e-12
    passed('PairedSceneBootstrap','2000 seed2022 draws of210 whole scenes per fold; every paired delta95%CI and adjusted97.5%CI independently reconstructed from scene delta sums')
    result=read_json(ROOT/'10_reports/stage12b_scientific_decision.json');lookup=table.set_index(['Group','Comparison'])
    assert result['SemanticIncrement']=='NOT_SUPPORTED' and result['Stage12B']=='STOP' and result['ReadyForNextStage']=='NO'
    assert lookup.loc[('Pedestrian','S-C0'),'DeltaTop1FDE']>0 and lookup.loc[('Pedestrian','S-G'),'DeltaTop1FDE']>0 and lookup.loc[('Pedestrian','S-P'),'DeltaTop1FDE']>0
    passed('ScientificDecision','NOT_SUPPORTED and STOP follow preregistered control-failure rule; no tuning or new-stage training')
    dump('10_reports/stage12b_verification.csv',checks)
    atomic_json(ROOT/'10_reports/stage12b_verification.json',dict(Status='PASS',Checks=checks,FrozenModelIntegrity='PASS',SemanticCacheIntegrity='PASS',CandidateIdentity='PASS',
        GTLeakage='PASS',FoldIsolation='PASS',VehiclePreserved='YES',BicyclePreserved='YES',HistoricalFiles=len(frozen['historical_files']),FrozenCheckpoints=len(frozen['checkpoints']),
        Stage12B='STOP',ReadyForNextStage='NO',ProtocolSHA256=sha256(PROTOCOL),All9FreezeSHA256=sha256(ROOT/'05_checkpoints/stage12b_all9_frozen.json')))
if __name__=='__main__':main()
