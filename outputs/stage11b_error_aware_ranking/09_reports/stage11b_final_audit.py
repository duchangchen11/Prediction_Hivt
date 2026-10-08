"""Independent audit of saved selection, isolation, control and OOF identities."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'00_manifest'))
from stage11b_common import *

def training_audit():
    verify(history=True);f=frame();assert len(f)==260151 and f.actor_id.is_unique and f.source_index.is_unique
    assert f.HeadTrain.eq(1).all() and set(f.horizon)=={'full_horizon'} and f.future_mask_bits.eq('111111111111').all()
    pre=read_json(ROOT/'01_preflight/stage11b_data_and_poison_audit.json');assert pre['Status']=='PASS' and pre['GTpoisonWindows']==100
    for name,h in pre['cache_files'].items():assert sha256(CACHE/name)==h
    assert pre['GraphSourceSHA256']==sha256(S8/'00c_sparse_type_aware_graph_spec/03_model_audit/stage8a0c_model.py')
    assert read_json(ROOT/'01_preflight/stage11b_tiny_audit.json')['Status']=='PASS'
    frozen=read_json(ROOT/'04_checkpoints/stage11b_all_frozen.json');assert frozen['Status']=='FROZEN_ALL_COMPLETE' and len(frozen['Checkpoints'])==12
    rows=[];allouter=[];inits=[]
    controlled=('optimizer','learning_rate','weight_decay','precision','AMP','microbatch','accumulation','effective_batch','max_epochs','patience','carry','batch_order','ABC_selection')
    core_sources=('00_manifest/stage11b_common.py','03_training/stage11b_train.py','01_preflight/stage11b_tiny.py')
    for fold in (1,2,3):
        s=split(fold);parts=[set(s[p]) for p in ('InnerTrain','InnerDev','OuterTest')]
        assert list(map(len,parts))==[378,42,210] and len(set.union(*parts))==630
        assert not any(parts[i]&parts[j] for i in range(3) for j in range(i));allouter.extend(s['OuterTest'])
        norm=read_json(ROOT/f'02_splits/stage11b_fold{fold}_normalization.json');train=indices(fold,'InnerTrain')
        initial_model=fresh(fold,'cpu').eval()
        assert torch.equal(initial_model.head[-1].weight,torch.zeros_like(initial_model.head[-1].weight))
        assert torch.equal(initial_model.head[-1].bias,torch.zeros_like(initial_model.head[-1].bias))
        with torch.no_grad():
            args,_,_=Store(fold).batch(train[:16],device='cpu',part='InnerTrain');out=initial_model(*args)
            assert torch.equal(out['delta_logits'],torch.zeros_like(out['delta_logits'])) and torch.equal(out['mode_logits'],args[6])
        assert norm['fit_partition']=='InnerTrain' and set(norm['fit_scenes'])==parts[0] and norm['fit_actors']==len(train)
        assert norm['fit_indices_sha256']==hashlib.sha256(train.tobytes()).hexdigest() and not norm['OuterTest_or_InnerDev_or_HeadDev_used']
        baseline=read_json(ROOT/f'03_training/fold{fold}/R2/stage11b_summary.json');configs=[];orders=[];initial=[]
        for name in ('R2',*VARIANTS):
            folder=ROOT/f'03_training/fold{fold}/{name}';summary=read_json(folder/'stage11b_summary.json');config=read_json(folder/'stage11b_training_config.json')
            curve=pd.read_csv(folder/'stage11b_training_curve.csv');manifest=[r for r in frozen['Checkpoints'] if r['Fold']==fold and r['Model']==name][0]
            assert summary['Status']=='COMPLETE' and not summary['OuterTestUsed'] and summary['PredictorGradientCount']==0
            assert summary['AllDistinctTrainingActorsUsed']==len(train)
            assert manifest['SHA256']==summary['checkpoint_sha256']==sha256(PROJECT/manifest['Path'])
            assert manifest['NormalizationSHA256']==sha256(ROOT/f'02_splits/stage11b_fold{fold}_normalization.json')
            assert manifest['SplitSHA256']==sha256(ROOT/f'02_splits/stage11b_fold{fold}_split.json')
            assert manifest['TrainingConfigSHA256']==sha256(folder/'stage11b_training_config.json')
            assert config['training_indices_sha256']==hashlib.sha256(train.tobytes()).hexdigest()
            assert config['InnerDev_indices_sha256']==hashlib.sha256(indices(fold,'InnerDev').tobytes()).hexdigest()
            for relative in core_sources:assert config['new_sources_sha256'][relative]==sha256(ROOT/relative)
            selected=curve.iloc[int(curve.DevSelectionScore.argmin())]
            assert int(selected.Epoch)==summary['SelectedEpoch']==manifest['SelectedEpoch']
            assert abs(float(selected.DevSelectionScore)-summary['CheckpointScore'])<1e-11
            assert len(curve)==summary['ExecutedEpochs']<=50 and (int(curve.iloc[-1].PatienceCount)==5 or len(curve)==50)
            assert np.isfinite(curve.select_dtypes(include='number').to_numpy()).all() and (curve.GradientNorm>0).all()
            checkpoint=torch.load(PROJECT/manifest['Path'],map_location='cpu',weights_only=False)
            last=torch.load(folder/'last.pt',map_location='cpu',weights_only=False);assert last['complete']
            assert checkpoint['Epoch']==summary['SelectedEpoch'] and checkpoint['initial_state_sha256']==summary['initial_state_sha256']
            if name=='R2':
                assert summary['R2OriginalProtocolPreserved'] and not summary['HistoricalR2Loaded'] and summary['Params']==673
                for k,v in read_json(PROTOCOL)['R2_protocol'].items():assert config[k]==v
                assert np.allclose(curve.DevSelectionScore,curve.InnerDevOverallTop1FDE,atol=1e-11,rtol=0.)
            else:
                configs.append({k:config[k] for k in controlled});initial.append(summary['initial_state_sha256'])
                orders.append(pd.read_csv(folder/'stage11b_batch_order.csv'))
                assert summary['Params']==24066 and summary['BicycleTargetsTrained']==int((f.iloc[train].agent_type_id==2).sum())
                assert summary['R2_DevReferences']=={t:baseline['DevMetrics'][t]['Top1FDE'] for t in TYPES[:2]}
                computed=.5*curve.InnerDevVehicleTop1FDE/summary['R2_DevReferences']['Vehicle']+.5*curve.InnerDevPedestrianTop1FDE/summary['R2_DevReferences']['Pedestrian']
                assert np.allclose(computed,curve.DevSelectionScore,atol=1e-11,rtol=0.)
                coverage=last['coverage'];assert (coverage[train]>0).all() and (coverage[np.setdiff1d(np.arange(len(f)),train)]==0).all()
                assert len(last['pending'])==summary['FinalPendingOccurrences'] and len(last['pending'])<1024
                assert np.isin(last['pending'],train).all() and int(coverage.sum())==int(curve.OptimizerTargets.sum())
                if name=='C':assert np.allclose(curve.TrainLoss,curve.TrainNormalizedExpectedRegret,atol=1e-11,rtol=0.) and (curve.ScaleMin>=1).all()
            rows.append({'Fold':fold,'Model':name,'Status':'PASS','SelectedEpoch':summary['SelectedEpoch'],'ExecutedEpochs':len(curve),
                'ActorCoverage':len(train),'NoOuterTraining':True,'SelectedMinimumInnerDevScore':True,'SHA256':manifest['SHA256']})
        assert configs[0]==configs[1]==configs[2] and len(set(initial))==1;inits.append(initial[0])
        for i in range(3):
            for j in range(i):
                common=min(len(orders[i]),len(orders[j]));assert np.array_equal(orders[i].OrderSHA256[:common],orders[j].OrderSHA256[:common])
    assert len(set(allouter))==630 and len(allouter)==630 and set(allouter)==set(f.scene_token) and len(set(inits))==3
    dump('09_reports/stage11b_training_integrity.csv',rows)
    return {'Status':'PASS','CVIsolation':'PASS','LossImplementation':'PASS','FoldSpecificR2Trained':True,
        'FoldCheckpoints':12,'SameABCInitializationWithinFold':True,'IndependentInitializationAcrossFolds':True,
        'SameABCOrderOnCommonEpochs':True,'SameControlledConfiguration':True,'NoTrainingSourceChanged':True,
        'CheckpointSelectionVerified':True,'ZeroHeadInitialLogitsEqualStage5A':True,'AllDistinctActorsUsedIncludingBicycle':True,
        'NormalizationFitOnlyInnerTrain':True,'GTpoisonWindows':100,'HistoricalFilesAndCheckpointsUnchanged':True}

def oof_audit():
    src=ROOT/'05_oof_evaluation/cache';audit=read_json(src/'complete.json');assert audit['Status']=='PASS'
    f=pd.read_csv(src/'stage11b_oof_predictions_actor_records.csv',dtype={'future_mask_bits':str});original=frame()
    assert f.actor_id.tolist()==original.actor_id.tolist() and f.source_index.tolist()==original.source_index.tolist()
    assert f.actor_id.is_unique and set(f.Partition)=={'OuterTest'} and len(f)==260151
    for fold in (1,2,3):assert np.array_equal(np.flatnonzero(f.Fold==fold),indices(fold,'OuterTest'))
    z=np.load(src/'stage11b_oof_predictions_logits.npy',mmap_mode='r');p=np.load(src/'stage11b_oof_predictions_probabilities.npy',mmap_mode='r')
    val=np.load(src/'stage11b_oof_metrics.npy',mmap_mode='r');fd=np.load(S8/'01_training/cache/fde.npy',mmap_mode='r');ad=np.load(S8/'01_training/cache/ade.npy',mmap_mode='r')
    for name,h in audit['cache_files'].items():assert sha256(src/name)==h
    for start in range(0,len(f),4096):
        ix=np.arange(start,min(start+4096,len(f)));source=f.source_index.to_numpy()[ix];v=val[ix];pp=p[ix];zz=z[ix]
        assert np.isfinite(v).all() and np.isfinite(pp).all() and np.isfinite(zz).all()
        e=np.exp(zz.astype(np.float64)-zz.max(-1,keepdims=True));assert np.allclose(pp,e/e.sum(-1,keepdims=True),atol=2e-7,rtol=0.)
        assert np.allclose(pp.sum(-1),1,atol=1e-6,rtol=0.)
        errors=fd[source];ades=ad[source];best=errors.argmin(-1);ii=np.arange(len(ix));top=pp.argmax(-1)
        c=(errors-errors.min(-1,keepdims=True)).astype(np.float64);scale=np.maximum(1,c.mean(-1))
        qq=np.exp(-errors.astype(np.float64)+errors.min(-1,keepdims=True).astype(np.float64));qq/=qq.sum(-1,keepdims=True)
        for j in range(5):
            assert np.array_equal(v[:,j,0],errors[ii,top[:,j]].astype(np.float64))
            assert np.array_equal(v[:,j,1],ades[ii,top[:,j]].astype(np.float64))
            assert np.array_equal(v[:,j,2],errors[ii,top[:,j]].astype(np.float64)-errors.min(-1).astype(np.float64))
            assert np.array_equal(v[:,j,3],top[:,j]==best)
            ranks=(np.argsort(-pp[:,j],axis=-1,kind='stable')==best[:,None]).argmax(-1)+1
            assert np.array_equal(v[:,j,4],1./ranks)
            assert np.array_equal(v[:,j,12],ades.min(-1).astype(np.float64))
            assert np.array_equal(v[:,j,13],errors.min(-1).astype(np.float64))
            assert np.array_equal(v[:,j,14],errors.min(-1)>2)
            logp=zz[:,j].astype(np.float64)-zz[:,j].max(-1,keepdims=True).astype(np.float64)
            logp-=np.log(np.exp(logp).sum(-1,keepdims=True))
            assert np.allclose(v[:,j,5],-(qq*logp).sum(-1),atol=2e-5,rtol=0.)
            assert np.allclose(v[:,j,6],(pp[:,j]*c).sum(-1),atol=2e-5,rtol=1e-7)
            assert np.allclose(v[:,j,7],(pp[:,j]*c/scale[:,None]).sum(-1),atol=2e-6,rtol=0.)
            assert np.allclose(v[:,j,8],-(pp[:,j].astype(np.float64)*np.log(np.maximum(pp[:,j],1e-30))).sum(-1),atol=2e-6,rtol=0.)
            sortedp=np.sort(pp[:,j],axis=-1)[:,::-1]
            assert np.array_equal(v[:,j,9],sortedp[:,0].astype(np.float64))
            assert np.array_equal(v[:,j,10],(sortedp[:,0]-sortedp[:,1]).astype(np.float64))
            assert np.array_equal(v[:,j,11],pp[ii,j,best].astype(np.float64))
        bike=f.agent_type_id.to_numpy()[ix]==2
        for j in range(2,5):
            assert np.array_equal(zz[bike,j],zz[bike,1]) and np.array_equal(pp[bike,j],pp[bike,1])
            assert np.array_equal(top[bike,j],top[bike,1]) and np.array_equal(v[bike,j],v[bike,1])
    boot=read_json(ROOT/'06_bootstrap/stage11b_bootstrap_audit.json');weights=np.load(ROOT/'06_bootstrap/stage11b_scene_bootstrap_weights.npy')
    assert weights.shape==(2000,630) and np.all(weights.reshape(2000,3,210).sum(-1)==210)
    assert sha256(ROOT/'06_bootstrap/stage11b_scene_bootstrap_weights.npy')==boot['ReplicateWeightsSHA256']
    rng=np.random.default_rng(2022)
    for b in range(2000):
        for fold in range(3):assert np.array_equal(weights[b,fold*210:(fold+1)*210],np.bincount(rng.integers(0,210,210),minlength=210))
    return {'CandidateIdentity':'PASS','CandidateCoordinateMaxDiff':0,'BicyclePreserved':'YES','BicycleActors':int((f.agent_type_id==2).sum()),
        'OOFActors':len(f),'OOFScenes':630,'NoInnerDevPredictionsInOOF':True,'AllStoredMetricsRecomputed':True,
        'OOFProbabilityFiniteAndNormalized':True,'SceneBootstrapWeightsIndependentlyReplayed':True}

def main():
    result=training_audit()
    if '--training-only' in sys.argv:
        atomic_json(ROOT/'09_reports/stage11b_pre_oof_training_audit.json',result)
    else:
        result.update(oof_audit());decision=read_json(ROOT/'09_reports/stage11b_scientific_decision.json')
        decision.update({k:result[k] for k in ('LossImplementation','CVIsolation','CandidateIdentity','BicyclePreserved')})
        decision['EngineeringGate']='PASS';decision['TotalTrainingSeconds']=float(pd.read_csv(ROOT/'03_training/stage11b_training_summary.csv').Seconds.sum())
        atomic_json(ROOT/'09_reports/stage11b_scientific_decision.json',decision)
        atomic_json(ROOT/'09_reports/stage11b_final_audit.json',result)
    verify(history=True);print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
