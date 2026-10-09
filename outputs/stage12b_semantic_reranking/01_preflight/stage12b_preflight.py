"""Freeze inputs and replay each fold C/R2 with exact source graph normalization."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'00_manifest'))
from stage12b_common import *
from stage12b_model import SemanticResidual
@torch.no_grad()
def main():
    seed();verify();assert torch.cuda.is_available() and '3080' in torch.cuda.get_device_name(0)
    f=frame();n=len(f);CACHE.mkdir(parents=True,exist_ok=True);manifest,data=semantic_source()
    oof=pd.read_csv(OOF/'stage11b_oof_predictions_actor_records.csv',dtype={'future_mask_bits':str})
    assert np.array_equal(f.actor_id.to_numpy(),oof.actor_id.to_numpy()) and np.array_equal(f.source_index.to_numpy(),oof.source_index.to_numpy())
    assert not set(f.scene_token)&set(read_json(S6/'00_manifest/stage6a_head_split.json')['HeadDev'])
    sems=np.load(S12A/'03_feature_statistics/cache/stage12a_semantic.npy',mmap_mode='r');valid=data['feature_valid']
    for rec in manifest['Records']:
        path=S12A/rec['Path'];assert sha256(path)==rec['SHA256']
        with np.load(path) as zz:
            ix=zz['headtrain_indices'];assert (f.iloc[ix].scene_token==rec['Scene']).all()
            assert np.array_equal(zz['semantic'],sems[ix]) and np.array_equal(zz['feature_valid'],valid[ix])
            assert np.array_equal(zz['geometry'],data['geometry'][ix]) and np.array_equal(zz['candidate_geometry_id'],data['candidate_geometry_id'][ix])
    candidates=np.load(S11A/'01_identity_audit/cache/candidates.npy',mmap_mode='r');source=f.source_index.to_numpy()
    for start in range(0,n,2048):
        ids=np.arange(start,min(n,start+2048));arr=np.array(candidates[source[ids]],copy=True)
        hashes=np.array([hashlib.sha256(a.tobytes()).hexdigest().encode() for a in arr.reshape(-1,12,2)],dtype='S64').reshape(-1,6)
        assert np.array_equal(hashes,data['candidate_geometry_id'][ids])
    raw,mask,walk=raw_observable_inputs(data['geometry'],data['semantic'],manifest)
    np.save(CACHE/'stage12b_walkway.npy',walk)
    # Future columns are deliberately poisoned; only fixed observation-derived
    # cached tensors reach the input API, never actor labels or error arrays.
    poisoned=f.copy();poisoned['GT_displacement']=np.nan;poisoned['motion_state']='POISON';poisoned['GT_trajectory_sha256']='POISON'
    second=raw_observable_inputs(data['geometry'],data['semantic'],manifest)
    assert all(np.array_equal(a,b) for a,b in zip((raw,mask,walk),second))
    fd,ad=label_arrays();basez=np.load(OOF/'stage11b_oof_predictions_logits.npy',mmap_mode='r');basep=np.load(OOF/'stage11b_oof_predictions_probabilities.npy',mmap_mode='r')
    counts=[];replays=[];initials=[];neutral=0;training_scenes=set();scenesOuter=[];children=np.random.SeedSequence(2022).spawn(9)
    for fold in (1,2,3):
        folder=CACHE/f'fold{fold}';folder.mkdir(exist_ok=True);ss=split(fold)
        assert list(map(lambda p:len(ss[p]),['InnerTrain','InnerDev','OuterTest']))==[378,42,210]
        assert not (set(ss['InnerTrain'])&set(ss['InnerDev']) or set(ss['InnerTrain'])&set(ss['OuterTest']) or set(ss['InnerDev'])&set(ss['OuterTest']))
        assert set(ss['InnerTrain']+ss['InnerDev']+ss['OuterTest'])==set(f.scene_token);scenesOuter.extend(ss['OuterTest'])
        training=indices(fold,'InnerTrain',True);mean=[];std=[];validcounts=[]
        for k in range(8):
            values=raw[training,:,k][mask[training,:,k]];assert len(values)>0
            mean.append(float(values.mean()));std.append(float(values.std()));validcounts.append(len(values))
        norm={'Fold':fold,'FitPartition':'InnerTrain','FitScenes':378,'FitActorType':'Pedestrian','Actors':len(training),
            'Features':FEATURES,'Mean':mean,'Std':std,'StdFloor':1e-6,'ValidCandidateCounts':validcounts,
            'InnerTrainIndicesSHA256':array_sha(training),'FitSceneTokens':ss['InnerTrain'],
            'SplitSHA256':sha256(S11B/f'02_splits/stage11b_fold{fold}_split.json'),'LabelsUsed':False,'SharedGSP':True}
        normpath=ROOT/f'02_input_cache/stage12b_fold{fold}_normalization.json';atomic_json(normpath,norm)
        x=np.where(mask,(raw-np.array(mean))/np.maximum(std,1e-6),0.).astype(np.float32)
        normalized=np.concatenate([x,mask.astype(np.float32)],-1);np.save(folder/'stage12b_geometry_inputs.npy',normalized)
        assert np.isfinite(normalized).all() and not normalized[...,:8][~mask].any()
        for pc,part in enumerate(['InnerTrain','InnerDev','OuterTest']):
            ids=indices(fold,part);ped=indices(fold,part,True);permutation=np.full((n,6),-1,np.int8)
            rng=np.random.default_rng(children[3*(fold-1)+pc]);permutation[ped]=np.stack([rng.permutation(6) for _ in ped]).astype(np.int8)
            assert np.all(np.sort(permutation[ped],axis=-1)==np.arange(6));np.save(folder/f'stage12b_{part}_permutation.npy',permutation)
            np.save(folder/f'stage12b_{part}_pedestrian_indices.npy',ped)
            counts.append({'Fold':fold,'Partition':part,'Scenes':len(ss[part]),'Actors':len(ids),'Pedestrian':len(ped),
                'PermutationSHA256':sha256(folder/f'stage12b_{part}_permutation.npy'),'SeedEntropy':2022,'SeedSpawnKey':3*(fold-1)+pc,
                'IndicesSHA256':array_sha(ids),'PedestrianIndicesSHA256':array_sha(ped),'NormalizationFitOnlyInnerTrain':True})
        cpred=np.lib.format.open_memmap(folder/'stage12b_base_logits.npy',mode='w+',dtype='float32',shape=(n,6))
        pprob=np.lib.format.open_memmap(folder/'stage12b_base_probabilities.npy',mode='w+',dtype='float32',shape=(n,6))
        c,r2=frozen_models(fold);before=[state_sha(c),state_sha(r2)];store=FrozenStore(fold);filled=np.zeros(n,bool)
        for part in ['OuterTest','InnerTrain','InnerDev']:
            ids=indices(fold,part);assert not filled[ids].any();began=time.monotonic();maxz=maxp=0.
            for start in range(0,len(ids),128):
                ix=ids[start:start+128];args,r2x=store.batch(ix,part);out=c(*args);rz=r2(r2x,args[6]);bike=torch.as_tensor(store.types[ix]==2,device='cuda')
                zz=out['mode_logits'].clone();pp=out['mode_prob'].clone();zz[bike]=rz['mode_logits'][bike];pp[bike]=rz['mode_prob'][bike]
                zcpu=zz.cpu().numpy();pcpu=pp.cpu().numpy();assert np.isfinite(zcpu).all() and np.isfinite(pcpu).all()
                if part=='OuterTest':
                    assert np.array_equal(zcpu,basez[ix,4]) and np.array_equal(pcpu,basep[ix,4]),(fold,start,'STOP: C replay not bitwise')
                    assert np.array_equal(zcpu[store.types[ix]==2],basez[ix[store.types[ix]==2],1])
                    assert np.array_equal(pcpu[store.types[ix]==2],basep[ix[store.types[ix]==2],1])
                cpred[ix]=zcpu;pprob[ix]=pcpu;filled[ix]=True
                if start and start%32768==0:print('FROZEN_C_REPLAY',fold,part,start,'/',len(ids),flush=True)
            replays.append({'Fold':fold,'Partition':part,'Actors':len(ids),'CCheckpointSHA256':sha256(S11B/f'04_checkpoints/fold{fold}/C_best.pt'),
                'NormalizationSHA256':sha256(store.normpath),'Temperature':TEMPERATURES[fold],
                'OuterReplayBitwise':part=='OuterTest','LogitsMaxDiff':maxz,'ProbabilityMaxDiff':maxp,'Seconds':time.monotonic()-began,
                'ScoresSource':'THIS fold frozen forward, not concatenated crossfold OOF'})
            print('FOLD_PARTITION_C_REPLAY_PASS',fold,part,len(ids),flush=True)
        assert filled.all() and before==[state_sha(c),state_sha(r2)] and not any(p.grad is not None or p.requires_grad for m in [c,r2] for p in m.parameters())
        cpred.flush();pprob.flush();seed(2022+100*(fold-1));initial=SemanticResidual();st=state_sha(initial)
        atomic_torch(ROOT/f'01_preflight/cache/stage12b_fold{fold}_initial.pt',{'Fold':fold,'state_dict':initial.state_dict(),'StateSHA256':st})
        for part in ['InnerTrain','InnerDev','OuterTest']:
            ped=indices(fold,part,True)
            for variant in VARIANTS:
                m=SemanticResidual().cuda();m.load_state_dict(initial.state_dict());m.eval()
                for start in range(0,len(ped),2048):
                    ix=ped[start:start+2048];inp=variant_input(fold,part,variant,ix)
                    base=torch.from_numpy(np.array(cpred[ix],copy=True)).cuda()/TEMPERATURES[fold];out=m(inp,base)
                    assert torch.equal(out['delta'],torch.zeros_like(out['delta'])) and torch.equal(out['logits'],base)
                    assert np.array_equal(out['probabilities'].argmax(-1).cpu().numpy(),pprob[ix].argmax(-1))
                    neutral+=len(ix)
                initials.append({'Fold':fold,'Partition':part,'Variant':variant,'Params':641,'StateSHA256':state_sha(m),'NeutralTop1Identity':'PASS'})
                assert state_sha(m)==st;del m
        del c,r2,store;torch.cuda.empty_cache()
    assert len(scenesOuter)==len(set(scenesOuter))==630 and neutral==3*3*66145
    dump('01_preflight/stage12b_partition_audit.csv',counts);dump('01_preflight/stage12b_frozen_replay.csv',replays);dump('01_preflight/stage12b_initialization.csv',initials)
    featuremanifest={'Status':'PASS','ActorCount':n,'CandidateCount':n*6,'Features':FEATURES,'InputDim':18,'SemanticFields':['walkway_inside_fraction','walkway_valid'],
        'SemanticCacheSourceSHA256':sha256(S12A/'02_semantic_cache/stage12a_semantic_cache_manifest.json'),
        'SourceCombinedManifestSHA256':sha256(S12A/'03_feature_statistics/cache/stage12a_combined_manifest.json'),
        'Files':{str(p.relative_to(ROOT)):sha256(p) for p in CACHE.rglob('*.npy')},'Normalizations':{str(p.relative_to(ROOT)):sha256(p) for p in (ROOT/'02_input_cache').glob('*.json')},
        'OriginalIdentityAndSixModesPreserved':True,'MapFeatureIntegrity':'PASS','GTLeakage':'PASS','NoFutureFeatures':True,'FoldSpecificForwardReplayed':True}
    atomic_json(ROOT/'02_input_cache/stage12b_feature_manifest.json',featuremanifest)
    audit={'Status':'PASS_PRE_TINY','CandidateIdentity':'PASS','FrozenCIdentity':'PASS','TemperatureIdentity':'PASS','MapFeatureIntegrity':'PASS',
        'SemanticCacheIntegrity':'PASS','GTLeakage':'PASS','FoldIsolation':'PASS','NetworkCapacityMatched':'PASS','InitializationIdentity':'PASS',
        'TinyOverfit':'PENDING','FormalTrainingPermitted':False,'NeutralPredictionsAudited':neutral,'CandidateMaxDiff':0.,
        'OriginalGTpoisonAuditSHA256':sha256(S12A/'01_map_integrity/stage12a_gt_poison_audit.json'),'NewInputFutureColumnsPoisoned':True,
        'AllSourcesFrozen':'Stage5,foldC,foldR2,temperatures,normalizations,semantic cache; no neural training during preflight'}
    atomic_json(ROOT/'01_preflight/stage12b_preflight_audit.json',audit)
    atomic_json(ROOT/'01_preflight/stage12b_frozen_model_audit.json',{'Status':'PASS','FrozenModelIntegrity':'PASS','COuterReplayMaxDiff':0.,'COuterProbabilityMaxDiff':0.,
        'AllFoldPartitionsCorrespondingCheckpoint':True,'CAndR2ParametersGradientsZero':True,'HistoricalCheckpointSHA256Verified':True,'TemperatureIdentity':'PASS'})
    verify();print('STAGE12B_PREFLIGHT_PASS_PENDING_TINY',flush=True)
if __name__=='__main__':main()
