"""Before training: unchanged graph, independent experts, GT-free inputs, exact routing."""
from stage10a_common import *
from stage10a_model import DualExpert
import copy, pandas as pd

@torch.no_grad()
def main():
    seed(); verify(history=True); assert not REG.exists(), 'Initialization registration is immutable'
    source=S8/'01_training/cache'; old=read_json(source/'manifest.json')
    for name in ('arg0.npy','arg1.npy','arg2.npy','arg6.npy','fde.npy','ade.npy','partition.npy','identities.csv'):
        assert sha256(source/name)==old['files'][name],name
    store=Store(); ids=pd.read_csv(store.root/'identities.csv')
    for name,h in store.manifest['files'].items(): assert sha256(store.root/name)==h
    split=read_json(SPLIT); train=set(split['HeadTrain']); dev=set(split['HeadDev'])
    assert np.array_equal(store.partition,ids.HeadTrain.to_numpy())
    assert set(ids.loc[ids.HeadTrain==1,'scene_token'])==train and set(ids.loc[ids.HeadTrain==0,'scene_token'])==dev
    assert np.array_equal(store.types,ids.agent_type.map(dict(zip(CLASSES,range(3)))).to_numpy())
    models={e:fresh(e,'cpu') for e in EXPERTS}
    pointers=[{p.data_ptr() for p in m.parameters()} for m in models.values()]
    assert not pointers[0]&pointers[1]
    for e,m in models.items():
        baseline=SparseGraphReranker('G1',seed=SEEDS[e])
        assert state_sha(m)==state_sha(baseline)
        assert set(m.state_dict())==set(baseline.state_dict())
        assert torch.count_nonzero(m.head[-1].weight)==0 and torch.count_nonzero(m.head[-1].bias)==0
        assert not hasattr(m,'gate') and not hasattr(m,'adapters') and not hasattr(m,'map_aggregator')
        assert sum(p.numel() for p in m.parameters())==PARAMS
    assert state_sha(models['vehicle'])!=state_sha(models['pedestrian'])
    predictor=frozen_predictor('cpu'); r2=frozen_r2(); states=[state_sha(predictor),state_sha(r2)]
    keys=['scene_token','sample_token','instance_token']
    lookup={tuple(getattr(r,k) for k in keys):i for i,r in enumerate(ids.itertuples())}
    records=[r for r in read_json(S6/'01_cache/stage6a_cache_manifest.json')['batches'] if r['split']=='train']
    # First frozen TRAIN batch, all windows: independently rebuild the G1 graph.
    record=records[0]; assert sha256(S6/record['path'])==record['sha256']
    payload=torch.load(S6/record['path'],map_location='cpu',weights_only=False)
    r2norm=read_json(S6/'02_features/stage6a_normalization.json'); checked=0; r2diff=0.
    for w in payload['windows']:
        selected=torch.where(w['target_mask']&w['future_mask'].all(-1))[0]
        if not len(selected): continue
        args,idx,keep=graph_from_window(w,selected)
        changed=copy.deepcopy(w); changed['GT'].fill_(float('nan'))
        changed['future_mask']=~changed['future_mask']; changed['target_mask']=~changed['target_mask']
        changed['lane_semantic']=torch.full((1,9),float('nan'))
        poison,pi,pk=graph_from_window(changed,selected)
        assert all(torch.equal(a,b) for a,b in zip(args,poison)) and torch.equal(idx,pi) and torch.equal(keep,pk)
        ix=np.array([lookup[(w['scene_token'],w['sample_token'],w['instance_tokens'][t])] for t in selected.tolist()])
        cached,fd,ad=store.batch(ix,'cpu'); assert all(torch.equal(a,b) for a,b in zip(args,cached))
        distance=(w['ego_prediction'][selected]-w['GT'][selected,None]).norm(dim=-1)
        assert torch.equal(fd,distance[:,:,-1]) and torch.equal(ad,distance.mean(-1))
        feat,flags,_,_=observable_features(*[w[k].cuda() for k in ('history','history_padding','agent_type','ego_prediction','mode_logits','mode_prob')])
        norm=r2normalize(feat,flags,r2norm,'R2')[selected]
        freshlog=r2(norm,w['mode_logits'][selected].cuda())['mode_logits']
        cachedlog=r2(store.r2_batch(ix),cached[3].cuda())['mode_logits']
        r2diff=max(r2diff,float((freshlog-cachedlog).abs().max())); checked+=len(selected)
        for e,m in models.items():
            ref=SparseGraphReranker('G1',seed=SEEDS[e])
            out=m(*args); reference=ref(*args[:3],None,None,None,args[3])
            assert all(torch.equal(out[k],reference[k]) for k in out)
    assert r2diff<1e-6
    # All frozen target contexts retain cross-type neighbors, including Bicycle.
    neighbor_counts=np.zeros((3,3),dtype=np.int64)
    for start in range(0,len(ids),4096):
        stop=min(start+4096,len(ids)); ty=store.types[start:stop]
        nt=np.asarray(store.args[0][start:stop,1:,0,:3]).argmax(-1); mask=np.asarray(store.args[2][start:stop])
        for a in range(3):
            for b in range(3): neighbor_counts[a,b]+=np.sum((ty==a)[:,None]&mask&(nt==b))
    assert neighbor_counts[0,2]>0 and neighbor_counts[1,2]>0
    write_csv(ROOT/'06_tables/stage10a_neighbor_type_counts.csv',[
        {'TargetType':CLASSES[a],'NeighborType':CLASSES[b],'DirectedCurrentNeighborOccurrences':int(neighbor_counts[a,b])}
        for a in range(3) for b in range(3)])
    # Neutral routing on >=1024 targets of each expert, with independent seed.
    dual=DualExpert(models['vehicle'].cuda(),models['pedestrian'].cuda(),r2).eval()
    neutral=[]
    for t,e in enumerate(EXPERTS):
        pool=np.flatnonzero((store.partition==1)&(store.types==t))
        chosen=np.random.default_rng(SEEDS[e]).choice(pool,1024,replace=False); ml=mp=0.
        for start in range(0,len(chosen),128):
            ix=chosen[start:start+128]; args,_,_=store.batch(ix)
            probe=args[3].new_zeros((len(ix),6,12,2))
            out=dual(*args,store.r2_batch(ix),probe)
            ml=max(ml,float((out['mode_logits']-args[3]).abs().max()))
            mp=max(mp,float((out['mode_prob']-args[3].softmax(-1)).abs().max()))
            assert out['candidates'] is probe and torch.equal(out['candidates'],probe)
        assert ml<1e-6 and mp<1e-6
        neutral.append({'Expert':e,'Targets':1024,'Partition':'HeadTrain','LogitMaxDiff':ml,'ProbabilityMaxDiff':mp,'Status':'PASS'})
        atomic_json(ROOT/'01_training'/e/'stage10a_neutral_audit.json',neutral[-1])
    # Every full-horizon Bicycle in TRAIN700, including HeadDev, keeps exact R2 logits/probabilities.
    bikes=np.flatnonzero(store.types==2); bikecount=0
    for start in range(0,len(bikes),128):
        ix=bikes[start:start+128]; args,_,_=store.batch(ix); feature=store.r2_batch(ix)
        probe=torch.arange(len(ix)*6*12*2,device='cuda',dtype=torch.float32).reshape(len(ix),6,12,2)
        out=dual(*args,feature,probe); ref=r2(feature,args[3])
        assert torch.equal(out['mode_logits'],ref['mode_logits']) and torch.equal(out['mode_prob'],ref['mode_prob'])
        assert out['candidates'] is probe and torch.equal(out['candidates'],probe); bikecount+=len(ix)
    assert bikecount==3138
    # Real frozen predictor candidates on first fixed 128 Bicycle targets.
    chosen=ids.iloc[bikes[:128]]; realcount=0
    for cachepath,rows in chosen.groupby('cache_path',sort=True):
        record=next(r for r in records if r['path']==cachepath)
        assert sha256(S6/cachepath)==record['sha256']
        block=torch.load(S6/cachepath,map_location='cpu',weights_only=False)
        for w in block['windows']:
            rr=rows[rows.dataset_index==w['dataset_index']]
            if rr.empty: continue
            target=torch.as_tensor(rr.node_in_graph.to_numpy()); ix=rr.source_index.to_numpy()
            args,_,_=graph_from_window(w,target); args=tuple(a.cuda() for a in args)
            candidate=w['ego_prediction'][target].cuda(); before=tensor_sha(candidate)
            out=dual(*args,store.r2_batch(ix),candidate); ref=r2(store.r2_batch(ix),args[3])
            assert out['candidates'] is candidate and tensor_sha(out['candidates'])==before
            assert torch.equal(out['mode_logits'],ref['mode_logits']) and torch.equal(out['mode_prob'],ref['mode_prob'])
            realcount+=len(target)
    assert realcount==128
    assert states==[state_sha(predictor),state_sha(r2)]
    assert not any(p.requires_grad or p.grad is not None for m in (predictor,r2) for p in m.parameters())
    write_csv(ROOT/'06_tables/stage10a_neutral_identity.csv',neutral)
    audit={'status':'PASS','HeadTrain630_HeadDev70_exact':True,'full_horizon_targets':290085,
        'identity_FDE_ADE_original_logits_bitwise':True,'TRAIN_windows_rebuilt':len(payload['windows']),'TRAIN_targets_rebuilt':checked,
        'GT_nan_future_target_mask_poison_ignored':True,'semantic_nan_ignored':True,'unchanged_G1_forward_bitwise':True,
        'R2_rebuild_maxdiff':r2diff,'independent_parameters':True,'each_parameters':PARAMS,'total_trainable_parameters':2*PARAMS,
        'neighbor_type_counts':neighbor_counts.tolist(),'Bicycle_neutral_logits_probabilities_bitwise_targets':bikecount,
        'Bicycle_actual_candidate_identity_targets':realcount,'Bicycle_candidate_probe_identity_targets':bikecount,
        'Bicycle_requires_grad':False,'predictor_R2_gradients':0,'predictor_R2_states_unchanged':True,
        'neutral_experts':neutral,'official_VAL_opened':False,'test_opened':False}
    atomic_json(ROOT/'01_training/stage10a_engineering_audit.json',audit)
    # Initial states were captured before any optimizer existed; source hashes are immutable.
    atomic_json(REG,{'status':'REGISTERED_BEFORE_TINY_AND_FORMAL','base_commit':BASE,
        'protocol_sha256':sha256(PROTOCOL),'normalization_sha256':sha256(NORM),'split_sha256':sha256(SPLIT),
        'cache_manifest_sha256':sha256(store.root/'manifest.json'),
        'training_sources':{p.name:sha256(p) for p in sorted(Path(__file__).parent.glob('stage10a_*.py'))},
        'initial_state_sha256':{e:state_sha(m) for e,m in models.items()},'seeds':SEEDS,
        'parameters_each':PARAMS,'parameters_total':2*PARAMS,'independent_parameters':True,
        'init_matches_untrained_G1':True,'last_score_zero_initialized':True,'pretrained_graph_weights_loaded':False,
        'checkpoint_selection':'minimum own-type entire HeadDev SoftCE','official_VAL_opened':False,'test_opened':False})
    print('STAGE10_ENGINEERING_AND_REGISTRATION_PASS',audit,flush=True)

if __name__=='__main__': main()
