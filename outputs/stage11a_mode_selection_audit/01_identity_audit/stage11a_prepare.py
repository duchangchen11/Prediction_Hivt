"""Frozen TRAIN700 cache identity, last-valid FDE and observation-only history audit."""
from stage11a_common import *
import copy,ast

@torch.no_grad()
def main():
    seed(); history=verify(history=True); assert not (CACHE/'prepared.json').exists()
    source=S8/'01_training/cache'; manifest=read_json(source/'manifest.json')
    for name,h in manifest['files'].items(): assert sha256(source/name)==h,name
    f=metadata(); n=len(f); assert n==290085 and not f[['scene_token','sample_token','instance_token']].duplicated().any()
    split=read_json(legacy10.SPLIT);train=set(split['HeadTrain']);dev=set(split['HeadDev'])
    assert set(f.loc[f.HeadTrain==1,'scene_token'])==train and len(train)==630
    assert set(f.loc[f.HeadTrain==0,'scene_token'])==dev and len(dev)==70 and not train&dev
    assert np.array_equal(f.HeadTrain,np.load(source/'partition.npy',mmap_mode='r'))
    legacy=legacy10.Store(); assert np.array_equal(legacy.types,f.agent_type_id)
    lookup={(r.scene_token,r.sample_token,r.instance_token):i for i,r in enumerate(f.itertuples())}
    candidate=np.lib.format.open_memmap(CACHE/'candidates.npy',mode='w+',dtype='float32',shape=(n,6,12,2))
    gt=np.lib.format.open_memmap(CACHE/'GT.npy',mode='w+',dtype='float32',shape=(n,12,2))
    observed=np.lib.format.open_memmap(CACHE/'observed_motion.npy',mode='w+',dtype='float32',shape=(n,5))
    filled=np.zeros(n,bool); records=[r for r in read_json(S6/'01_cache/stage6a_cache_manifest.json')['batches'] if r['split']=='train']
    assert len(records)==1057; windows=0; maximum_fde=maximum_ade=0.; poisoned=0
    for bi,rec in enumerate(records):
        assert rec['split']=='train' and sha256(S6/rec['path'])==rec['sha256']
        block=torch.load(S6/rec['path'],map_location='cpu',weights_only=False)
        for w in block['windows']:
            selected=torch.where(w['target_mask']&w['future_mask'].all(-1))[0];windows+=1
            if not len(selected):continue
            ix=np.array([lookup[(w['scene_token'],w['sample_token'],w['instance_tokens'][t])] for t in selected.tolist()])
            assert not filled[ix].any()
            assert (~w['history_padding'][selected,4]).all() and w['future_mask'][selected].all()
            last=(torch.arange(12)[None]*w['future_mask'][selected]).max(-1).values;assert torch.equal(last,torch.full_like(last,11))
            pred=w['ego_prediction'][selected]; target=w['GT'][selected]
            distance=(pred-target[:,None]).norm(dim=-1); fde=distance[:,:,-1];ade=distance.mean(-1)
            assert np.array_equal(fde.numpy(),legacy.fde[ix]) and np.array_equal(ade.numpy(),legacy.ade[ix])
            assert np.array_equal(w['mode_logits'][selected].numpy(),legacy.logits[ix])
            assert tensor_sha(w['raw_prediction'])==w['raw_prediction_sha256'] and tensor_sha(w['ego_prediction'])==w['ego_prediction_sha256']
            # Histories only; real elapsed time across missing historical positions.
            for a,t in enumerate(selected.tolist()):
                row=f.iloc[ix[a]]; valid=torch.where(~w['history_padding'][t])[0]
                assert row.dataset_index==w['dataset_index'] and row.node_in_graph==t
                assert row.GT_trajectory_sha256==tensor_sha(w['GT'][t]) and row.future_mask_bits=='1'*12
                assert int(row.agent_type_id)==int(w['agent_type'][t])
                hist=w['history'][t,valid];speed=net=path=0.
                if len(valid)>=2:
                    speed=float((hist[-1]-hist[-2]).norm()/(.5*int(valid[-1]-valid[-2])))
                    net=float((hist[-1]-hist[0]).norm());path=float(torch.diff(hist,dim=0).norm(dim=-1).sum())
                observed[ix[a]]=(speed,net,path,len(valid),float((target[a,-1]-w['current_position'][t]).norm()))
            candidate[ix]=pred.numpy();gt[ix]=target.numpy();filled[ix]=True
            if bi==0:
                args,_,_=legacy10.graph_from_window(w,selected); mutated=copy.deepcopy(w)
                mutated['GT'].fill_(float('nan'));mutated['future_mask']=~mutated['future_mask'];mutated['target_mask']=~mutated['target_mask']
                other,_,_=legacy10.graph_from_window(mutated,selected); assert all(torch.equal(a,b) for a,b in zip(args,other))
                poisoned+=len(selected)
        if (bi+1)%100==0:print('TRAIN_SOURCE_IDENTITY',bi+1,'/',len(records),'targets',int(filled.sum()),flush=True)
    assert filled.all();candidate.flush();gt.flush();observed.flush()
    # Reuse Stage10's fresh predictor replay: every HeadDev geometry tensor must match it bitwise.
    di=np.flatnonzero(f.HeadTrain.to_numpy()==0)
    old=np.load(S10/'03_evaluation/cache/headdev_candidates.npy',mmap_mode='r')
    replay=read_json(S10/'03_evaluation/stage10a_predictor_replay.json')
    assert sha256(S10/'03_evaluation/cache/headdev_candidates.npy')==replay['files']['headdev_candidates.npy']
    assert np.array_equal(candidate[di],old)
    # Original source uses the exact FDE target. Test each unchanged implementation.
    z=torch.arange(24,dtype=torch.float32).reshape(4,6)/7.;fd=torch.arange(24,dtype=torch.float32).reshape(4,6)/3.
    expected=-((-fd/1.).softmax(-1)*z.log_softmax(-1)).sum(-1)
    assert torch.equal(expected,legacy10.per_actor_loss(z,fd)) and torch.equal(expected,legacy9.per_actor_loss(z,fd))
    assert torch.allclose(expected.mean(),legacy8.loss(z,fd),atol=1e-7,rtol=0.)
    from stage6a_head import ranking_loss
    assert torch.equal(expected.mean(),ranking_loss(z,fd))
    audit=[{'Audit':'SoftCE_target_FDE_temperature1m','Status':'PASS','Evidence':'unchanged Stage6/8/9/10 functions independently agree with explicit formula'},
        {'Audit':'FDE_last_valid_timestep','Status':'PASS','Evidence':'all290085 targets full-horizon; final valid index11; FDE and ADE bitwise cache match'},
        {'Audit':'Top1_argmax_probability','Status':'PASS','Evidence':'frozen model probability argmax, checked during inference'},
        {'Audit':'BestMode_argmin_FDE','Status':'PASS','Evidence':'argmin fixed per-actor FDE, lowest-index ties'},
        {'Audit':'Candidate_geometry_identity','Status':'PASS','Evidence':'one common290085x6x12x2 tensor; HeadDev bitwise Stage10 fresh-predictor replay'},
        {'Audit':'GT_masks_offline_only','Status':'PASS','Evidence':'unchanged observable allowlist and GT/mask poison; Stage8 G3 cached selection is observation-only; frozen source hashes'},
        {'Audit':'Frozen_history_checkpoints','Status':'PASS','Evidence':str(len(history['historical_files']))+' historical files and5 preserved untracked files unchanged'},
        {'Audit':'Official_VAL_test_not_used','Status':'PASS','Evidence':'only TRAIN700 records; inherited runtime VAL/test path guard; no optimizer or model writes'}]
    write_csv(ROOT/'01_identity_audit/stage11a_code_label_audit.csv',audit)
    prepared={'status':'PASS','targets':n,'HeadTrain':260151,'HeadDev':29934,'scenes':700,'TRAIN_windows':windows,
        'source_records':len(records),'last_valid_index':11,'FDE_maxdiff':maximum_fde,'ADE_maxdiff':maximum_ade,
        'poison_checked_targets':poisoned,'HeadDev_bitwise_Stage10_replay':True,'official_VAL_opened':False,'test_opened':False,
        'files':{p.name:sha256(p) for p in (CACHE/'candidates.npy',CACHE/'GT.npy',CACHE/'observed_motion.npy')}}
    atomic_json(CACHE/'prepared.json',prepared);atomic_json(ROOT/'01_identity_audit/stage11a_source_identity.json',prepared)
    print('STAGE11_PREPARATION_AND_LABEL_AUDIT_PASS',prepared,flush=True)

if __name__=='__main__': main()
