"""Locked HeadDev70 evaluation, with fresh frozen predictor replay and exact Bicycle routing."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'01_training/00_code'))
from stage10a_common import *
from stage10a_model import DualExpert, Expert
import pandas as pd
MODELS=('R0','R2','G1','DualExpert')
METRICS=('minADE6','minADEOracle6','minFDE6','MR6','Top1ADE','Top1FDE','OracleGap','HitRate','BestModeRank','MRR','SoftCE')

@torch.no_grad()
def replay_headdev(store, identities, predictor):
    """Only TRAIN split source batches intersecting frozen HeadDev70 are opened."""
    dest=ROOT/'03_evaluation/cache'; audit=ROOT/'03_evaluation/stage10a_predictor_replay.json'
    if audit.exists():
        a=read_json(audit); assert a['status']=='PASS'
        for name,h in a['files'].items(): assert sha256(dest/name)==h
        return np.load(dest/'headdev_candidates.npy',mmap_mode='r'),np.load(dest/'headdev_GT_displacement.npy',mmap_mode='r')
    dev=set(read_json(SPLIT)['HeadDev']); di=np.flatnonzero(store.partition==0); n=len(di); assert n==29934
    positions=np.full(len(identities),-1,dtype=np.int64); positions[di]=np.arange(n)
    lookup={(r.scene_token,r.sample_token,r.instance_token):i for i,r in enumerate(identities.itertuples())}
    candidates=np.lib.format.open_memmap(dest/'headdev_candidates.npy',mode='w+',dtype='float32',shape=(n,6,12,2))
    displacement=np.lib.format.open_memmap(dest/'headdev_GT_displacement.npy',mode='w+',dtype='float32',shape=(n,))
    filled=np.zeros(n,bool)
    records=[r for r in read_json(S6/'01_cache/stage6a_cache_manifest.json')['batches'] if r['split']=='train' and dev&set(r['scene_tokens'])]
    ds=SceneDataset('train'); r2norm=read_json(S6/'02_features/stage6a_normalization.json')
    before=state_sha(predictor); source_windows=dev_windows=0; source_records=[]
    for bi,rec in enumerate(records):
        assert rec['split']=='train' and sha256(S6/rec['path'])==rec['sha256']
        old=torch.load(S6/rec['path'],map_location='cpu',weights_only=False)
        live=forward_batch(predictor,ds,rec['start']); assert len(live['windows'])==len(old['windows'])
        for w,ow in zip(live['windows'],old['windows']):
            assert w['dataset_index']==ow['dataset_index']
            assert w['raw_prediction_sha256']==ow['raw_prediction_sha256'] and w['ego_prediction_sha256']==ow['ego_prediction_sha256']
            assert w['instance_tokens']==ow['instance_tokens']
            for key in ('GT','future_mask','target_mask','agent_type','history','history_padding','mode_logits','mode_prob'):
                assert torch.equal(w[key],ow[key]),key
            source_windows+=1
            if w['scene_token'] not in dev: continue
            dev_windows+=1
            selected=torch.where(w['target_mask']&w['future_mask'].all(-1))[0]
            if not len(selected): continue
            ix=np.array([lookup[(w['scene_token'],w['sample_token'],w['instance_tokens'][t])] for t in selected.tolist()])
            assert np.all(store.partition[ix]==0); pp=positions[ix]; assert not filled[pp].any()
            args,_,_=graph_from_window(w,selected); cached,fde,ade=store.batch(ix,'cpu')
            assert all(torch.equal(a,b) for a,b in zip(args,cached)), 'Live/frozen G1 graph identity'
            prediction=w['ego_prediction'][selected]; distance=(prediction-w['GT'][selected,None]).norm(dim=-1)
            assert torch.equal(fde,distance[:,:,-1]) and torch.equal(ade,distance.mean(-1))
            feat,flags,_,_=observable_features(*[w[k].cuda() for k in ('history','history_padding','agent_type','ego_prediction','mode_logits','mode_prob')])
            normalized=r2normalize(feat.cpu(),flags.cpu(),r2norm,'R2')[selected]
            assert torch.equal(normalized,store.r2_batch(ix,'cpu')), 'Fresh frozen R2 input identity'
            for t,j in zip(selected.tolist(),ix):
                row=identities.iloc[j]
                assert row.node_in_graph==t and row.dataset_index==w['dataset_index']
                assert row.GT_trajectory_sha256==tensor_sha(w['GT'][t])
                assert row.future_mask_bits==''.join('1' if x else '0' for x in w['future_mask'][t].tolist())
                assert row.agent_type_id==int(w['agent_type'][t])
            candidates[pp]=prediction.numpy()
            displacement[pp]=(w['GT'][selected,-1]-w['current_position'][selected]).norm(dim=-1).numpy()
            filled[pp]=True
        source_records.append({'path':rec['path'],'sha256':rec['sha256'],'start':rec['start']})
        if (bi+1)%20==0: print('HEADDEV_REPLAY',bi+1,'/',len(records),'TARGETS',int(filled.sum()),flush=True)
    assert filled.all() and state_sha(predictor)==before
    assert not predictor.training and not any(p.requires_grad or p.grad is not None for p in predictor.parameters())
    candidates.flush(); displacement.flush(); ds.clear()
    atomic_json(audit,{'status':'PASS','split':'HeadDev70 within official TRAIN700','targets':n,'scenes':70,
        'source_batches':source_records,'source_windows_bitwise_replayed':source_windows,'HeadDev_windows':dev_windows,
        'fresh_raw_and_ego_predictions_bitwise_equal_frozen':True,'GT_future_mask_target_identity_bitwise':True,
        'all_G1_observation_features_bitwise_equal_cache':True,'all_R2_features_bitwise_equal_cache':True,
        'future_GT_used_only_for_supervision_or_offline_metrics':True,'predictor_state_unchanged':True,
        'official_VAL_opened':False,'test_opened':False,
        'files':{p.name:sha256(p) for p in (dest/'headdev_candidates.npy',dest/'headdev_GT_displacement.npy')}})
    return candidates,displacement

@torch.no_grad()
def main():
    seed(); verify(); frozen=read_json(FROZEN); assert frozen['status']=='FROZEN_ALL_COMPLETE' and frozen['training_prohibited']
    assert read_json(ROOT/'01_training/stage10a_tiny_gate.json')['status']=='PASS'
    register=ROOT/'03_evaluation/stage10a_headdev_registration.json'
    specification={'status':'LOCKED_BEFORE_UNIFIED_HEADDEV','split_sha256':sha256(SPLIT),'models':MODELS,
        'checkpoint_manifest_sha256':sha256(FROZEN),'evaluation_source_sha256':sha256(__file__),
        'evidence_role':'HeadDev feasibility screening, checkpoint-selected; not independent confirmation',
        'official_VAL_loaded':False,'test_loaded':False}
    if register.exists(): assert read_json(register)==json.loads(json.dumps(specification))
    else: atomic_json(register,specification)
    assert not (ROOT/'03_evaluation/stage10a_complete.json').exists(), 'HeadDev unified evaluation already complete'
    models={e:fresh(e) for e in EXPERTS}
    for rec in frozen['checkpoints']:
        path=PROJECT/rec['Path']; assert sha256(path)==rec['SHA256']
        cp=torch.load(path,map_location='cpu',weights_only=False)
        assert cp['expert']==rec['Expert'] and cp['registration_sha256']==sha256(REG)
        models[rec['Expert']].load_state_dict(cp['state_dict']); models[rec['Expert']].eval().requires_grad_(False)
    r2=frozen_r2(); predictor=frozen_predictor(); states=[state_sha(x) for x in (r2,predictor)]
    assert sha256(G1PATH)==G1SHA
    g1=Expert(2022).cuda(); g1.load_state_dict(torch.load(G1PATH,map_location='cpu',weights_only=False)['state_dict'])
    g1.eval().requires_grad_(False)
    dual=DualExpert(models['vehicle'],models['pedestrian'],r2).eval()
    store=Store(); identities=pd.read_csv(store.root/'identities.csv',dtype={'future_mask_bits':str}); ids=np.flatnonzero(store.partition==0)
    candidates,displacement=replay_headdev(store,identities,predictor)
    rows=[]; bicycle_count=0; probability_blocks={m:[] for m in MODELS}; logit_blocks={m:[] for m in MODELS}
    for start in range(0,len(ids),128):
        ix=ids[start:start+128]; args,fde,ade=store.batch(ix); feature=store.r2_batch(ix)
        prediction=torch.from_numpy(np.array(candidates[start:start+len(ix)],copy=True)).cuda(); before=tensor_sha(prediction)
        base=r2(feature,args[3]); out=dual(*args,feature,prediction); ref=g1(*args)
        logits={'R0':args[3],'R2':base['mode_logits'],'G1':ref['mode_logits'],'DualExpert':out['mode_logits']}
        probability={'R0':args[3].softmax(-1),'R2':base['mode_prob'],'G1':ref['mode_prob'],'DualExpert':out['mode_prob']}
        assert out['candidates'] is prediction and tensor_sha(out['candidates'])==before
        bike=torch.as_tensor(store.types[ix]==2,device='cuda')
        assert torch.equal(logits['DualExpert'][bike],logits['R2'][bike]) and torch.equal(probability['DualExpert'][bike],probability['R2'][bike])
        assert torch.equal(out['candidates'][bike],prediction[bike]); bicycle_count+=int(bike.sum())
        best=fde.argmin(-1); ii=torch.arange(len(ix),device='cuda'); minimum=fde[ii,best]; minade=ade[ii,best]
        metrics={}; tops={}
        for m in MODELS:
            q=probability[m]; assert torch.isfinite(q).all() and torch.isfinite(logits[m]).all()
            assert torch.allclose(q.sum(-1),torch.ones(len(ix),device='cuda'),atol=1e-6,rtol=0.)
            top=q.argmax(-1); tops[m]=top.cpu()
            rank=(q.argsort(dim=-1,descending=True,stable=True)==best[:,None]).long().argmax(-1)+1
            metrics[m]=torch.stack((minade,ade.min(-1).values,minimum,(minimum>2).float(),ade[ii,top],fde[ii,top],
                fde[ii,top]-minimum,(top==best).float(),rank.float(),1/rank.float(),per_actor_loss(logits[m],fde)),-1).cpu().numpy()
            assert np.array_equal(metrics[m][:,:4],metrics['R0'][:,:4])
            probability_blocks[m].append(q.cpu().numpy()); logit_blocks[m].append(logits[m].cpu().numpy())
        for a,j in enumerate(ix):
            old=identities.iloc[j]
            row={k:old[k] for k in ('scene_token','sample_token','instance_token','dataset_index','node_in_graph','agent_type','motion_state')}
            row.update({'source_index':int(j),'horizon':'full_horizon','GT_endpoint_displacement_m':float(displacement[start+a]),'best_mode':int(best[a])})
            for m in MODELS:
                row.update({m+'_'+metric:float(x) for metric,x in zip(METRICS,metrics[m][a])})
                row[m+'_top1_mode']=int(tops[m][a])
            rows.append(row)
        if (start//128+1)%50==0: print('HEADDEV_ALL_MODELS',len(rows),'/',len(ids),flush=True)
    assert len(rows)==29934 and len({r['scene_token'] for r in rows})==70
    assert len({(r['scene_token'],r['sample_token'],r['instance_token']) for r in rows})==29934 and bicycle_count==158
    assert states==[state_sha(x) for x in (r2,predictor)]
    assert not any(p.requires_grad or p.grad is not None for m in (r2,predictor) for p in m.parameters())
    write_csv(ROOT/'03_evaluation/stage10a_actor_results.csv',rows)
    np.savez_compressed(ROOT/'03_evaluation/cache/stage10a_mode_outputs.npz',
        source_index=ids,**{m+'_prob':np.concatenate(probability_blocks[m]) for m in MODELS},
        **{m+'_logits':np.concatenate(logit_blocks[m]) for m in MODELS})
    candidate_audit={'status':'PASS','targets':29934,'scenes':70,'models':MODELS,'fresh_prediction_cache_identity':'bitwise',
        'candidate_maxdiff':0.,'minADE6_maxdiff':0.,'minADEOracle6_maxdiff':0.,'minFDE6_maxdiff':0.,'MR6_maxdiff':0.,
        'Bicycle_targets':158,'Bicycle_logits_bitwise_R2':True,'Bicycle_probability_bitwise_R2':True,'Bicycle_candidates_bitwise_R2':True,
        'predictor_gradient_count':0,'R2_gradient_count':0,'candidate_requires_grad':False,'predictor_R2_states_unchanged':True,
        'official_VAL_opened':False,'test_opened':False}
    atomic_json(ROOT/'03_evaluation/stage10a_candidate_identity.json',candidate_audit)
    write_csv(ROOT/'06_tables/stage10a_candidate_identity.csv',[
        {'Model':m,'Targets':29934,'CandidateMaxDiff':0.,'minADE6MaxDiff':0.,'minADEOracle6MaxDiff':0.,
            'minFDE6MaxDiff':0.,'MR6MaxDiff':0.,'BicycleLogitsBitwiseR2':int(m in ('R2','DualExpert')),
            'BicycleProbabilityBitwiseR2':int(m in ('R2','DualExpert')),'Status':'PASS'} for m in MODELS])
    atomic_json(ROOT/'03_evaluation/stage10a_complete.json',{'status':'PASS','split':'HeadDev70','targets':29934,'scenes':70,
        'models':MODELS,'successful_unified_HeadDev_passes':1,'official_VAL_opened':False,'test_opened':False,
        'actor_csv_sha256':sha256(ROOT/'03_evaluation/stage10a_actor_results.csv'),
        'checkpoint_manifest_sha256':sha256(FROZEN),'evidence_role':specification['evidence_role']})
    verify(); print('STAGE10_HEADDEV_EVALUATION_PASS',candidate_audit,flush=True)

if __name__=='__main__': main()
