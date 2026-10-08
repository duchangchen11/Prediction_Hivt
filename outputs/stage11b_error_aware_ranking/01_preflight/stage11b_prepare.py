"""HeadTrain-only raw feature reconstruction and nested per-fold moments."""
from pathlib import Path
import sys,copy
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'00_manifest'))
from stage11b_common import *

def raw_graph(w,targets):
    obs=old.observable_window(w,'',np.zeros(2),0.);node=old.node_features(obs);idx,keep=old.neighbors(obs)
    edge=old.interaction_edges(obs,idx,keep,targets)
    return torch.cat((node[targets,None],node[idx[targets]]),1),edge,keep[targets],w['mode_logits'][targets]

@torch.no_grad()
def main():
    seed();verify(history=True);assert REG.exists();CACHE.mkdir(exist_ok=True)
    assert not (CACHE/'prepared.json').exists()
    original=pd.read_csv(S10/'01_training/cache/identities.csv',dtype={'future_mask_bits':str})
    heads=read_json(S6/'00_manifest/stage6a_head_split.json');scenes=set(heads['HeadTrain']);forbidden=set(heads['HeadDev'])
    f=original[original.HeadTrain==1].copy().reset_index(drop=True);assert len(f)==260151
    assert set(f.scene_token)==scenes and len(scenes)==630 and not scenes&forbidden
    motion=np.load(S11/'01_identity_audit/cache/observed_motion.npy',mmap_mode='r')
    for j,name in enumerate(('recent_speed','history_net','history_path','history_valid_count','GT_displacement')):f[name]=motion[f.source_index,j]
    f['actor_id']=f.scene_token+'|'+f.sample_token+'|'+f.instance_token;assert not f.actor_id.duplicated().any()
    f.to_csv(CACHE/'identities.csv',index=False)
    ordered=sorted(scenes,key=lambda s:(hashlib.sha256(('2022|outer|'+s).encode()).hexdigest(),s))
    outer=[ordered[k*210:(k+1)*210] for k in range(3)];rows=[]
    for fold in range(1,4):
        test=outer[fold-1];rest=scenes-set(test)
        order=sorted(rest,key=lambda s:(hashlib.sha256((f'2022|inner|fold{fold}|'+s).encode()).hexdigest(),s))
        dev=order[:42];train=order[42:];assert (len(train),len(dev),len(test))==(378,42,210)
        assert not (set(train)&set(dev) or set(train)&set(test) or set(dev)&set(test))
        atomic_json(ROOT/f'02_splits/stage11b_fold{fold}_split.json',{'Fold':fold,'InnerTrain':train,'InnerDev':dev,'OuterTest':test,
            'seed':2022,'protocol_sha256':sha256(PROTOCOL),'HeadDev_overlap':0,'scene_isolation':'PASS'})
        for part,ss in (('InnerTrain',train),('InnerDev',dev),('OuterTest',test)):
            count=f.groupby('scene_token').size()
            for s in ss:rows.append({'Fold':fold,'Partition':part,'SceneToken':s,'ActorCount':int(count[s])})
    assert set(sum(outer,[]))==scenes and len(set(sum(outer,[])))==630
    dump('02_splits/stage11b_fold_split.csv',rows)
    src=S8/'01_training/cache';manifest=read_json(src/'manifest.json')
    for name in ('arg0.npy','arg1.npy','arg2.npy','arg6.npy','fde.npy','ade.npy','partition.npy','identities.csv'):
        assert sha256(src/name)==manifest['files'][name],name
    packed=[np.load(src/f'arg{k}.npy',mmap_mode='r') for k in (0,1,2)]
    fd=np.load(src/'fde.npy',mmap_mode='r');ad=np.load(src/'ade.npy',mmap_mode='r')
    cp=np.load(S11/'01_identity_audit/cache/candidates.npy',mmap_mode='r')
    x=np.lib.format.open_memmap(CACHE/'r2_raw.npy',mode='w+',dtype='float32',shape=(len(f),6,19))
    fl=np.lib.format.open_memmap(CACHE/'r2_flags.npy',mode='w+',dtype='bool',shape=(len(f),3))
    lookup={(r.scene_token,r.sample_token,r.instance_token):i for i,r in enumerate(f.itertuples())};filled=np.zeros(len(f),bool)
    records=[r for r in read_json(S6/'01_cache/stage6a_cache_manifest.json')['batches'] if r['split']=='train']
    poison=0;windows=0;neighbor_bike=0;skipped=0
    for bi,record in enumerate(records):
        assert sha256(S6/record['path'])==record['sha256'];block=torch.load(S6/record['path'],map_location='cpu',weights_only=False)
        for w in block['windows']:
            if w['scene_token'] not in scenes:skipped+=1;continue
            targets=torch.where(w['target_mask']&w['future_mask'].all(-1))[0]
            if not len(targets):continue
            ids=np.array([lookup[(w['scene_token'],w['sample_token'],w['instance_tokens'][t])] for t in targets.tolist()])
            assert not filled[ids].any();source=f.source_index.to_numpy()[ids]
            assert np.array_equal(cp[source],w['ego_prediction'][targets].numpy())
            dist=(w['ego_prediction'][targets]-w['GT'][targets,None]).norm(dim=-1)
            assert np.array_equal(fd[source],dist[:,:,-1].numpy()) and np.array_equal(ad[source],dist.mean(-1).numpy())
            feat,flags,_,_=observable_features(*[w[n].cuda() for n in ('history','history_padding','agent_type','ego_prediction','mode_logits','mode_prob')])
            x[ids]=feat[targets].cpu().numpy();fl[ids]=flags[targets].cpu().numpy();filled[ids]=True;windows+=1
            if poison<100:
                graph=raw_graph(w,targets)
                assert all(np.array_equal(packed[k][source],graph[k].numpy()) for k in range(3))
                poisoned=copy.deepcopy(w);poisoned['GT'].fill_(float('nan'));poisoned['future_mask']=~poisoned['future_mask'];poisoned['target_mask']=~poisoned['target_mask']
                other=raw_graph(poisoned,targets);assert all(torch.equal(a,b) for a,b in zip(graph,other))
                feat2,flags2,_,_=observable_features(*[poisoned[n].cuda() for n in ('history','history_padding','agent_type','ego_prediction','mode_logits','mode_prob')])
                assert torch.equal(feat,feat2) and torch.equal(flags,flags2)
                neighbor_bike+=int(((graph[0][:,1:,0,2]==1)&graph[2]).sum());poison+=1
        if (bi+1)%100==0:print('HEADTRAIN_RAW',bi+1,'/',len(records),'actors',int(filled.sum()),flush=True)
    assert filled.all() and poison==100;x.flush();fl.flush()
    normalization=[];initial=[]
    for fold in range(1,4):
        train=indices(fold,'InnerTrain');sums={(k,c):[0,0.,0.] for k,cc in ((0,range(3,15)),(1,range(11))) for c in cc}
        rsums={c:[0,0.,0.] for c in range(19)}
        for start in range(0,len(train),1024):
            ix=train[start:start+1024];source=f.source_index.to_numpy()[ix]
            node,edge,mask=[torch.from_numpy(np.array(a[source],copy=True)) for a in packed]
            nv=torch.cat((torch.ones((len(ix),1),dtype=torch.bool),mask),1)[...,None].expand(-1,9,6);ev=mask[:,None,:,None].expand(-1,6,8,6)
            for k,raw,valid,cc in ((0,node,nv,range(3,15)),(1,edge,ev,range(11))):
                for c in cc:
                    v=raw[...,c][valid].double();r=sums[k,c];r[0]+=len(v);r[1]+=float(v.sum());r[2]+=float(v.square().sum())
            xx=x[ix].astype(np.float64);flags=fl[ix]
            for c in range(19):
                v=xx[flags[:,0],:,c] if 12<=c<15 else xx[:,:,c];r=rsums[c]
                r[0]+=v.size;r[1]+=float(v.sum());r[2]+=float(np.square(v).sum())
        stats={'0':{},'1':{}}
        for (k,c),(count,total,sq) in sums.items():
            mean=total/count;std=float(np.sqrt(max(0,sq/count-mean*mean)));stats[str(k)][str(c)]={'count':count,'mean':mean,'std':std}
        rmean=[];rstd=[]
        for c,(count,total,sq) in rsums.items():
            mean=total/count;rmean.append(mean);rstd.append(float(np.sqrt(max(0,sq/count-mean*mean))))
        norm={'Fold':fold,'Status':'PASS','fit_partition':'InnerTrain','fit_scenes':split(fold)['InnerTrain'],'fit_actors':len(train),
            'fit_indices_sha256':hashlib.sha256(train.tobytes()).hexdigest(),'graph':stats,'R2':{'mean':rmean,'std':rstd},
            'epsilon':1e-6,'std_ddof':0,'OuterTest_or_InnerDev_or_HeadDev_used':False,'split_sha256':sha256(ROOT/f'02_splits/stage11b_fold{fold}_split.json')}
        atomic_json(ROOT/f'02_splits/stage11b_fold{fold}_normalization.json',norm)
        m=fresh(fold,'cpu');m2=fresh(fold,'cpu');assert state_sha(m)==state_sha(m2)
        assert torch.equal(m.head[-1].weight,torch.zeros_like(m.head[-1].weight)) and torch.equal(m.head[-1].bias,torch.zeros_like(m.head[-1].bias))
        initial.append({'Fold':fold,'Seed':2022+100*(fold-1),'StateSHA256':state_sha(m),'ABCBitwiseShared':True,'Params':24066})
        normalization.append({'Fold':fold,'FitScenes':378,'FitActors':len(train),'SHA256':sha256(ROOT/f'02_splits/stage11b_fold{fold}_normalization.json'),'Isolation':'PASS'})
        print('FOLD_NORMALIZATION_PASS',fold,len(train),flush=True)
    assert len(set(r['StateSHA256'] for r in initial))==3
    dump('02_splits/stage11b_normalization_manifest.csv',normalization);dump('01_preflight/stage11b_initialization.csv',initial)
    summary={'Status':'PASS','HeadTrainScenes':630,'HeadTrainActors':len(f),'HeadDevActorsUsed':0,'RawWindows':windows,
        'HeadDevWindowsSkipped':skipped,'CVIsolation':'PASS','OuterSceneUnion':630,'OuterSceneMultiplicity':1,
        'GTpoisonWindows':poison,'BikeNeighborOccurrencesInPoisonSample':neighbor_bike,
        'CandidateIdentity':'PASS','CandidateMaxDiff':0,'GraphSourceSHA256':sha256(S8/'00c_sparse_type_aware_graph_spec/03_model_audit/stage8a0c_model.py'),
        'raw_R2_recomputed_without_old_normalization':True,'R2ProtocolReproducible':True,'R2HistoricalCheckpointLoaded':False,
        'PredictorSHA256':PREDICTOR_SHA,'cache_files':{p.name:sha256(p) for p in (CACHE/'r2_raw.npy',CACHE/'r2_flags.npy',CACHE/'identities.csv')}}
    atomic_json(CACHE/'prepared.json',summary);atomic_json(ROOT/'01_preflight/stage11b_data_and_poison_audit.json',summary)
    verify(history=True);print('STAGE11B_PREPARE_PASS',flush=True)
if __name__=='__main__':main()
