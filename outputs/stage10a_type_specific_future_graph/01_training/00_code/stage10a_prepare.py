"""Exact TRAIN identities and read-only frozen R2 feature join; no VAL access."""
from stage10a_common import *
import pandas as pd

@torch.no_grad()
def main():
    seed(); verify(); assert not REG.exists()
    dest=ROOT/'01_training/cache'; manifest=dest/'manifest.json'
    assert not manifest.exists(), 'Preserve prepared cache; do not recreate'
    ids=pd.read_csv(S8/'01_training/cache/identities.csv'); n=len(ids); assert n==290085
    keys=['scene_token','sample_token','instance_token']
    lookup={tuple(getattr(r,k) for k in keys):i for i,r in enumerate(ids.itertuples())}; assert len(lookup)==n
    features=np.lib.format.open_memmap(dest/'r2_features.npy',mode='w+',dtype='float32',shape=(n,6,19))
    fde=np.load(S8/'01_training/cache/fde.npy',mmap_mode='r'); ade=np.load(S8/'01_training/cache/ade.npy',mmap_mode='r')
    logits=np.load(S8/'01_training/cache/arg6.npy',mmap_mode='r'); node=np.load(S8/'01_training/cache/arg0.npy',mmap_mode='r')
    partition=np.load(S8/'01_training/cache/partition.npy',mmap_mode='r'); filled=np.zeros(n,bool)
    fm=read_json(S6/'02_features/stage6a_feature_manifest.json'); metadata=[None]*n; counts=[]
    split=read_json(SPLIT); train=set(split['HeadTrain']); dev=set(split['HeadDev'])
    assert len(train)==630 and len(dev)==70 and not train&dev
    for name,flag in (('headtrain',1),('headdev',0)):
        rec=fm['partitions'][name]; p=S6/rec['path']; assert sha256(p)==rec['sha256']
        d=torch.load(p,map_location='cpu',weights_only=False)
        assert d['full_horizon_only'] and d['split_sha256']==sha256(SPLIT)
        ix=np.array([lookup[tuple(r[k] for k in keys)] for r in d['rows']])
        assert not filled[ix].any() and np.all(partition[ix]==flag)
        assert np.array_equal(fde[ix],d['FDE_by_mode'].numpy()) and np.array_equal(ade[ix],d['ADE_by_mode'].numpy())
        assert np.array_equal(logits[ix],d['base_logits'].numpy()) and np.array_equal(node[ix,0,:,3],d['base_logits'].numpy())
        features[ix]=d['features_R2'].numpy(); filled[ix]=True
        for j,r in zip(ix,d['rows']):
            assert r['horizon']=='full_horizon'
            assert r['node_in_graph']==int(ids.iloc[j].node_in_graph) and r['dataset_index']==int(ids.iloc[j].dataset_index)
            label=CLASSES[int(node[j,0,0,:3].argmax())]
            assert r['agent_type']==label.lower() and r['agent_type_id']==CLASSES.index(label)
            metadata[j]={'source_index':int(j),**r,'agent_type':label,'HeadTrain':int(flag)}
        for c in CLASSES: counts.append({'Partition':'HeadTrain' if flag else 'HeadDev','ActorType':c,'Count':sum(r['agent_type']==c.lower() for r in d['rows'])})
        print('JOIN',name,len(ix),flush=True)
        del d
    assert filled.all() and np.isfinite(features).all() and set(ids.scene_token)==train|dev
    assert np.array_equal(partition,np.array([int(s in train) for s in ids.scene_token]))
    features.flush(); write_csv(dest/'identities.csv',metadata); write_csv(ROOT/'06_tables/stage10a_target_counts.csv',counts)
    rec={'status':'PASS','targets':n,'HeadTrain':260151,'HeadDev':29934,'counts':counts,'official_VAL_loaded':False,'semantic_arrays_opened':False,
        'join':'exact scene/sample/instance/node/window/full-horizon identity; FDE/ADE/original logits bitwise equal',
        'source_future_manifest_sha256':sha256(S8/'01_training/cache/manifest.json'),'R2_feature_manifest_sha256':sha256(S6/'02_features/stage6a_feature_manifest.json'),
        'split_sha256':sha256(SPLIT),'normalization_sha256':sha256(NORM),'files':{p.name:sha256(p) for p in (dest/'r2_features.npy',dest/'identities.csv')}}
    atomic_json(manifest,rec); atomic_json(ROOT/'00_manifest/stage10a_feature_cache_manifest.json',rec)
    print('STAGE10_PREPARE_PASS',counts,flush=True)

if __name__=='__main__': main()
