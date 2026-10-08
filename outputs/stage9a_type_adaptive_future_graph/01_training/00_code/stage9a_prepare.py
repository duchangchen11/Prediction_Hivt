"""Read-only TRAIN cache join for frozen R2 and t0 gate context; no VAL loads."""
from stage9a_common import *
import pandas as pd

@torch.no_grad()
def main():
    seed();verify();dest=ROOT/'01_training/cache';manifest=dest/'manifest.json'
    if manifest.exists():print('PRESERVE_PREPARED_CACHE',flush=True);return
    ids=pd.read_csv(S8/'01_training/cache/identities.csv');n=len(ids);assert n==290085
    key=lambda r:(r.scene_token,r.sample_token,r.instance_token)
    lookup={key(r):i for i,r in enumerate(ids.itertuples())};assert len(lookup)==n
    base=np.lib.format.open_memmap(dest/'r2_logits.npy',mode='w+',dtype='float32',shape=(n,6))
    context=np.lib.format.open_memmap(dest/'current_context.npy',mode='w+',dtype='float32',shape=(n,2))
    filled=np.zeros(n,bool);r2=frozen_r2();r2before=state_sha(r2)
    features=read_json(S6/'02_features/stage6a_feature_manifest.json')
    source_fde=np.load(S8/'01_training/cache/fde.npy',mmap_mode='r');source_node=np.load(S8/'01_training/cache/arg0.npy',mmap_mode='r')
    for part in ('headtrain','headdev'):
        rec=features['partitions'][part];p=S6/rec['path'];assert sha256(p)==rec['sha256']
        d=torch.load(p,map_location='cpu',weights_only=False)
        ix=np.array([lookup[(r['scene_token'],r['sample_token'],r['instance_token'])] for r in d['rows']])
        assert np.array_equal(source_fde[ix],d['FDE_by_mode'].numpy())
        assert np.array_equal(source_node[ix,0,:,3],d['base_logits'].numpy())
        for start in range(0,len(ix),512):
            end=min(start+512,len(ix));out=r2(d['features_R2'][start:end].cuda(),d['base_logits'][start:end].cuda())
            base[ix[start:end]]=out['mode_logits'].cpu().numpy()
        del d
        print('R2_BASE',part,len(ix),flush=True)
    records=[r for r in read_json(S6/'01_cache/stage6a_cache_manifest.json')['batches'] if r['split']=='train']
    for bi,rec in enumerate(records):
        p=S6/rec['path'];assert sha256(p)==rec['sha256']
        payload=torch.load(p,map_location='cpu',weights_only=False)
        for w in payload['windows']:
            # Current graph context constructed before any supervision access.
            obs=observable_window(w,'',np.zeros(2),0.);idx,keep=neighbors(obs)
            targets=torch.where(w['target_mask']&w['future_mask'].all(-1))[0]
            ix=np.array([lookup[(w['scene_token'],w['sample_token'],w['instance_tokens'][t])] for t in targets.tolist()],dtype=np.int64)
            if not len(ix):continue
            assert not filled[ix].any();context[ix]=context_from_window(w,idx,keep,targets).numpy();filled[ix]=True
        if (bi+1)%50==0:print('CURRENT_CONTEXT',bi+1,'/',len(records),flush=True)
    assert filled.all() and np.isfinite(base).all() and np.isfinite(context).all()
    assert state_sha(r2)==r2before and not any(p.grad is not None or p.requires_grad for p in r2.parameters())
    base.flush();context.flush()
    atomic_json(manifest,{'status':'PASS','targets':n,'HeadTrain':260151,'HeadDev':29934,'official_VAL_loaded':False,
        'semantic_features_opened':False,'R2_gradient_count':0,'R2_state_unchanged':True,
        'R2_features_manifest_sha256':sha256(S6/'02_features/stage6a_feature_manifest.json'),
        'Stage8_future_cache_manifest_sha256':sha256(S8/'01_training/cache/manifest.json'),
        'files':{p.name:sha256(p) for p in (dest/'r2_logits.npy',dest/'current_context.npy')},
        'join':'exact scene/sample/instance identities; FDE and original logits bitwise checked'})
    print('STAGE9_PREPARE_PASS',flush=True)

if __name__=='__main__':main()
