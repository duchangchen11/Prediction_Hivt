"""Bounded-process feature cache, independent label sidecar, train-only moments."""
from stage8a1_common import *
import multiprocessing as mp

def initialize():
    global INDEX
    torch.set_num_threads(1);INDEX=SparseSemanticIndex()

def worker(rec):
    root=ROOT/'01_training/cache';dest=root/(Path(rec['path']).stem+'.pt')
    if dest.exists():return str(dest)
    source=S6/rec['source_path'];assert sha256(source)==rec['source_sha256']
    assert sha256(C/rec['path'])==rec['sha256']
    frames=ROOT/'02_graph_cache'/rec['frame_path'];assert sha256(frames)==rec['frame_sha256']
    cache=torch.load(source,map_location='cpu',weights_only=False);split=read_json(SPLIT)
    blocks=[[] for _ in range(7)];fd=[];ad=[];part=[];rows=[]
    with np.load(C/rec['path']) as sel,np.load(frames) as frame:
        for j,w in enumerate(cache['windows']):
            obs,selected,node,idx,keep,feature=graph_window(w,rec,j,INDEX,sel,frame)
            # GT/future/target masks are first accessed AFTER observable graph.
            targets=torch.where(w['target_mask']&w['future_mask'].all(-1))[0]
            if not len(targets):continue
            assert not obs.predicted.requires_grad
            edge=interaction_edges(obs,idx,keep,targets)
            args=pack_targets(node,idx,keep,edge,selected,feature,targets)
            for k,a in enumerate(args):blocks[k].append(a)
            dist=(obs.predicted[targets]-w['GT'][targets,None]).norm(dim=-1)
            fd.append(dist[:,:,-1]);ad.append(dist.mean(-1))
            istrain=w['scene_token'] in split['HeadTrain'];assert istrain or w['scene_token'] in split['HeadDev']
            part.extend([int(istrain)]*len(targets))
            for t in targets.tolist():rows.append({'scene_token':w['scene_token'],'sample_token':w['sample_token'],
                'instance_token':w['instance_tokens'][t],'dataset_index':w['dataset_index'],'node_in_graph':t,'HeadTrain':int(istrain)})
    args=[torch.cat(b) for b in blocks]
    assert all(torch.isfinite(a).all() for a in args)
    atomic_torch(dest,{'args':args,'fde':torch.cat(fd),'ade':torch.cat(ad),'partition':np.array(part,dtype=np.int8),'rows':rows,
        'source_sha256':rec['source_sha256'],'selector_sha256':rec['sha256']})
    return str(dest)

def merge(paths):
    root=ROOT/'01_training/cache';n=290085
    args=[np.lib.format.open_memmap(root/f'arg{k}.npy',mode='w+',dtype=DTYPES[k],shape=(n,)+shape) for k,shape in enumerate(SHAPES)]
    fde=np.lib.format.open_memmap(root/'fde.npy',mode='w+',dtype='float32',shape=(n,6))
    ade=np.lib.format.open_memmap(root/'ade.npy',mode='w+',dtype='float32',shape=(n,6))
    partition=np.lib.format.open_memmap(root/'partition.npy',mode='w+',dtype='int8',shape=(n,))
    sums={(k,c):[0,0.,0.] for k,cc in CONT.items() for c in cc};offset=0
    with (root/'identities.csv').open('w') as fh:
        writer=None
        for path in paths:
            b=torch.load(path,map_location='cpu',weights_only=False);m=len(b['fde']);sl=slice(offset,offset+m)
            for k,a in enumerate(b['args']):args[k][sl]=a.numpy()
            fde[sl]=b['fde'].numpy();ade[sl]=b['ade'].numpy();partition[sl]=b['partition']
            if writer is None:writer=csv.DictWriter(fh,fieldnames=list(b['rows'][0]));writer.writeheader()
            writer.writerows(b['rows']);tr=torch.from_numpy(b['partition']==1)
            aa=tuple(a[tr] for a in b['args'])
            for (k,c),mask in valid_columns(aa).items():
                values=aa[k][...,c][mask].double();s=sums[k,c];s[0]+=len(values);s[1]+=float(values.sum());s[2]+=float(values.square().sum())
            offset+=m
    assert offset==n
    for a in [*args,fde,ade,partition]:a.flush()
    stats={str(k):{} for k in CONT}
    for (k,c),(count,total,sq) in sums.items():
        mu=total/count;std=float(np.sqrt(max(0.,sq/count-mu*mu)))
        stats[str(k)][str(c)]={'count':count,'mean':mu,'std':std,'normalized_mean':0.,'normalized_std':std/(std+1e-6),'constant':std<1e-10}
    atomic_json(NORM,{'status':'PASS','fit_scenes':630,'fit_partition':'HeadTrain only',
        'split_sha256':sha256(SPLIT),'statistics':stats,'continuous_columns':CONT,'epsilon':1e-6,'std_ddof':0,
        'population':'valid packed full-target contexts; neighbor nodes counted per target occurrence; each valid mode edge counted once per target',
        'conditional_validity':'node valid target/neighbor; interaction real neighbor; tangent/heading lane-only; polygon-inside polygon-only; all padding stays zero',
        'unchanged':'type/semantic one-hot, binary flags, valid flags, original output logits; nonlane semantic9/heading/tangent remain0',
        'HeadDev_used':False,'VAL_used':False,'finite':True})
    audit_sums={(k,c):[0,0.,0.] for k,cc in CONT.items() for c in cc}
    for start in range(0,n,2048):
        ids=np.arange(start,min(start+2048,n));ids=ids[partition[ids]==1]
        aa=tuple(torch.from_numpy(np.array(a[ids],copy=True)) for a in args);norm=normalize_args(aa,stats)
        for (k,c),mask in valid_columns(aa).items():
            v=norm[k][...,c][mask].double();s=audit_sums[k,c];s[0]+=len(v);s[1]+=float(v.sum());s[2]+=float(v.square().sum())
    measured=[]
    for (k,c),(count,total,sq) in audit_sums.items():
        mu=total/count;std=float(np.sqrt(max(0.,sq/count-mu*mu)));expected=stats[str(k)][str(c)]['normalized_std']
        assert abs(mu)<1e-5 and abs(std-expected)<1e-5,(k,c,mu,std)
        measured.append({'Tensor':k,'Column':c,'Count':count,'Mean':mu,'Std':std,'ExpectedStd':expected})
    write_csv(ROOT/'06_tables/stage8a1_normalization_audit.csv',measured)
    atomic_json(root/'manifest.json',{'status':'PASS','targets':n,'HeadTrain':int((partition==1).sum()),'HeadDev':int((partition==0).sum()),
        'normalization_sha256':sha256(NORM),'official_VAL_loaded':False,'graph_source_frozen':True,
        'files':{p.name:sha256(p) for p in [*sorted(root.glob('*.npy')),root/'identities.csv']}})

def register():
    verify();dest=ROOT/'00_manifest/stage8a1_registration.json'
    assert not FROZEN.exists()
    initial={v:fresh(v,'cpu') for v in PARAMS}
    for block in ('node_encoder','norm','head'):
        a=getattr(initial['G1'],block).state_dict()
        for v in ('G2','G3'):assert all(torch.equal(x,getattr(initial[v],block).state_dict()[k]) for k,x in a.items())
    atomic_json(dest,{'status':'REGISTERED_BEFORE_TRAINING','base_commit':BASE,'graph_spec_sha256':sha256(SPEC),
        'split_sha256':sha256(SPLIT),'seeds':{'common':2022,'interaction':2123,'map':2124},'params':PARAMS,
        'initial_state_sha256':{v:state_sha(m) for v,m in initial.items()},'initialization_shared_bitwise':True,
        'architecture_fidelity':'0C raw15+raw15+17 interaction value input47; hidden64+hidden64+17 attention input145; required exact parameter counts preserved',
        'optimizer':'AdamW','lr':.001,'weight_decay':.0001,'max_epochs':50,'patience':5,'selection':'minimum HeadDev ranking loss; strict improvement',
        'microbatch':128,'accumulation':8,'effective_batch':1024,
        'epoch_remainder':'continuous no-replacement shuffled epoch stream; carry final remainder forward; dev at epoch boundary; no smaller optimizer step; final incomplete gradient discarded',
        'tiny':{'count':128,'optimizer_updates':300,'FP32':True,'pass':'final_loss<0.8*initial_loss; finite nonzero delta; frozen predictor gradients0'},
        'normalization_sha256':sha256(NORM),'feature_manifest_sha256':sha256(ROOT/'01_training/cache/manifest.json'),
        'training_sequence':['G1','G2','G3'],'temperature_m':1.,'checkpoint_freeze_before_VAL':True,'test_used':False,
        'map_groups':'actor-level fixed over all6 modes: MapNonEmpty=any real map edge; ZeroMap=no real map edge in anymode; RouteCenterlineAvailable=Vehicle/Bicycle any lane/connector; PedSemanticSpecificAvailable=Pedestrian any crossing/walkway',
        'interaction_distance_bins_m':[0,2,5,10,20,50,'inf'],
        'bootstrap':{'seed':2022,'replicates':1000,'unit':'whole scene','shared_resampling':True,'percentiles':[2.5,97.5]},
        'decisions':'Exact user Stage8A-1 rules sections43-51; negative FDE delta improves; reliable harm means CI_lower>0; overlapping subgroups descriptive/unadjusted',
        'recommended_model':'G3 if PaperUsableFSCG YES; otherwise G1 if G1-R2 Overall CI_upper<0 and no reliable Vehicle/Pedestrian harm; otherwise R2',
        'new_sources_sha256':{p.name:sha256(p) for p in sorted(Path(__file__).parent.glob('stage8a1_*.py'))}})

def main():
    seed();verify();root=ROOT/'01_training/cache'
    if not (root/'manifest.json').exists():
        records=[r for r in read_json(C/'02_graph_cache/stage8a0c_selector_manifest.json')['batches'] if r['split']=='train']
        paths=[]
        with mp.get_context('spawn').Pool(2,initializer=initialize) as pool:
            for i,p in enumerate(pool.imap(worker,records,chunksize=1)):
                paths.append(p)
                if (i+1)%25==0:print('TRAIN_FEATURES',i+1,'/',len(records),flush=True)
        merge(paths)
    register();print('PREPARE_PASS',read_json(root/'manifest.json')['HeadTrain'],flush=True)

if __name__=='__main__':main()
