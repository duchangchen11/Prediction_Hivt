"""Resume per-scene extraction from the audited immutable Stage8 quota cache."""
from pathlib import Path
import sys,time,multiprocessing
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'00_manifest'),str(Path(__file__).resolve().parent)]
from stage12a_common import *
from stage12a_features import semantic_features,geometric_features
def extract_scene(task):
    scene,ids,force_replay=task
    ids=np.asarray(ids,np.int64);dest=root/f'stage12a_scene_{scene}.npz';audit=dest.with_suffix('.json')
    prior=read_json(audit) if dest.exists() and audit.exists() else None
    if prior and prior.get('FeatureCodeSHA256')==feature_sha and not force_replay:
        info=prior;assert sha256(dest)==info['SHA256'] and info['ProtocolSHA256']==protocol_sha
        with np.load(dest) as z:assert np.array_equal(z['headtrain_indices'],ids)
    else:
        src=f.iloc[ids].source_index.to_numpy();local=np.array(candidates[src],copy=True)
        loc=regions[int(meta['location'][ids[0]])];assert np.all(meta['location'][ids]==meta['location'][ids[0]])
        c=np.cos(meta['yaw'][ids]);s=np.sin(meta['yaw'][ids]);R=np.stack((c,-s,s,c),-1).reshape(-1,2,2)
        points=np.einsum('nkth,njh->nktj',local.astype(np.float64),R)+meta['origin'][ids,None,None,:]
        selection={k:np.array(meta[k][ids].reshape(-1,8),copy=True) for k in ['entity_ids','entity_types','geometry_distances']}
        selection['map_mask']=selection['entity_ids']>=0
        assert np.array_equal(selection['map_mask'],args[5][src].reshape(-1,8))
        sem,valid=semantic_features(index,points.reshape(-1,12,2),loc,selection,np.array(args[4][src].reshape(-1,8,11)))
        geom=geometric_features(local,meta['current'][ids],meta['history_heading'][ids],meta['history_heading_valid'][ids],
            f.iloc[ids],np.array(args[0][src]),np.array(args[1][src]),np.array(args[2][src]))
        gid=np.array([hashlib.sha256(a.tobytes()).hexdigest().encode() for a in local.reshape(-1,12,2)],dtype='S64').reshape(-1,6)
        optimization_diff=0.
        if prior:
            assert sha256(dest)==prior['SHA256']
            with np.load(dest) as old:
                optimization_diff=float(np.max(np.abs(old['semantic']-sem.reshape(len(ids),6,-1))))
                assert optimization_diff<1e-10 and np.array_equal(old['feature_valid'],valid.reshape(len(ids),6,-1))
                assert np.array_equal(old['geometry'],geom) and np.array_equal(old['candidate_geometry_id'],gid)
        atomic_npz(dest,headtrain_indices=ids,semantic=sem.reshape(len(ids),6,-1),feature_valid=valid.reshape(len(ids),6,-1),
            geometry=geom,entity_ids=selection['entity_ids'].reshape(-1,6,8),entity_types=selection['entity_types'].reshape(-1,6,8),
            candidate_geometry_id=gid,map_region=np.full(len(ids),regions.index(loc),np.int8))
        info={'Scene':scene,'Actors':len(ids),'Candidates':len(ids)*6,'SHA256':sha256(dest),'ProtocolSHA256':protocol_sha,
            'Path':str(dest.relative_to(ROOT)),'Region':loc,'FutureLabelsUsed':False,
            'FeatureCodeSHA256':feature_sha,'OptimizationReplayMaxDiff':optimization_diff,
            'CPUExecution':'three fork workers, one scene per task','CPUReplayAgainstSequentialCache':bool(force_replay)}
        atomic_json(audit,info)
    assert not any(str(p.relative_to(PROJECT)) in READS for p in LABEL_FILES)
    return info,ids
def main():
    global index,f,regions,root,meta,args,candidates,feature_sha,protocol_sha
    os.environ['STAGE12A_PHASE']='extract';torch.set_num_threads(1);verify()
    assert read_json(ROOT/'01_map_integrity/stage12a_preflight_gate.json')['Status']=='PASS'
    index=SparseSemanticIndex();f=observed_identities();regions=sorted(index.regions)
    root=ROOT/'02_semantic_cache/cache';root.mkdir(exist_ok=True)
    frames=ROOT/'01_map_integrity/cache'
    meta={k:np.load(frames/f'stage12a_{k}.npy',mmap_mode='r') for k in ['origin','yaw','location','current','history_heading',
        'history_heading_valid','entity_ids','entity_types','geometry_distances']}
    args={k:np.load(TRAIN/f'arg{k}.npy',mmap_mode='r') for k in (0,1,2,3,4,5)}
    candidates=np.load(S11A/'01_identity_audit/cache/candidates.npy',mmap_mode='r')
    records=[];filled=np.zeros(len(f),bool);began=time.monotonic()
    feature_sha=sha256(ROOT/'02_semantic_cache/stage12a_features.py');protocol_sha=sha256(PROTOCOL)
    tasks=[(scene,ids,j<3) for j,(scene,ids) in enumerate(f.groupby('scene_token',sort=True).indices.items())]
    # Read-only mmaps and original map objects are shared by fork; workers have
    # disjoint scene filenames. Canonical ordered results preserve the manifest.
    with multiprocessing.get_context('fork').Pool(3) as pool:
        for number,(info,ids) in enumerate(pool.imap(extract_scene,tasks,chunksize=1),1):
            records.append(info);filled[ids]=True
            if number%10==0 or number==630:print('SEMANTIC_SCENE_FROZEN',number,'/630 actors',int(filled.sum()),'seconds',round(time.monotonic()-began,1),flush=True)
    assert filled.all() and len(records)==630 and sum(r['Candidates'] for r in records)==1560906
    assert not any(str(p.relative_to(PROJECT)) in READS for p in LABEL_FILES)
    atomic_json(ROOT/'02_semantic_cache/stage12a_semantic_cache_manifest.json',{'Status':'FROZEN_ALL_630','Scenes':630,'Actors':len(f),
        'Candidates':len(f)*6,'SemanticFields':SEM_FIELDS,'GeometryFields':GEO_FIELDS,'Records':records,
        'ProtocolSHA256':sha256(PROTOCOL),'GT_FDE_ADE_LabelsUsed':False,'ForbiddenFutureLabelFileReads':0,
        'SourceCandidateSHA256':sha256(S11A/'01_identity_audit/cache/candidates.npy'),'NoCandidateOrScoreModification':True,
        'FeatureCodeSHA256':feature_sha,'OptimizationChangesDefinition':False,
        'MaxOptimizationReplayDiff':max(r.get('OptimizationReplayMaxDiff',0) for r in records),
        'CPUWorkers':3,'CPUSequentialReplayScenes':sum(r.get('CPUReplayAgainstSequentialCache',False) for r in records),
        'Seconds':time.monotonic()-began,'InvalidComponents':'same Stage8 policy; discarded one invalid Boston walkway component, no repair',
        'CoveragePopulation':'Stage11B HeadTrain630 full-horizon OOF; all six candidate map selections retained including empty context'})
    print('ALL_SEMANTIC_FEATURES_FROZEN_BEFORE_ERROR_ANALYSIS',flush=True)
if __name__=='__main__':main()
