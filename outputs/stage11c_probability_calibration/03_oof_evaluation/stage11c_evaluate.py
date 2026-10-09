"""Fixed-temperature OOF development evaluation; no selection from OOF."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'00_manifest'))
from stage11c_common import *
FIELDS=('OracleBestModeNLL','BrierScore','SoftCE','ExpectedFDE','ExpectedRegret','NormalizedExpectedRegret',
    'PredictionEntropy','Top1Probability','OracleModeProbability','Top1FDE','Top1ADE','minFDE6','minADEOracle6','MR6')
def actor_metrics(z,p,fd,ad):
    z=np.asarray(z,np.float64);p=np.asarray(p,np.float64);fd=np.asarray(fd,np.float64);ad=np.asarray(ad,np.float64)
    assert np.isfinite(z).all() and np.isfinite(p).all() and np.all(p>=0) and np.all(p<=1)
    assert np.allclose(p.sum(-1),1,atol=1e-6,rtol=0)
    top=p.argmax(-1);best=fd.argmin(-1);ii=np.arange(len(z));lp=stable_logprob(z)
    target=np.zeros_like(p);target[ii,best]=1
    q=np.exp(stable_logprob(-fd));cost=fd-fd.min(-1,keepdims=True);scale=np.maximum(1,cost.mean(-1))
    value=np.stack((-lp[ii,best],np.square(p-target).sum(-1),-(q*lp).sum(-1),
        (p*fd).sum(-1),(p*cost).sum(-1),(p*cost).sum(-1)/scale,
        -xlogy(p,p).sum(-1),p.max(-1),p[ii,best],fd[ii,top],ad[ii,top],
        fd.min(-1),ad.min(-1),(fd.min(-1)>2).astype(np.float64)),-1)
    assert np.isfinite(value).all()
    return value,top,best
def summarize(values,p,top,best,mask,group,model,fold='Pooled'):
    ece,count,confidence,correct,_=reliability(p,top,best,mask)
    conf=p[mask].max(-1);quantile=np.quantile(conf,[.5,.9,.99])
    return {'Fold':fold,'Group':group,'Model':model,'Count':int(mask.sum()),
        **dict(zip(FIELDS,values[mask].mean(0))), 'Top1ModeECE':ece,
        'Top1ProbabilityP50':float(quantile[0]),'Top1ProbabilityP90':float(quantile[1]),'Top1ProbabilityP99':float(quantile[2]),
        'FractionAbove0.9':float((conf>.9).mean()),'FractionAbove0.99':float((conf>.99).mean()),
        'OracleTop1MatchRate':float((top[mask]==best[mask]).mean())}
def exact(a,b): return a.dtype==b.dtype and a.shape==b.shape and np.ascontiguousarray(a).tobytes()==np.ascontiguousarray(b).tobytes()
def main():
    verify();fit=read_json(ROOT/'02_calibration/stage11c_temperature_fit_isolation.json');assert fit['Status']=='PASS'
    for fold in (1,2,3):
        rec=fit['Fits'][fold-1];assert rec['ProtocolSHA256']==sha256(PROTOCOL)
        assert set(rec['InnerDevSceneTokens'])==set(split(fold)['InnerDev'])
        for name,h in rec['cache_files'].items(): assert sha256(ROOT/f'02_calibration/fold{fold}/cache'/name)==h
    f=frame().copy();meta=pd.read_csv(OOF/'stage11b_oof_predictions_actor_records.csv',dtype={'future_mask_bits':str})
    assert np.array_equal(f.actor_id,meta.actor_id) and np.array_equal(f.source_index,meta.source_index)
    assert (meta.Partition=='OuterTest').all();foldid=meta.Fold.to_numpy()
    for fold in (1,2,3):
        assert set(f.loc[foldid==fold,'scene_token'])==set(split(fold)['OuterTest'])
    complete=read_json(OOF/'complete.json');assert complete['Status']=='PASS'
    for name,h in complete['cache_files'].items(): assert sha256(OOF/name)==h
    zz=np.load(OOF/'stage11b_oof_predictions_logits.npy',mmap_mode='r')
    pp=np.load(OOF/'stage11b_oof_predictions_probabilities.npy',mmap_mode='r')
    oldvalues=np.load(OOF/'stage11b_oof_metrics.npy',mmap_mode='r')
    src=f.source_index.to_numpy();fd=np.array(np.load(S8/'01_training/cache/fde.npy',mmap_mode='r')[src],copy=True)
    ad=np.array(np.load(S8/'01_training/cache/ade.npy',mmap_mode='r')[src],copy=True)
    candidates=np.load(S11A/'01_identity_audit/cache/candidates.npy',mmap_mode='r')
    candidate_sha=sha256(S11A/'01_identity_audit/cache/candidates.npy')
    bike=f.agent_type_id.to_numpy()==2;vp=~bike;n=len(f);dest=ROOT/'03_oof_evaluation/cache';dest.mkdir(exist_ok=True)
    assert not (dest/'stage11c_evaluation_complete.json').exists()
    # Never round calibrated probabilities back to FP32: close top-mode ties
    # could otherwise change the discrete decision. Bicycle outputs stay FP32.
    calz=np.asarray(zz[:,4],np.float64).copy();calp=np.asarray(pp[:,4],np.float64).copy()
    for rec in fit['Fits']:
        m=vp&(foldid==rec['Fold']);T=rec['Temperature'];assert 1<=T<=1000
        calz[m]/=T;calp[m]=np.exp(stable_logprob(calz[m]))
    rawtop=np.asarray(pp[:,4]).argmax(-1);caltop=calp.argmax(-1)
    changed=np.flatnonzero(rawtop!=caltop)
    if len(changed):
        atomic_json(ROOT/'01_frozen_audit/stage11c_top1_identity_audit.json',{'Status':'FAIL','ChangedActors':len(changed),'FirstIndices':changed[:20].tolist()})
        raise RuntimeError('STOP: any mode change requires float/tie investigation')
    assert np.array_equal(np.asarray(zz[:,4]).argmax(-1),rawtop)
    assert np.array_equal(calz.argmax(-1),rawtop)
    vpblock={'headtrain_indices':np.flatnonzero(vp),'logits':calz[vp],'probabilities':calp[vp]}
    bikeblock={'headtrain_indices':np.flatnonzero(bike),'logits':np.array(zz[bike,1],copy=True),
        'probabilities':np.array(pp[bike,1],copy=True),'candidates':np.array(candidates[src[bike]],copy=True),
        'top1_mode':rawtop[bike].copy(),'Top1FDE':np.array(oldvalues[bike,1,0],copy=True)}
    np.savez(dest/'stage11c_calibrated_vehicle_pedestrian.npz',**vpblock)
    np.savez(dest/'stage11c_bicycle_foldr2_passthrough.npz',**bikeblock)
    reloaded=np.load(dest/'stage11c_bicycle_foldr2_passthrough.npz')
    bikefields={
        'logits':exact(reloaded['logits'],np.array(zz[bike,1])) and exact(reloaded['logits'],np.array(zz[bike,4])),
        'probabilities':exact(reloaded['probabilities'],np.array(pp[bike,1])) and exact(reloaded['probabilities'],np.array(pp[bike,4])),
        'candidates':exact(reloaded['candidates'],np.array(candidates[src[bike]])),
        'top1_mode':exact(reloaded['top1_mode'],np.asarray(pp[bike,1]).argmax(-1)) and exact(reloaded['top1_mode'],rawtop[bike]),
        'Top1FDE':exact(reloaded['Top1FDE'],np.array(oldvalues[bike,1,0])) and exact(reloaded['Top1FDE'],np.array(oldvalues[bike,4,0]))}
    assert all(bikefields.values()) and int(bike.sum())==2980
    values=np.lib.format.open_memmap(dest/'stage11c_actor_probability_metrics.npy',mode='w+',dtype='float64',shape=(n,6,len(FIELDS)))
    rows=[];foldrows=[];reliabilityrows=[];histro=[];groupmasks=groups(f)
    for j,model in enumerate(MODELS):
        z,p=(np.asarray(zz[:,j]),np.asarray(pp[:,j],np.float64)) if j<5 else (calz,calp)
        val,top,best=actor_metrics(z,p,fd,ad);values[:,j]=val
        for field,oldcol in [('Top1FDE',0),('Top1ADE',1),('minFDE6',13),('minADEOracle6',12),('MR6',14)]:
            assert np.array_equal(val[:,FIELDS.index(field)],oldvalues[:,min(j,4),oldcol]),(model,field)
        for group,mask in groupmasks.items():
            rows.append(summarize(val,p,top,best,mask,group,model))
            for fold in (1,2,3):
                m=mask&(foldid==fold)
                if m.any(): foldrows.append(summarize(val,p,top,best,m,group,model,fold))
            _,count,confidence,correct,_=reliability(p,top,best,mask)
            bins=len(count)
            for b in range(bins):
                reliabilityrows.append({'Group':group,'Model':model,'Bin':b,'Lower':b/bins,'Upper':(b+1)/bins,
                    'Count':int(count[b]),'ConfidenceSum':float(confidence[b]),'CorrectSum':float(correct[b]),
                    'MeanConfidence':float(confidence[b]/count[b]) if count[b] else None,
                    'OracleTop1MatchRate':float(correct[b]/count[b]) if count[b] else None})
            hist,edges=np.histogram(p[mask].max(-1),bins=np.linspace(0,1,51))
            for b,c in enumerate(hist): histro.append({'Group':group,'Model':model,'Lower':edges[b],'Upper':edges[b+1],'Count':int(c)})
        print('OOF_METRICS_COMPLETE',model,flush=True)
    values.flush()
    identityfields=['Top1FDE','Top1ADE','minFDE6','minADEOracle6','MR6']
    identity={k:bool(np.array_equal(values[:,4,FIELDS.index(k)],values[:,5,FIELDS.index(k)])) for k in identityfields}
    assert all(identity.values())
    candidate_blocks=hashlib.sha256();bicycle_candidate_blocks=hashlib.sha256()
    # Every actor routes all six candidates to exactly the same frozen source.
    for start in range(0,n,1024):
        ix=np.arange(start,min(n,start+1024));raw=candidates[src[ix]];cal=candidates[src[ix]]
        assert exact(raw,cal) and np.isfinite(raw).all()
        candidate_blocks.update(raw.tobytes());bicycle_candidate_blocks.update(raw[bike[ix]].tobytes())
    assert sha256(S11A/'01_identity_audit/cache/candidates.npy')==candidate_sha
    topaudit={'Status':'PASS','Top1Identity':'PASS','Actors':n,'Scenes':630,'ChangedModes':0,
        'RawProbabilityAndLogitArgmaxEqual':True,'CalibratedProbabilityAndLogitArgmaxEqual':True,
        'TieRule':'lowest index, identical raw/cal','CandidateCoordinatesExact':True,'CandidateMaxDiff':0,
        'HeadTrainCandidateBlockSHA256':candidate_blocks.hexdigest(),'CandidateSourceSHA256Before':candidate_sha,
        'CandidateSourceSHA256After':candidate_sha,'MetricIdentity':identity,'MetricMaxDiff':{k:0 for k in identityfields}}
    bikeaudit={'Status':'PASS','BicyclePreserved':'YES','Actors':int(bike.sum()),'BitwiseFields':bikefields,
        'CandidateBlockSHA256':array_sha(reloaded['candidates']),
        'LogitsSHA256':array_sha(reloaded['logits']),'ProbabilitiesSHA256':array_sha(reloaded['probabilities']),
        'LogitDType':str(reloaded['logits'].dtype),'ProbabilityDType':str(reloaded['probabilities'].dtype),
        'Routing':'original frozen FoldR2 FP32 block, no temperature scaling or softmax recomputation'}
    atomic_json(ROOT/'01_frozen_audit/stage11c_top1_identity_audit.json',topaudit)
    atomic_json(ROOT/'01_frozen_audit/stage11c_bicycle_identity_audit.json',bikeaudit)
    dump('07_tables/stage11c_oof_probability_metrics.csv',rows)
    dump('07_tables/stage11c_type_metrics.csv',[r for r in rows if r['Group'] in TYPES])
    dump('07_tables/stage11c_fold_metrics.csv',foldrows)
    diagfields=['Fold','Group','Model','Count','PredictionEntropy','Top1Probability','Top1ProbabilityP50',
        'Top1ProbabilityP90','Top1ProbabilityP99','FractionAbove0.9','FractionAbove0.99','Top1ModeECE','OracleTop1MatchRate']
    dump('05_probability_diagnostics/stage11c_probability_diagnostics.csv',[{k:r[k] for k in diagfields} for r in rows])
    dump('05_probability_diagnostics/stage11c_reliability_bins.csv',reliabilityrows)
    dump('05_probability_diagnostics/stage11c_probability_histograms.csv',histro)
    f['Fold']=foldid;f['Partition']='OuterTest (OOF development evidence)';f.to_csv(dest/'stage11c_actor_records.csv',index=False)
    np.save(dest/'stage11c_calibrated_top1_probability.npy',calp.max(-1))
    np.save(dest/'stage11c_calibrated_oracle_event.npy',(caltop==fd.argmin(-1)).astype(np.int8))
    atomic_json(dest/'stage11c_evaluation_complete.json',{'Status':'PASS','fields':FIELDS,'models':MODELS,
        'ProbabilitySumMaxAbsErrorRaw':float(np.max(np.abs(pp.astype(np.float64).sum(-1)-1))),
        'ProbabilitySumMaxAbsErrorCalibratedVP':float(np.max(np.abs(calp[vp].sum(-1)-1))),
        'AllFinite':True,'Top1Identity':'PASS','BicyclePreserved':'YES','ProtocolSHA256':sha256(PROTOCOL),
        'TemperaturesSHA256':sha256(ROOT/'02_calibration/stage11c_temperature_values.csv'),
        'cache_files':{p.name:sha256(p) for p in dest.glob('*') if p.suffix in ('.npy','.npz','.csv')}})
    verify();print('OOF_GEOMETRY_TOP1_BICYCLE_NUMERICAL_PASS',flush=True)
if __name__=='__main__':main()
