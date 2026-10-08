"""Reuse HeadDev outputs; infer missing frozen scores on TRAIN700, never optimize."""
from stage11a_common import *

@torch.no_grad()
def main():
    seed(); verify(); assert read_json(CACHE/'prepared.json')['status']=='PASS'
    assert not (CACHE/'inference_complete.json').exists()
    f=metadata(); n=len(f); store=legacy8.Store('train'); own=legacy10.Store()
    r2base=np.load(S9/'01_training/cache/r2_logits.npy',mmap_mode='r')
    context=np.load(S9/'01_training/cache/current_context.npy',mmap_mode='r')
    candidates=np.load(CACHE/'candidates.npy',mmap_mode='r')
    source=S10/'03_evaluation/cache/stage10a_mode_outputs.npz'
    saved=np.load(source); di=np.flatnonzero(f.HeadTrain.to_numpy()==0)
    assert np.array_equal(di,saved['source_index'])
    for name,h in read_json(S9/'01_training/cache/manifest.json')['files'].items():
        assert sha256(S9/'01_training/cache'/name)==h
    logits=np.lib.format.open_memmap(CACHE/'model_logits.npy',mode='w+',dtype='float32',shape=(n,len(MODELS),6))
    probabilities=np.lib.format.open_memmap(CACHE/'model_probabilities.npy',mode='w+',dtype='float32',shape=(n,len(MODELS),6))
    result=np.lib.format.open_memmap(CACHE/'actor_metrics.npy',mode='w+',dtype='float32',shape=(n,len(MODELS),len(METRICS)))
    logits[:]=np.nan;probabilities[:]=np.nan;result[:]=np.nan
    models=make_models(); before={k:state_sha(m) for k,m in models.items()}; filled=np.zeros(n,bool); anchor_maxdiff=0.
    for partition,ids in (('HeadTrain',np.flatnonzero(f.HeadTrain.to_numpy()==1)),('HeadDev',di)):
        for start in range(0,len(ids),128):
            ix=ids[start:start+128]; args,fde,ade=store.batch(ix); feature=own.r2_batch(ix)
            candidate=torch.from_numpy(np.array(candidates[ix],copy=True)).cuda(); candidate_before=tensor_sha(candidate)
            z={};p={}
            if partition=='HeadDev':
                for name in ('R0','R2','G1','DualExpert'):
                    z[name]=torch.from_numpy(saved[name+'_logits'][start:start+len(ix)].copy()).cuda()
                    p[name]=torch.from_numpy(saved[name+'_prob'][start:start+len(ix)].copy()).cuda()
                assert torch.equal(z['R0'],args[6])
            else:
                z['R0']=args[6];p['R0']=args[6].softmax(-1)
                base=models['R2'](feature,args[6]);z['R2']=base['mode_logits'];p['R2']=base['mode_prob']
                out=models['G1'](*args);z['G1']=out['mode_logits'];p['G1']=out['mode_prob']
                out=models['DualExpert'](*args[:3],args[6],feature,candidate)
                z['DualExpert']=out['mode_logits'];p['DualExpert']=out['mode_prob']
                assert out['candidates'] is candidate
            out=models['G3'](*args);z['G3']=out['mode_logits'];p['G3']=out['mode_prob']
            oldbase=torch.from_numpy(np.array(r2base[ix],copy=True)).cuda()
            anchor_maxdiff=max(anchor_maxdiff,float((oldbase-z['R2']).abs().max()))
            gate=legacy9.gate_features(args[0],torch.from_numpy(np.array(context[ix],copy=True)).cuda())
            for name in ('T2','T3'):
                out=models[name](*args[:3],oldbase,gate);z[name]=out['mode_logits'];p[name]=out['mode_prob']
            for name,t in (('VehicleExpert',0),('PedestrianExpert',1)):
                valid=torch.as_tensor(own.types[ix]==t,device='cuda')
                z[name]=z['DualExpert'].clone();p[name]=p['DualExpert'].clone()
                z[name][~valid]=float('nan');p[name][~valid]=float('nan')
            bike=torch.as_tensor(own.types[ix]==2,device='cuda')
            assert torch.equal(z['DualExpert'][bike],z['R2'][bike]) and torch.equal(p['DualExpert'][bike],p['R2'][bike])
            assert not candidate.requires_grad and tensor_sha(candidate)==candidate_before
            for mi,name in enumerate(MODELS):
                valid=torch.as_tensor(applicable(f.iloc[ix],name),device='cuda')
                assert torch.isfinite(z[name][valid]).all() and torch.isfinite(p[name][valid]).all()
                assert torch.allclose(p[name][valid].sum(-1),torch.ones(int(valid.sum()),device='cuda'),atol=1e-6,rtol=0.)
                logits[ix,mi]=z[name].cpu().numpy();probabilities[ix,mi]=p[name].cpu().numpy()
                if bool(valid.any()):
                    mm=logit_metrics(z[name][valid],p[name][valid],fde[valid],ade[valid])
                    assert torch.isfinite(mm).all(); result[ix[valid.cpu().numpy()],mi]=mm.cpu().numpy()
            filled[ix]=True
            if (start//128+1)%200==0:print('FROZEN_INFERENCE',partition,start+len(ix),'/',len(ids),flush=True)
    assert filled.all() and anchor_maxdiff<1e-6; freeze_audit(models,before);verify()
    logits.flush();probabilities.flush();result.flush()
    # Historical split metrics: R0/R2/G1/G3 plus Stage9 and Stage10 retained.
    checks=[]; headdev=di; values=np.asarray(result[headdev])
    def compare(name,metric,reference,source,mask=None):
        selected=np.ones(len(headdev),bool) if mask is None else np.asarray(mask)
        actual=float(values[selected,MODELS.index(name),METRICS.index(metric)].astype(np.float64).mean())
        error=abs(actual-reference);checks.append({'Model':name,'Metric':metric,'Count':int(selected.sum()),
            'HistoricalValue':reference,'RecomputedValue':actual,'AbsDiff':error,'Tolerance':1e-6,'Status':'PASS' if error<1e-6 else 'FAIL','Source':source})
        assert error<1e-6,(name,metric,error)
    history=read_json(S6/'03_training/stage6a_r2_summary.json')
    for name,old in (('R0',history['headdev_step0']),('R2',history['selected_HeadDev_metrics'])):
        for m,k in (('SoftCE','rank_loss'),('Top1ADE','Top1ADE'),('Top1FDE','Top1FDE'),('HitRate','HitRate')):
            compare(name,m,old[k],'Stage6 frozen HeadDev summary')
    for name in ('G1','G3'):
        old=read_json(S8/'01_training'/name/'formal/stage8a1_summary.json')['selected']
        for m,k in (('SoftCE','dev_loss'),('Top1ADE','dev_Top1ADE'),('Top1FDE','dev_Top1FDE'),('Regret','dev_OracleGap')):
            compare(name,m,old[k],'Stage8 frozen HeadDev summary')
    for name in ('T2','T3'):
        old=read_json(S9/'01_training'/name/'formal/stage9a_training_summary.json')['selected']
        compare(name,'SoftCE',old['OverallDevSoftCE'],'Stage9 frozen HeadDev summary')
        compare(name,'Top1FDE',old['OverallDevTop1FDE'],'Stage9 frozen HeadDev summary')
        for typ in TYPES:
            mask=f.iloc[di].agent_type==typ
            compare(name,'SoftCE',old[typ+'DevLoss'],'Stage9 frozen type HeadDev summary',mask)
            compare(name,'Top1FDE',old[typ+'DevTop1FDE'],'Stage9 frozen type HeadDev summary',mask)
    for name,e,typ in (('VehicleExpert','vehicle','Vehicle'),('PedestrianExpert','pedestrian','Pedestrian')):
        old=read_json(S10/'01_training'/e/'formal/stage10a_training_summary.json')['selected'];mask=f.iloc[di].agent_type==typ
        for m,k in (('SoftCE','DevSoftCE'),('Top1ADE','DevTop1ADE'),('Top1FDE','DevTop1FDE')):
            compare(name,m,old[k],'Stage10 frozen expert HeadDev summary',mask)
    stage10=pd.read_csv(S10/'06_tables/stage10a_type_results.csv')
    for _,r in stage10[stage10.Model=='DualExpert'].iterrows():
        mask=None if r.Group=='Overall' else f.iloc[di].agent_type==r.Group
        for m,k in (('SoftCE','SoftCE'),('Top1ADE','Top1ADE'),('Top1FDE','Top1FDE'),('Regret','OracleGap')):
            compare('DualExpert',m,r[k],'Stage10 frozen type HeadDev table',mask)
    write_csv(ROOT/'01_identity_audit/stage11a_historical_reproduction.csv',checks)
    rows=read_json(ROOT/'01_identity_audit/stage11a_source_identity.json')
    summary={'status':'PASS','targets':n,'HeadTrain':260151,'HeadDev':29934,'models':MODELS,
        'HeadDev_existing_outputs_reused':['R0','R2','G1','DualExpert','VehicleExpert via Vehicle route','PedestrianExpert via Pedestrian route'],
        'missing_scores_frozen_forward':['G3','T2','T3','HeadTrain G1/DualExpert/R2'],
        'R2_Historical512_vs_common128_logit_maxdiff':anchor_maxdiff,'expert_scope':'own target type only',
        'model_states_unchanged':True,'models_eval':True,'model_parameter_gradients':0,'optimizer_updates':0,'new_checkpoints':0,
        'candidate_maxdiff':0.,'official_VAL_opened':False,'test_opened':False,'historical_reproduction_checks':len(checks),
        'files':{p.name:sha256(p) for p in (CACHE/'model_logits.npy',CACHE/'model_probabilities.npy',CACHE/'actor_metrics.npy')}}
    atomic_json(CACHE/'inference_complete.json',summary);atomic_json(ROOT/'01_identity_audit/stage11a_frozen_forward_audit.json',summary)
    print('STAGE11_FROZEN_SCORE_AND_REPRODUCTION_PASS',summary,flush=True)

if __name__=='__main__': main()
