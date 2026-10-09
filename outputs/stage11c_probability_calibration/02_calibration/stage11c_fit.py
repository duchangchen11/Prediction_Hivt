"""Exactly three scalar temperatures, fitted without access to OOF logits."""
from pathlib import Path
import sys, os
os.environ['STAGE11C_PHASE']='fit'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'00_manifest'))
from stage11c_common import *
from scipy.optimize import minimize_scalar
def fit_one(z,fde,typ,options):
    z=np.asarray(z,np.float64);best=fde.argmin(-1);ii=np.arange(len(z))
    v=typ==0;p=typ==1;assert v.any() and p.any() and (v|p).all()
    def loss(T):
        nll=-stable_logprob(z/T)[ii,best]
        return float(.5*nll[v].mean()+.5*nll[p].mean())
    lower,upper=options['bounds']
    result=minimize_scalar(lambda logT:loss(float(np.exp(logT))),bounds=(np.log(lower),np.log(upper)),
        method='bounded',options={'xatol':options['xatol'],'maxiter':options['maxiter']})
    assert result.success and np.isfinite(result.fun)
    candidates=[(loss(lower),lower),(loss(upper),upper),(float(result.fun),float(np.exp(result.x)))]
    objective,T=min(candidates)
    scaled=z/T;lp=stable_logprob(scaled);probs=np.exp(lp)
    derivative=scaled[ii,best]-(probs*scaled).sum(-1)
    macro_derivative=float(.5*derivative[v].mean()+.5*derivative[p].mean())
    assert lower<=T<=upper and np.isfinite(probs).all() and np.allclose(probs.sum(-1),1,rtol=0,atol=1e-12)
    assert np.array_equal(z.argmax(-1),scaled.argmax(-1)) and np.array_equal(z.argmax(-1),probs.argmax(-1))
    if lower<T<upper: assert abs(macro_derivative)<1e-6,macro_derivative
    elif T==upper: assert macro_derivative<=1e-6
    elif T==lower: assert macro_derivative>=-1e-6
    return T,{'NLLRawMacro':loss(1.),'NLLCalibratedMacro':objective,'MacroDerivativeLogT':macro_derivative,
        'Boundary':'BOUNDARY_HIT' if T==upper else ('LOWER_BOUNDARY' if T==lower else 'INTERIOR'),
        'OptimizerSuccess':bool(result.success),'OptimizerEvaluations':int(result.nfev),
        'OptimizerMessage':str(result.message),'EndpointNLL_T1':loss(lower),'EndpointNLL_T1000':loss(upper)}
@torch.inference_mode()
def main():
    seed();frozen=verify();protocol=read_json(PROTOCOL);rows=[];devrows=[];fits=[]
    assert read_json(REG)['Status']=='REGISTERED_BEFORE_FIT'
    assert not (ROOT/'02_calibration/stage11c_temperature_values.csv').exists()
    for fold in (1,2,3):
        store=DevStore(fold);ids=store.allowed;f=frame().iloc[ids]
        s=split(fold);assert set(f.scene_token)==set(s['InnerDev']) and not set(f.scene_token)&set(s['OuterTest'])
        cp=S11B/f'04_checkpoints/fold{fold}/C_best.pt'
        expected=next(r for r in frozen['Stage11BCheckpoints'] if r['Fold']==fold and r['Model']=='C')
        assert sha256(cp)==expected['SHA256']
        saved=torch.load(cp,map_location='cpu',weights_only=False)
        assert saved['Fold']==fold and saved['Model']=='C'
        assert saved['normalization_sha256']==sha256(store.normpath)
        assert saved['split_sha256']==sha256(S11B/f'02_splits/stage11b_fold{fold}_split.json')
        model=SparseGraphReranker('G1',seed=2022+100*(fold-1)).cuda()
        model.load_state_dict(saved['state_dict']);model.eval().requires_grad_(False)
        assert sum(p.numel() for p in model.parameters())==24066
        state_before=state_sha(model);zs=[];ps=[];fds=[]
        for start in range(0,len(ids),128):
            ix=ids[start:start+128];args,fd=store.batch(ix)
            out=model(*args);zs.append(out['mode_logits'].cpu().numpy());ps.append(out['mode_prob'].cpu().numpy());fds.append(fd)
        z=np.concatenate(zs);p=np.concatenate(ps);fd=np.concatenate(fds);typ=f.agent_type_id.to_numpy()
        keep=(typ==0)|(typ==1);assert np.array_equal(z[keep].argmax(-1),p[keep].argmax(-1))
        top=p.argmax(-1);topfde=fd[np.arange(len(fd)),top]
        means={t:float(topfde[typ==j].astype(np.float64).mean()) for j,t in enumerate(TYPES[:2])}
        replay_score=.5*sum(means[t]/saved['R2_DevReferences'][t] for t in TYPES[:2])
        assert abs(replay_score-saved['Score'])<1e-12,(fold,replay_score,saved['Score'])
        state_after=state_sha(model);assert state_before==state_after
        assert all(not p.requires_grad and p.grad is None for p in model.parameters())
        z,fd,typ=z[keep],fd[keep],typ[keep];kept=ids[keep]
        dest=ROOT/f'02_calibration/fold{fold}/cache';dest.mkdir(exist_ok=True)
        for name,arr in [('logits',z),('fde',fd),('types',typ),('headtrain_indices',kept)]: np.save(dest/f'stage11c_innerdev_{name}.npy',arr)
        T,detail=fit_one(z,fd,typ,protocol['Temperature'])
        record={'Fold':fold,'Temperature':T,'FitPartition':'InnerDev42','FitScenes':42,'FitActors':len(z),
            'VehicleCount':int((typ==0).sum()),'PedestrianCount':int((typ==1).sum()),'BicycleFitCount':0,
            **detail,'CheckpointSHA256':sha256(cp),'SplitSHA256':saved['split_sha256'],
            'NormalizationSHA256':saved['normalization_sha256'],'ProtocolSHA256':sha256(PROTOCOL),
            'InnerDevIndicesSHA256':array_sha(kept),'InnerDevSceneTokens':sorted(s['InnerDev']),
            'ReplayedCheckpointScore':replay_score,'ReplayTop1FDE':means,'NoNeuralTraining':True,
            'ModelStateSHA256Before':state_before,'ModelStateSHA256After':state_after,
            'OuterTestTemperatureSelection':False,'HeadDevVALTestUsed':False,
            'cache_files':{p.name:sha256(p) for p in dest.glob('*.npy')}}
        atomic_json(ROOT/f'02_calibration/fold{fold}/stage11c_temperature.json',record);fits.append(record)
        rows.append({k:record[k] for k in ['Fold','Temperature','FitScenes','FitActors','VehicleCount','PedestrianCount',
            'NLLRawMacro','NLLCalibratedMacro','MacroDerivativeLogT','Boundary','OptimizerEvaluations','ProtocolSHA256']})
        best=fd.argmin(-1);ii=np.arange(len(z))
        for label,temp in [('C Raw',1.),('C Calibrated',T)]:
            nll=-stable_logprob(z.astype(np.float64)/temp)[ii,best]
            for group,mask in [('Vehicle',typ==0),('Pedestrian',typ==1)]:
                devrows.append({'Fold':fold,'Group':group,'Model':label,'Count':int(mask.sum()),
                    'OracleBestModeNLL':float(nll[mask].mean()),'Temperature':temp,'Partition':'InnerDev42 (reused checkpoint-selection data)'})
        print('FOLD_TEMPERATURE',fold,T,detail,flush=True)
        del model,store;torch.cuda.empty_cache()
    dump('02_calibration/stage11c_temperature_values.csv',rows)
    dump('02_calibration/stage11c_innerdev_calibration.csv',devrows)
    assert not any(p.startswith(str(OOF.relative_to(PROJECT))+'/') for p in READS)
    audit={'Status':'PASS','TemperatureFitIsolation':'PASS','FittedScalars':3,'PerTypeTemperature':False,
        'NeuralTrainingPerformed':False,'CheckpointWrites':False,'FitOOFReads':0,
        'Fits':fits,'ReadOnlyInputFiles':sorted(READS),'ReuseOfInnerDevForSelectionAndCalibration':True,
        'OfficialVALTestHeadDevRead':False,'ProtocolSHA256':sha256(PROTOCOL)}
    atomic_json(ROOT/'02_calibration/stage11c_temperature_fit_isolation.json',audit)
    verify();print('THREE_TEMPERATURES_FIT_ISOLATION_PASS',flush=True)
if __name__=='__main__':main()
