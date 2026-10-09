"""Independent FP64 Torch replay of scalar fits and full OOF metrics."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'00_manifest'))
from stage11c_common import *
@torch.inference_mode()
def main():
    seed();frozen=verify(history=True);protocol=read_json(PROTOCOL)
    fit=read_json(ROOT/'02_calibration/stage11c_temperature_fit_isolation.json');assert fit['Status']=='PASS'
    fitchecks=[]
    for rec in fit['Fits']:
        fold=rec['Fold'];cache=ROOT/f'02_calibration/fold{fold}/cache'
        ix=np.load(cache/'stage11c_innerdev_headtrain_indices.npy')
        assert np.array_equal(ix,indices(fold,'InnerDev')[np.isin(frame().iloc[indices(fold,'InnerDev')].agent_type_id,[0,1])])
        assert array_sha(ix)==rec['InnerDevIndicesSHA256']
        assert not set(frame().iloc[ix].scene_token)&set(split(fold)['OuterTest'])
        assert rec['ModelStateSHA256Before']==rec['ModelStateSHA256After']
        z=torch.from_numpy(np.load(cache/'stage11c_innerdev_logits.npy')).double()
        fd=torch.from_numpy(np.load(cache/'stage11c_innerdev_fde.npy')).double()
        typ=np.load(cache/'stage11c_innerdev_types.npy');best=fd.argmin(-1);ii=torch.arange(len(z))
        T=rec['Temperature'];assert 1<=T<=1000
        for value,key in [(1.,'NLLRawMacro'),(T,'NLLCalibratedMacro')]:
            nll=-(z/value).log_softmax(-1)[ii,best]
            macro=float(.5*nll[typ==0].mean()+.5*nll[typ==1].mean())
            assert abs(macro-rec[key])<1e-11,(fold,key)
        # NLL is convex in inverse temperature. Verify first-order optimum
        # without optimizing or selecting another temperature.
        scaled=z/T;p=scaled.softmax(-1)
        d=scaled[ii,best]-(p*scaled).sum(-1)
        grad=float(.5*d[typ==0].mean()+.5*d[typ==1].mean())
        assert abs(grad-rec['MacroDerivativeLogT'])<1e-11
        if 1<T<1000: assert abs(grad)<1e-6
        else: assert (T==1000 and grad<=1e-6) or (T==1 and grad>=-1e-6)
        assert np.array_equal(z.argmax(-1).numpy(),scaled.argmax(-1).numpy())
        fitchecks.append({'Fold':fold,'Temperature':T,'TorchNLLReplay':'PASS','ConvexInverseTemperatureFirstOrderCheck':'PASS','LogTGradient':grad})
    assert fit['FitOOFReads']==0 and not any(p.startswith(str(OOF.relative_to(PROJECT))+'/') for p in fit['ReadOnlyInputFiles'])
    cache=ROOT/'03_oof_evaluation/cache';complete=read_json(cache/'stage11c_evaluation_complete.json')
    for name,h in complete['cache_files'].items(): assert sha256(cache/name)==h,name
    f=frame();src=f.source_index.to_numpy();n=len(f)
    fd=torch.from_numpy(np.array(np.load(S8/'01_training/cache/fde.npy',mmap_mode='r')[src])).double()
    ad=torch.from_numpy(np.array(np.load(S8/'01_training/cache/ade.npy',mmap_mode='r')[src])).double()
    rawz=np.load(OOF/'stage11b_oof_predictions_logits.npy',mmap_mode='r')
    rawp=np.load(OOF/'stage11b_oof_predictions_probabilities.npy',mmap_mode='r')
    values=np.load(cache/'stage11c_actor_probability_metrics.npy',mmap_mode='r')
    fields=complete['fields'];vp=np.load(cache/'stage11c_calibrated_vehicle_pedestrian.npz')
    bike=np.load(cache/'stage11c_bicycle_foldr2_passthrough.npz')
    calz=np.empty((n,6),np.float64);calp=np.empty((n,6),np.float64)
    for block in [vp,bike]:
        ix=block['headtrain_indices'];calz[ix]=block['logits'];calp[ix]=block['probabilities']
    best=fd.argmin(-1);oracle=fd.min(-1).values;cost=fd-oracle[:,None];scale=cost.mean(-1).clamp(min=1);q=(-fd).softmax(-1);ii=torch.arange(n)
    maxdiff={k:0. for k in fields};binchecks=0
    table=pd.read_csv(ROOT/'07_tables/stage11c_oof_probability_metrics.csv')
    for j,name in enumerate(MODELS):
        z=torch.from_numpy(np.array(rawz[:,j] if j<5 else calz)).double()
        p=torch.from_numpy(np.array(rawp[:,j] if j<5 else calp)).double()
        lp=z.log_softmax(-1);top=p.argmax(-1);target=torch.zeros_like(p);target[ii,best]=1
        replay=torch.stack((-lp[ii,best],((p-target)**2).sum(-1),-(q*lp).sum(-1),(p*fd).sum(-1),
            (p*cost).sum(-1),(p*cost).sum(-1)/scale,-torch.special.xlogy(p,p).sum(-1),p.max(-1).values,p[ii,best],
            fd[ii,top],ad[ii,top],oracle,ad.min(-1).values,(oracle>2).double()),-1).numpy()
        for k,field in enumerate(fields):
            error=float(np.max(np.abs(replay[:,k]-values[:,j,k])));maxdiff[field]=max(maxdiff[field],error)
            assert error<1e-10,(name,field,error)
        for group,mask in groups(f).items():
            row=table[(table.Group==group)&(table.Model==name)].iloc[0]
            assert int(row.Count)==int(mask.sum())
            for k,field in enumerate(fields): assert abs(float(replay[mask,k].mean())-float(row[field]))<1e-10,(name,group,field)
            conf=p[mask].max(-1).values;hits=(top[mask]==best[mask]).double();binid=torch.clamp((conf*15).long(),max=14)
            ece=0.
            for b in range(15):
                m=binid==b
                if m.any(): ece+=float(torch.abs((conf[m]-hits[m]).sum()))/int(mask.sum())
            assert abs(ece-float(row.Top1ModeECE))<1e-11,(name,group,ece,row.Top1ModeECE)
            binchecks+=1
        print('INDEPENDENT_TORCH_OOF_REPLAY_PASS',name,flush=True)
    assert read_json(ROOT/'01_frozen_audit/stage11c_top1_identity_audit.json')['Status']=='PASS'
    assert read_json(ROOT/'01_frozen_audit/stage11c_bicycle_identity_audit.json')['Status']=='PASS'
    ba=read_json(ROOT/'04_bootstrap/stage11c_bootstrap_audit.json');assert ba['Status']=='PASS' and ba['ECERecomputedEachReplicate']
    assert complete['AllFinite'] and complete['ProbabilitySumMaxAbsErrorCalibratedVP']<1e-12
    verify(history=True)
    result={'Status':'PASS','FrozenModelIntegrity':'PASS','TemperatureFitIsolation':'PASS','CalibrationEngineering':'PASS',
        'Top1Identity':'PASS','BicyclePreserved':'YES','HistoricalTrackedFilesUnchanged':len(frozen['historical_files']),
        'PreservedStage2CUntrackedFiles':len(frozen['preserved_untracked_files']),'CheckpointsUnchanged':len(frozen['checkpoints']),
        'Stage11BCheckpointsUnchanged':12,'ThreeScalarFitChecks':fitchecks,'FullOOFActors':n,
        'IndependentTorchMetricMaxAbsDiff':maxdiff,'IndependentGroupECEChecks':binchecks,
        'NoNeuralTrainingOrCheckpointWrites':True,'OfficialVALTestHeadDevDataUsed':False,
        'DevelopmentEvidenceOnly':True,'ProtocolSHA256':sha256(PROTOCOL),'FrozenManifestSHA256':sha256(FROZEN)}
    atomic_json(ROOT/'08_reports/stage11c_final_audit.json',result)
    print('STAGE11C_INDEPENDENT_FINAL_AUDIT_PASS',flush=True)
if __name__=='__main__':main()
