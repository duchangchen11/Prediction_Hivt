"""Paired whole-scene intervals, including ECE recomputation per replicate."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'00_manifest'))
from stage11c_common import *
def main():
    verify();proto=read_json(PROTOCOL);cfg=proto['Bootstrap'];assert cfg['replicates']==2000
    dest=ROOT/'04_bootstrap/cache';dest.mkdir(exist_ok=True)
    cache=ROOT/'03_oof_evaluation/cache';done=read_json(cache/'stage11c_evaluation_complete.json');assert done['Status']=='PASS'
    f=pd.read_csv(cache/'stage11c_actor_records.csv',dtype={'future_mask_bits':str})
    scenes=[s for fold in (1,2,3) for s in sorted(split(fold)['OuterTest'])]
    oldaudit=read_json(S11B/'06_bootstrap/stage11b_bootstrap_audit.json')
    assert scenes==oldaudit['SceneOrder'] and len(set(scenes))==630
    si=f.scene_token.map({s:i for i,s in enumerate(scenes)}).to_numpy();assert not pd.isna(si).any()
    assert np.array_equal(si//210+1,f.Fold.to_numpy())
    rng=np.random.default_rng(cfg['seed']);weights=np.zeros((2000,630),np.int32)
    for b in range(2000):
        for fold in range(3): weights[b,fold*210:(fold+1)*210]=np.bincount(rng.integers(0,210,210),minlength=210)
    oldweights=np.load(S11B/'06_bootstrap/stage11b_scene_bootstrap_weights.npy',mmap_mode='r')
    assert np.array_equal(weights,oldweights) and np.all(weights.reshape(2000,3,210).sum(-1)==210)
    np.save(dest/'stage11c_scene_bootstrap_weights.npy',weights);W=weights.astype(np.float64)
    vals=np.load(cache/'stage11c_actor_probability_metrics.npy',mmap_mode='r');fields=done['fields']
    pp=np.load(OOF/'stage11b_oof_predictions_probabilities.npy',mmap_mode='r')
    confidence=[np.asarray(pp[:,4]).max(-1).astype(np.float64),np.load(cache/'stage11c_calibrated_top1_probability.npy')]
    hit=np.load(cache/'stage11c_calibrated_oracle_event.npy').astype(np.float64)
    bins=proto['ECE']['bins'];binids=[np.minimum((c*bins).astype(np.int64),bins-1) for c in confidence]
    scopes=[('Pooled',name,m) for name,m in groups(f).items()]
    scopes.extend((f'Fold{fold}',name,m&(f.Fold.to_numpy()==fold)) for fold in (1,2,3) for name,m in groups(f).items() if name in TYPES[:2])
    rows=[];repdata=[];directchecks=0
    for scope,group,mask in scopes:
        count=np.bincount(si[mask],minlength=630);denom=W@count
        assert np.all(denom>0)
        outputs={}
        for metric in ('OracleBestModeNLL','BrierScore','ExpectedRegret'):
            j=fields.index(metric)
            sums=[np.bincount(si[mask],weights=vals[mask,k,j],minlength=630) for k in (4,5)]
            outputs[metric]=[W@s/denom for s in sums]
        eces=[]
        for model in (0,1):
            key=si[mask]*bins+binids[model][mask]
            confsum=np.bincount(key,weights=confidence[model][mask],minlength=630*bins).reshape(630,bins)
            hitsum=np.bincount(key,weights=hit[mask],minlength=630*bins).reshape(630,bins)
            # Nonlinear absolute difference is evaluated AFTER every resample.
            samples=np.abs(W@confsum-W@hitsum).sum(-1)/denom;eces.append(samples)
            for b in (0,1,37,999,1999):
                actorweights=weights[b,si[mask]].astype(np.float64)
                c=np.bincount(binids[model][mask],weights=actorweights*confidence[model][mask],minlength=bins)
                h=np.bincount(binids[model][mask],weights=actorweights*hit[mask],minlength=bins)
                direct=np.abs(c-h).sum()/actorweights.sum()
                assert abs(direct-samples[b])<2e-12,(scope,group,model,b,direct,samples[b])
                directchecks+=1
        outputs['Top1ModeECE']=eces
        for metric,(raw,cal) in outputs.items():
            delta=cal-raw;lo,hi=np.percentile(delta,cfg['CI_percentiles'])
            if metric=='Top1ModeECE':
                point=[]
                for model in (0,1):
                    c=np.bincount(binids[model][mask],weights=confidence[model][mask],minlength=bins)
                    h=np.bincount(binids[model][mask],weights=hit[mask],minlength=bins)
                    point.append(float(np.abs(c-h).sum()/mask.sum()))
            else: point=[float(vals[mask,k,fields.index(metric)].mean()) for k in (4,5)]
            rows.append({'Scope':scope,'Group':group,'Metric':metric,'Count':int(mask.sum()),'Scenes':int((count>0).sum()),
                'C_Raw':point[0],'C_Calibrated':point[1],'DeltaCalMinusRaw':point[1]-point[0],
                'CI95Low':float(lo),'CI95High':float(hi),'Replicates':2000,'Seed':2022,'Pairing':'whole scene, within outer fold',
                'Interpretation':'descriptive OOF development evidence; temperatures/checkpoints fixed'})
            repdata.append(delta)
        print('SCENE_BOOTSTRAP_COMPLETE',scope,group,flush=True)
    np.save(dest/'stage11c_paired_metric_delta_replicates.npy',np.stack(repdata,axis=1))
    dump('04_bootstrap/stage11c_bootstrap_ci.csv',rows)
    atomic_json(ROOT/'04_bootstrap/stage11c_bootstrap_audit.json',{'Status':'PASS','Replicates':2000,'Seed':2022,
        'Unit':'paired whole scene within each outer210 fold, pooled630','SceneOrder':scenes,
        'Stage11BWeightsBitwiseReproduced':True,'WeightsSHA256':sha256(dest/'stage11c_scene_bootstrap_weights.npy'),
        'ECEBins':bins,'ECERecomputedEachReplicate':True,'DirectActorWeightedECEChecks':directchecks,
        'FixedTemperatureConditionalIntervals':True,'CheckpointReselectionOrTemperatureRefit':False,
        'Percentiles':cfg['CI_percentiles'],'CI':'descriptive, no multiplicity-controlled or independent-test interpretation'})
    print('PAIRED_BOOTSTRAP_ECE_RECOMPUTATION_PASS',flush=True)
if __name__=='__main__':main()
