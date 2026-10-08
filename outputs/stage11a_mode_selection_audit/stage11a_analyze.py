"""Registered frozen-score statistics. No model, loss or optimizer changes."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'01_identity_audit'))
from stage11a_common import *
from scipy.special import softmax
from scipy.stats import spearmanr
from itertools import combinations

def dump(relative,rows):
    pd.DataFrame(rows).to_csv(ROOT/relative,index=False,float_format='%.12g')

def main():
    verify(); assert read_json(CACHE/'inference_complete.json')['status']=='PASS'
    f=actor_frame(); n=len(f); spec=read_json(PROTOCOL)
    fd,ad=fde_ade(); fd=np.asarray(fd,dtype=np.float64); ad=np.asarray(ad,dtype=np.float64)
    z,p=scores(); p=np.asarray(p); mm=np.asarray(metrics(),dtype=np.float64)
    top=np.argmax(np.nan_to_num(p,nan=-np.inf),axis=-1)
    oracle=fd.min(-1); oracle_ade=ad.min(-1); best=fd.argmin(-1)
    topfde=fd[np.arange(n)[:,None],top]; topade=ad[np.arange(n)[:,None],top]
    regret=topfde-oracle[:,None]; ce=mm[:,:,0]
    for j,m in enumerate(MODELS):
        valid=applicable(f,m)
        assert np.array_equal(topfde[valid,j],mm[valid,j,1])
        assert np.array_equal(topade[valid,j],mm[valid,j,2])
        assert np.max(np.abs(regret[valid,j]-mm[valid,j,3]))<1e-4
    q=softmax(-fd,axis=-1); entropy=-(q*np.log(np.maximum(q,1e-300))).sum(-1)
    ordered=np.sort(fd,axis=-1); qo=np.sort(q,axis=-1)[:,::-1]
    soft=np.stack((ordered[:,0],ordered[:,1],ordered[:,-1],ordered[:,1]-ordered[:,0],
        ordered[:,-1]-ordered[:,0],qo[:,0],entropy,entropy/np.log(6),qo[:,0],qo[:,1],qo[:,0]-qo[:,1]),-1)
    softcols=['best_FDE','second_best_FDE','worst_FDE','best_second_gap','worst_best_gap','max_q',
        'entropy_q','normalized_entropy','q_best','q_second','q_margin']
    np.save(CACHE/'soft_label_actor.npy',soft)
    pairs=list(combinations(range(6),2)); candidates=np.load(CACHE/'candidates.npy',mmap_mode='r')
    endpoints=np.lib.format.open_memmap(CACHE/'endpoint_pair_distance.npy',mode='w+',dtype='float32',shape=(n,15))
    geom=np.lib.format.open_memmap(CACHE/'candidate_geometry_actor.npy',mode='w+',dtype='float64',shape=(n,8))
    for start in range(0,n,4096):
        c=np.asarray(candidates[start:start+4096]); e=c[:,:,-1,:].astype(np.float64)
        distances=np.stack([np.linalg.norm(e[:,a]-e[:,b],axis=-1) for a,b in pairs],-1)
        rms=np.stack([np.sqrt(np.mean(np.sum((c[:,a].astype(np.float64)-c[:,b])**2,-1),-1)) for a,b in pairs],-1)
        exact=np.stack([np.all(c[:,a].view('uint32')==c[:,b].view('uint32'),axis=(1,2)) for a,b in pairs],-1)
        endpoints[start:start+len(c)]=distances
        geom[start:start+len(c)]=np.stack((distances.min(-1),distances.mean(-1),
            np.sqrt(np.mean(np.sum((e-e.mean(1,keepdims=True))**2,-1),-1)),distances.max(-1),
            (distances<=spec['duplicate_endpoint_threshold_m']).any(-1),
            (rms<=spec['duplicate_trajectory_RMS_threshold_m']).any(-1),exact.any(-1),rms.min(-1)),-1)
    endpoints.flush();geom.flush()
    gcols=['MinEndpointPairDistance','MeanEndpointPairDistance','EndpointSpreadRMS','EndpointDiameter',
        'NearDuplicateEndpointRate','NearDuplicateTrajectoryRate','ExactDuplicateTrajectoryRate','MinTrajectoryPairRMS']
    atomic_json(ROOT/'06_candidate_geometry/stage11a_geometry_schema.json',{'columns':gcols,'pair_columns':pairs,
        'near_duplicate_endpoint_threshold_m':.25,'near_duplicate_trajectory_RMS_threshold_m':.25,
        'thresholds_registered_before_statistics':True,'per_actor_arrays':'01_identity_audit/cache/; local only'})
    summary=[]; softrows=[]; grows=[]; training=[]; pedrows=[]; align=[]; switchrows=[]; decomposition=[]
    allswitch=[]; mi=MODELS.index('R2'); eps=spec['delta_sign_tolerances']['FDE_Regret_m']; ece=spec['delta_sign_tolerances']['SoftCE']
    decision_witness=[]; costly_witness=[]
    groups=group_masks(f)
    for partition in ('HeadTrain','HeadDev'):
        part=np.asarray(f.Partition==partition)
        for group,mask0 in groups.items():
            mask=part&mask0; ix=np.flatnonzero(mask)
            if not len(ix):continue
            base={'Partition':partition,'Group':group,'Count':len(ix)}
            sf=soft[ix]; sr={**base,**dict(zip(softcols,sf.mean(0))),
                'DiffuseTargetRate':float(((sf[:,7]>=.9)&(sf[:,10]<=.05)).mean())}
            softrows.append(sr)
            grows.append({**base,'minADEOracle':float(oracle_ade[ix].mean()),'minFDE':float(oracle[ix].mean()),
                **dict(zip(gcols,np.asarray(geom[ix]).mean(0))),
                'OracleFDE_le1m_Rate':float((oracle[ix]<=1).mean())})
            for j,m in enumerate(MODELS):
                ids=ix[applicable(f.iloc[ix],m)]
                if not len(ids):continue
                b={**base,'Count':len(ids),'Model':m}
                summary.append({**b,'SoftCE':float(ce[ids,j].mean()),'Top1FDE':float(topfde[ids,j].mean()),
                    'Top1ADE':float(topade[ids,j].mean()),'OracleFDE':float(oracle[ids].mean()),
                    'minADEOracle':float(oracle_ade[ids].mean()),'OracleGap':float(regret[ids,j].mean()),
                    'HitRate':float((top[ids,j]==best[ids]).mean()),
                    'GoodCandidateWrongSelectionRate':float(((oracle[ids]<=1)&(regret[ids,j]>.5)).mean()),
                    'OracleFractionOfTop1FDE':float(oracle[ids].sum()/topfde[ids,j].sum())})
                if partition=='HeadTrain':
                    valid=part&applicable(f,m); loss=ce[ids,j]; kl=loss-entropy[ids]
                    total_ce=ce[valid,j].sum();total_kl=(ce[valid,j]-entropy[valid]).sum(); ge=mm[ids,j,7]
                    training.append({**b,'PopulationFraction':float(len(ids)/valid.sum()),'MeanSoftCE':float(loss.mean()),
                        'SoftCE_P10':float(np.quantile(loss,.1)),'SoftCE_Median':float(np.median(loss)),
                        'SoftCE_P90':float(np.quantile(loss,.9)),'SoftCE_P99':float(np.quantile(loss,.99)),
                        'TotalSoftCE':float(loss.sum()),'SoftCEContributionShare':float(loss.sum()/total_ce),
                        'MeanTargetEntropy':float(entropy[ids].mean()),'MeanExcessCE_KL':float(kl.mean()),
                        'ExcessCEContributionShare':float(kl.sum()/total_kl),
                        'MeanLogitGradL1':float(mm[ids,j,5].mean()),'MeanLogitGradL2':float(mm[ids,j,6].mean()),
                        'MeanLogitGradSquaredL2':float(ge.mean()),'LogitGradSquaredL2Share':float(ge.sum()/mm[valid,j,7].sum()),
                        'MeanMicro1024LogitGradL2':float(mm[ids,j,6].mean()/1024),
                        'GradientScope':'output logits only; shared parameter gradients unresolved'})
                if partition=='HeadDev' and group.startswith('Pedestrian') and m in ('R2','G1','PedestrianExpert'):
                    delta=topfde[ids,j]-topfde[ids,mi]; changed=top[ids,j]!=top[ids,mi]; harm=delta[delta>eps]
                    pedrows.append({**b,'Top1FDE':float(topfde[ids,j].mean()),'OracleFDE':float(oracle[ids].mean()),
                        'OracleGap':float(regret[ids,j].mean()),'mode_change_rate':float(changed.mean()),
                        'switch_harm':float(harm.sum()/len(ids)),'MeanHarmConditional':float(harm.mean()) if len(harm) else 0,
                        'GrossSwitchHarm':float(harm.sum()),'SoftCE':float(ce[ids,j].mean()),
                        'DeltaTop1FDE':float(delta.mean())})
                if partition=='HeadDev' and m in ('G1','G3','T2','T3','VehicleExpert','PedestrianExpert','DualExpert'):
                    dc=ce[ids,j]-ce[ids,mi];dr=regret[ids,j]-regret[ids,mi]
                    bad=(dc< -ece)&(dr>eps); down=dc< -ece
                    correlation=spearmanr(dc,dr) if len(ids)>2 and np.std(dc)>0 and np.std(dr)>0 else (np.nan,np.nan)
                    align.append({**b,'Comparison':'R2->'+m,'DeltaSoftCE':float(dc.mean()),'DeltaRegret_m':float(dr.mean()),
                        'CEDown_RegretUp_Count':int(bad.sum()),'CEDown_RegretUp_Rate':float(bad.mean()),
                        'CEDown_Count':int(down.sum()),'CEDown_RegretUp_ConditionalRate':float(bad.sum()/max(1,down.sum())),
                        'CEDown_RegretDown_Rate':float(((dc< -ece)&(dr< -eps)).mean()),
                        'CEUp_RegretUp_Rate':float(((dc>ece)&(dr>eps)).mean()),
                        'CEUp_RegretDown_Rate':float(((dc>ece)&(dr< -eps)).mean()),
                        'SpearmanDeltaCE_DeltaRegret':float(correlation[0]),'p_value_descriptive':float(correlation[1])})
                    if m in SWITCH and group in ('Vehicle','Pedestrian') and dc.mean()< -ece and dr.mean()>eps and bad.mean()>=.01:
                        decision_witness.append({'Model':m,'Group':group,'DeltaSoftCE':float(dc.mean()),'DeltaRegret':float(dr.mean()),'MismatchRate':float(bad.mean())})
            if partition=='HeadDev':
                for m in SWITCH:
                    j=MODELS.index(m); delta=topfde[ix,j]-topfde[ix,mi];changed=top[ix,j]!=top[ix,mi]
                    gain=-delta[delta< -eps];harm=delta[delta>eps]
                    net=float(delta.sum());assert abs(net-(harm.sum()-gain.sum()))<eps*len(ix)
                    row={**base,'Comparison':'R2->'+m,'changed_count':int(changed.sum()),'improved_count':len(gain),
                        'worsened_count':len(harm),'zero_cost_changed_count':int((changed&(np.abs(delta)<=eps)).sum()),
                        'mean_gain':float(gain.mean()) if len(gain) else 0,'mean_harm':float(harm.mean()) if len(harm) else 0,
                        'median_gain':float(np.median(gain)) if len(gain) else 0,'median_harm':float(np.median(harm)) if len(harm) else 0,
                        **{'p%d_harm'%qv:float(np.quantile(harm,qv/100)) if len(harm) else 0 for qv in (90,95,99)},
                        'total_FDE_gain':float(gain.sum()),'total_FDE_harm':float(harm.sum()),'net_delta_FDE':float(delta.mean()),
                        'total_net_delta_FDE':net,'mode_change_rate':float(changed.mean())}
                    orderedharm=np.sort(harm)[::-1]
                    for pc in (10,5,1):
                        cnt=int(np.ceil(len(harm)*pc/100));tail=float(orderedharm[:cnt].sum())
                        row['top%d_harm_count'%pc]=cnt;row['top%d_harm_share'%pc]=tail/harm.sum() if len(harm) else 0
                    switchrows.append(row)
                    if group in TYPES and delta.mean()>eps and (len(harm)<=len(gain) or row['top10_harm_share']>=.5):costly_witness.append(row)
    # Explicit historical-speed static screen, not a training filter or category inference.
    static=(np.asarray(f.history_valid_count)>=2)&(np.asarray(f.recent_speed)<.2)&(oracle<=1)
    train=np.asarray(f.Partition=='HeadTrain')
    for group,mask in {'ObservableStaticLowError':static,'Complement':~static,
        'VehicleObservableStaticLowError':static&np.asarray(f.agent_type=='Vehicle'),
        'PedestrianObservableStaticLowError':static&np.asarray(f.agent_type=='Pedestrian')}.items():
        for j,m in enumerate(MODELS):
            valid=train&applicable(f,m);ids=np.flatnonzero(valid&mask)
            if not len(ids):continue
            loss=ce[ids,j];kl=loss-entropy[ids]
            training.append({'Partition':'HeadTrain','Group':group,'Count':len(ids),'Model':m,
                'PopulationFraction':float(len(ids)/valid.sum()),'MeanSoftCE':float(loss.mean()),'TotalSoftCE':float(loss.sum()),
                'SoftCEContributionShare':float(loss.sum()/ce[valid,j].sum()),'MeanTargetEntropy':float(entropy[ids].mean()),
                'MeanExcessCE_KL':float(kl.mean()),'ExcessCEContributionShare':float(kl.sum()/(ce[valid,j]-entropy[valid]).sum()),
                'MeanLogitGradL1':float(mm[ids,j,5].mean()),'MeanLogitGradL2':float(mm[ids,j,6].mean()),
                'MeanLogitGradSquaredL2':float(mm[ids,j,7].mean()),
                'LogitGradSquaredL2Share':float(mm[ids,j,7].sum()/mm[valid,j,7].sum()),
                'MeanMicro1024LogitGradL2':float(mm[ids,j,6].mean()/1024),
                'GradientScope':'output logits only; shared parameter gradients unresolved'})
    dev=np.flatnonzero(f.Partition.to_numpy()=='HeadDev')
    for m in SWITCH:
        j=MODELS.index(m); delta=topfde[dev,j]-topfde[dev,mi]
        record=f.iloc[dev][['actor_id','scene_token','sample_token','instance_token','agent_type','GT_displacement','recent_speed']].copy()
        record=record.rename(columns={'scene_token':'scene_id','agent_type':'actor_type'})
        record['Comparison']='R2->'+m;record['new_model']=m;record['source_index']=dev
        record['R2_top_mode']=top[dev,mi];record['new_top_mode']=top[dev,j];record['best_FDE_mode']=best[dev]
        record['R2_top_FDE']=topfde[dev,mi];record['new_top_FDE']=topfde[dev,j];record['oracle_FDE']=oracle[dev]
        record['delta_FDE']=delta;record['mode_changed']=top[dev,j]!=top[dev,mi]
        record['switch_improved']=delta< -eps;record['switch_worsened']=delta>eps
        record['delta_SoftCE']=ce[dev,j]-ce[dev,mi];record['delta_Regret']=regret[dev,j]-regret[dev,mi]
        record.to_csv(ROOT/f'02_switch_regret/stage11a_{m}_actor_records.csv',index=False,float_format='%.12g')
        allswitch.append(record)
    pool=pd.concat(allswitch,ignore_index=True); chosen=[];seen=set()
    for category,mask,ascending in (
        ('PedestrianHarm',(pool.actor_type=='Pedestrian')&pool.switch_worsened,False),
        ('VehicleHarm',(pool.actor_type=='Vehicle')&pool.switch_worsened,False),
        ('Success',pool.switch_improved,True)):
        count=0
        for _,r in pool[mask].sort_values(['delta_FDE','actor_id','new_model'],ascending=[ascending,True,True]).iterrows():
            if r.instance_token in seen:continue
            seen.add(r.instance_token);chosen.append({**r.to_dict(),'CaseCategory':category,'CaseNumber':len(chosen)+1})
            count+=1
            if count==10:break
        assert count==10
    dump('08_cases/stage11a_case_selection.csv',chosen)
    # Numerical output-logit gradient validation. No frozen parameter participates in autograd.
    sample=dev[:128];zt=torch.from_numpy(np.array(z[sample,mi],copy=True)).double().requires_grad_(True)
    qt=torch.from_numpy(q[sample]); loss=-(qt*zt.log_softmax(-1)).sum();loss.backward()
    maxdiff=float((zt.grad-(zt.detach().softmax(-1)-qt)).abs().max());assert maxdiff<1e-12
    frozen_analytic_diff=0.
    for j,m in enumerate(MODELS):
        valid=applicable(f.iloc[dev],m); grad=p[dev[valid],j].astype(np.float64)-q[dev[valid]]
        frozen_analytic_diff=max(frozen_analytic_diff,float(np.max(np.abs(np.sum(grad**2,-1)-mm[dev[valid],j,7]))))
    assert frozen_analytic_diff<2e-6
    atomic_json(ROOT/'07_training_distribution/stage11a_gradient_audit.json',{'Status':'PASS',
        'isolated_logit_autograd_maxdiff':maxdiff,'cached_gradient_energy_maxdiff':frozen_analytic_diff,
        'frozen_parameter_backward_calls':0,'optimizer_updates':0,'gradient_scope':'dSoftCE/doutput_logit only; no shared-parameter dominance claim'})
    ped=np.asarray((f.Partition=='HeadDev')&(f.agent_type=='Pedestrian'));du=MODELS.index('DualExpert')
    delta=topfde[:,du]-topfde[:,mi];high=ped&(np.asarray(f.GT_displacement)>5);low=ped&~high
    recentvalid=ped&(np.asarray(f.history_valid_count)>=2);speedhigh=recentvalid&(np.asarray(f.recent_speed)>=1);speedlow=recentvalid&~speedhigh
    pedharm=np.maximum(delta[ped],0).sum();motion={'PedCount':int(ped.sum()),'GTGreater5mCount':int(high.sum()),
        'GTGreater5mMeanDelta':float(delta[high].mean()),'GTAtMost5mMeanDelta':float(delta[low].mean()),
        'PedMeanDelta':float(delta[ped].mean()),'GTGreater5mGrossHarmShare':float(np.maximum(delta[high],0).sum()/pedharm),
        'ObservedSpeedAtLeast1mpsCount':int(speedhigh.sum()),'ObservedSpeedAtLeast1mpsMeanDelta':float(delta[speedhigh].mean()),
        'ObservedSpeedBelow1mpsMeanDelta':float(delta[speedlow].mean()),'HistoryInsufficientCount':int((ped&~recentvalid).sum())}
    motionbias=(motion['GTGreater5mMeanDelta']>motion['GTAtMost5mMeanDelta'] and motion['PedMeanDelta']>0
        and motion['GTGreater5mGrossHarmShare']>=.5 and motion['ObservedSpeedAtLeast1mpsMeanDelta']>motion['ObservedSpeedBelow1mpsMeanDelta'])
    softdiff=float(((soft[ped,7]>=.9)&(soft[ped,10]<=.05)).mean());fraction=float(oracle[ped].sum()/topfde[ped,mi].sum())
    geometry='NO' if fraction<=.2 else 'YES' if fraction>=.8 else 'PARTIAL'
    objective='ErrorAwareRanking' if decision_witness and costly_witness else 'HardCE' if decision_witness else 'None'
    decisions={'LossRankingMismatch':'CONFIRMED' if decision_witness else 'NOT_CONFIRMED',
        'CostlyModeSwitchProblem':'CONFIRMED' if costly_witness else 'NOT_CONFIRMED',
        'PedestrianMotionBias':'CONFIRMED' if motionbias else 'NOT_CONFIRMED',
        'SoftTargetAmbiguity':'CONFIRMED' if softdiff>=.5 else 'NOT_CONFIRMED',
        'CandidateGeometryBottleneck':geometry,'RecommendedNextObjective':objective,
        'ReadyForControlledLossExperiment':'YES' if objective!='None' else 'NO',
        'LossRankingMismatchWitnesses':decision_witness,'CostlyModeSwitchWitnesses':costly_witness,
        'PedestrianMotionEvidence':motion,'HeadDevPedestrianDiffuseTargetRate':softdiff,'PedestrianR2OracleFraction':fraction,
        'engineering_audits':'PASS','qualification':'registered descriptive decisions, repeatedly selected HeadDev; no causal or independent confirmation',
        'training_authorized_or_executed':False,'optimizer_updates':0}
    dump('02_switch_regret/stage11a_switch_summary.csv',switchrows)
    dump('03_pedestrian_analysis/stage11a_pedestrian_motion.csv',pedrows)
    atomic_json(ROOT/'03_pedestrian_analysis/stage11a_motion_bias_evidence.json',motion)
    dump('04_soft_label_analysis/stage11a_soft_label_groups.csv',softrows)
    atomic_json(ROOT/'04_soft_label_analysis/stage11a_soft_label_schema.json',{'columns':softcols,'temperature_m':1,'sweeps':0,'primary_decision_split':'HeadDev'})
    dump('05_loss_alignment/stage11a_all_model_group_metrics.csv',summary)
    dump('05_loss_alignment/stage11a_loss_regret_alignment.csv',align)
    dump('06_candidate_geometry/stage11a_geometry_groups.csv',grows)
    dump('07_training_distribution/stage11a_training_loss_gradient.csv',training)
    atomic_json(ROOT/'09_reports/stage11a_decisions.json',decisions)
    verify();atomic_json(ROOT/'01_identity_audit/stage11a_statistics_audit.json',{'Status':'PASS','Rows':n,
        'HeadDev':len(dev),'Cases':len(chosen),'UniqueCaseInstances':len(seen),'all_model_top1_identity':True,
        'switch_gain_harm_net_identity':True,'isolated_logit_gradient_validated':True,'frozen_checkpoint_updates':0,
        'protocol_sha256':sha256(PROTOCOL),'geometry_per_actor_schema':gcols,'soft_label_per_actor_schema':softcols})
    print('STAGE11_STATISTICS_PASS',json.dumps({k:v for k,v in decisions.items() if isinstance(v,(str,float,bool,int))}),flush=True)

if __name__=='__main__':main()
