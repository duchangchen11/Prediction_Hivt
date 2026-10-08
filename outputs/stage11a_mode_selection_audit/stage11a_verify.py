"""Independent artifact consistency checks and final frozen-history preservation."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'01_identity_audit'))
from stage11a_common import *
from scipy.special import logsumexp,softmax

@torch.no_grad()
def main():
    history=verify(history=True);f=actor_frame();n=len(f);fd,ad=fde_ade();z,p=scores(); mm=metrics();checks=[]
    def check(name,value,evidence):
        checks.append({'Audit':name,'Status':'PASS' if value else 'FAIL','Evidence':evidence});assert value,(name,evidence)
    for artifact in ('stage11a_code_label_audit.csv','stage11a_historical_reproduction.csv'):
        a=pd.read_csv(ROOT/'01_identity_audit'/artifact);check(artifact,(a.Status=='PASS').all(),str(len(a))+' checks')
    sums=0.;cemax=0.;gmax=0.
    for start in range(0,n,4096):
        ix=np.arange(start,min(n,start+4096));ff=fd[ix].astype(np.float64);qq=softmax(-ff,axis=-1)
        for j,m in enumerate(MODELS):
            v=applicable(f.iloc[ix],m);zz=z[ix[v],j].astype(np.float64);pp=p[ix[v],j].astype(np.float64)
            if not v.any():continue
            sums=max(sums,float(np.abs(pp.sum(-1)-1).max()))
            targetce=-(qq[v]*(zz-logsumexp(zz,axis=-1,keepdims=True))).sum(-1)
            cemax=max(cemax,float(np.abs(targetce-mm[ix[v],j,0]).max()))
            gmax=max(gmax,float(np.abs(np.sum((pp-qq[v])**2,-1)-mm[ix[v],j,7]).max()))
            top=pp.argmax(-1);check_fde=ff[v][np.arange(v.sum()),top]
            assert np.array_equal(check_fde,mm[ix[v],j,1])
    check('Probability_normalization',sums<1e-6,str(sums));check('All_actor_independent_SoftCE_formula',cemax<2e-5,str(cemax))
    check('All_actor_analytical_gradient_energy',gmax<2e-6,str(gmax))
    summary=pd.read_csv(ROOT/'02_switch_regret/stage11a_switch_summary.csv')
    dev=f[f.Partition=='HeadDev'];devix=dev.index.to_numpy();mi=MODELS.index('R2')
    for m in SWITCH:
        record=pd.read_csv(ROOT/f'02_switch_regret/stage11a_{m}_actor_records.csv')
        check(m+'_actor_order',np.array_equal(record.actor_id,dev.actor_id),str(len(record)))
        check(m+'_regret_delta_identity',np.max(np.abs(record.delta_FDE-record.delta_Regret))<1e-8,'common oracle cancels')
        for typ in TYPES:
            sub=record[record.actor_type==typ];r=summary[(summary.Group==typ)&(summary.Comparison=='R2->'+m)].iloc[0]
            check(m+'_'+typ+'_switch_count',int(sub.mode_changed.sum())==r.changed_count,'per-actor/aggregate')
            check(m+'_'+typ+'_switch_mean',abs(sub.delta_FDE.mean()-r.net_delta_FDE)<1e-10,'per-actor/aggregate')
            check(m+'_'+typ+'_gain_harm',abs(r.total_FDE_harm-r.total_FDE_gain-r.total_net_delta_FDE)<1e-6,'gross harm - gross gain = net')
    train=pd.read_csv(ROOT/'07_training_distribution/stage11a_training_loss_gradient.csv')
    for m in MODELS:
        typed=train[(train.Model==m)&train.Group.isin(TYPES)]
        check(m+'_type_shares',all(abs(typed[col].sum()-1)<1e-9 for col in
            ('PopulationFraction','SoftCEContributionShare','ExcessCEContributionShare','LogitGradSquaredL2Share')),'disjoint type contributions')
    states=f[f.agent_type=='Vehicle'].groupby(['Partition','motion_state'],dropna=False).size().reset_index(name='Count')
    states.to_csv(ROOT/'07_training_distribution/stage11a_vehicle_state_counts.csv',index=False)
    check('Vehicle_state_count_accounting',int(states.Count.sum())==int((f.agent_type=='Vehicle').sum()),'unclassified states retained')
    case=pd.read_csv(ROOT/'08_cases/stage11a_case_selection.csv');ca=read_json(ROOT/'08_cases/stage11a_case_audit.json')
    check('Fixed_case_counts',len(case)==case.instance_token.nunique()==30 and all(case.CaseCategory.value_counts()==10),'10 pedestrian harm,10 vehicle harm,10 success')
    for name,h in ca['figures'].items():assert sha256(ROOT/'08_cases'/name)==h
    score=pd.read_csv(ROOT/'08_cases/stage11a_case_mode_scores.csv');pairs=pd.read_csv(ROOT/'08_cases/stage11a_case_endpoint_pairs.csv')
    check('Case_endpoint_pairs',len(pairs)==450 and all(pairs.groupby('CaseNumber').size()==15),'all15 unique endpoint pairs per case')
    for r in case.itertuples():
        i=int(r.source_index)
        for row in score[score.CaseNumber==r.CaseNumber].itertuples():
            j=MODELS.index(row.Model);k=row.Mode
            assert abs(float(z[i,j,k])-row.Logit)<1e-9 and abs(float(p[i,j,k])-row.Probability)<1e-10
            assert bool(row.Selected)==bool(k==p[i,j].argmax()) and bool(row.Oracle)==bool(k==fd[i].argmin())
    check('Case_scores_identity',True,str(len(score))+' mode score records')
    # Revalidate every TRAIN record's raw and transformed predictor SHA after the explicit raw-hash check was added.
    records=[r for r in read_json(S6/'01_cache/stage6a_cache_manifest.json')['batches'] if r['split']=='train'];windows=0
    for bi,r in enumerate(records):
        block=torch.load(S6/r['path'],map_location='cpu',weights_only=False)
        for w in block['windows']:
            assert tensor_sha(w['raw_prediction'])==w['raw_prediction_sha256']
            assert tensor_sha(w['ego_prediction'])==w['ego_prediction_sha256'];windows+=1
        if (bi+1)%250==0:print('RAW_PREDICTOR_HASH_CHECK',bi+1,'/',len(records),flush=True)
    check('All_TRAIN_raw_and_ego_prediction_SHA',True,str(windows)+' windows in'+str(len(records))+' TRAIN records')
    for artifact in ('prepared.json','inference_complete.json'):
        for name,h in read_json(CACHE/artifact)['files'].items():assert sha256(CACHE/name)==h
    check('Local_cache_manifest_SHA',True,'all saved inference/prepared cache hashes')
    check('Protocol_unchanged',sha256(PROTOCOL)==read_json(REG)['protocol_sha256'],'registered before statistics')
    refs=read_json(ROOT/'09_reports/literature/stage11a_verified_references.json')
    check('Three_original_sources',len(refs['references'])==3,'primary-source anchors and limitations recorded')
    check('No_new_checkpoint',not any(p.suffix in ('.pt','.pth','.ckpt') for p in ROOT.rglob('*')),'all Stage11A artifacts')
    verify(history=True);check('All_historical_files_preserved',True,str(len(history['historical_files']))+' tracked history and5 untracked files')
    write_csv(ROOT/'01_identity_audit/stage11a_final_checks.csv',checks)
    atomic_json(ROOT/'01_identity_audit/stage11a_final_integrity.json',{'Status':'PASS','checks':len(checks),
        'HeadTrain':260151,'HeadDev':29934,'scene_split':'630/70 from official TRAIN700 only',
        'optimizer_updates':0,'new_checkpoints':0,'frozen_parameter_backward_calls':0,'official_VAL_test_used':False,
        'history_files_preserved':len(history['historical_files']),'preserved_untracked_files':5,
        'probability_sum_maxdiff':sums,'independent_SoftCE_maxdiff':cemax,'analytical_gradient_energy_maxdiff':gmax,
        'TRAIN_raw_prediction_windows_checked':windows,'cases':30,'case_instances':30,'protocol_sha256':sha256(PROTOCOL)})
    print('STAGE11_FINAL_INTEGRITY_PASS',len(checks),flush=True)

if __name__=='__main__':main()
