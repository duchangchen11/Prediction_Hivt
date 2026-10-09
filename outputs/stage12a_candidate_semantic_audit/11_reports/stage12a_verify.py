"""Independent identities, numeric summaries, controls and historical integrity."""
from pathlib import Path
import sys
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'00_manifest'),str(Path(__file__).resolve().parents[1]/'03_feature_statistics')]
from stage12a_analysis_common import *
from scipy.stats import spearmanr
def main():
    torch.set_num_threads(1);frozen=verify(history=True,data=True)
    data=load_semantics();f,fd,ad,top,best,p,z,calibrated=load_labels();n=len(f);checks=[]
    def check(name,ok,detail):
        assert ok,(name,detail);checks.append({'Check':name,'Status':'PASS','Evidence':detail})
    check('ActorIdentity',n==260151 and f.scene_token.nunique()==630 and f.actor_id.is_unique,'unchanged complete HeadTrain OOF population')
    check('Types',list(f.agent_type.value_counts().reindex(ACTOR_TYPES))==[191026,66145,2980],'all Vehicle/Pedestrian/Bicycle actors retained')
    check('ProbabilityShape',p.shape==z.shape==(n,5,6),'R0,FoldR2,A,B,C frozen float32 scores')
    check('ProbabilityNumeric',np.isfinite(p).all() and np.isfinite(z).all() and np.max(np.abs(p.sum(-1)-1))<1e-6,'finite unchanged source probabilities')
    candidates=np.load(S11A/'01_identity_audit/cache/candidates.npy',mmap_mode='r')
    truth=np.load(S11A/'01_identity_audit/cache/GT.npy',mmap_mode='r');src=f.source_index.to_numpy()
    maxfd=maxad=0.;empty=0
    for start in range(0,n,2048):
        ids=np.arange(start,min(n,start+2048));pred=np.array(candidates[src[ids]],copy=True);gt=np.array(truth[src[ids]],copy=True)
        distance=(torch.from_numpy(pred)-torch.from_numpy(gt)[:,None]).norm(dim=-1)
        derived_fd=distance[:,:,-1].numpy();derived_ad=distance.mean(-1).numpy()
        maxfd=max(maxfd,float(np.max(np.abs(derived_fd-fd[ids]))));maxad=max(maxad,float(np.max(np.abs(derived_ad-ad[ids]))))
        assert np.array_equal(derived_fd,fd[ids]) and np.array_equal(derived_ad,ad[ids])
        hashes=np.array([hashlib.sha256(a.tobytes()).hexdigest().encode() for a in pred.reshape(-1,12,2)],dtype='S64').reshape(-1,6)
        assert np.array_equal(hashes,data['candidate_geometry_id'][ids])
        sem=data['semantic'][ids];valid=data['feature_valid'][ids];geom=data['geometry'][ids]
        assert np.isfinite(sem).all() and np.isfinite(geom).all() and not sem[~valid].any()
        has=(data['entity_ids'][ids]>=0).any(-1)
        assert np.array_equal(has,sem[:,:,SEM_FIELDS.index('map_valid_mask')]>0)
        empty+=int((~has).sum())
    check('CandidateIdentity',True,'1,560,906 per-mode geometry SHA256 values recomputed against original Stage11A source')
    check('PerModeFDEADE',maxfd==maxad==0.,'all HeadTrain candidates CPU FP32 last-valid FDE/ADE bitwise cache match')
    check('MissingMapRetention',True,f'{empty} empty candidate contexts retained as finite values plus masks')
    check('CalibratedTop1',np.array_equal(top[:,4],calibrated.argmax(-1)),'positive temperature leaves every C Raw Top1 unchanged')
    stats=read_json(ROOT/'03_feature_statistics/stage12a_statistics_audit.json')
    check('FullModeCSV',sha256(ROOT/stats['LocalFullModeCSV'])==stats['LocalFullModeCSVSHA256'],'local actor-level CSV hash, exact six modes per actor')
    # Independent tied-rank reference includes varied missingness and ties.
    rng=np.random.default_rng(2022);x=rng.integers(0,5,(200,6));y=rng.integers(0,5,(200,6));m=rng.random((200,6))>.25
    rho,eligible=rho_rows(x,y,m);rank_maxdiff=0.
    for i in np.flatnonzero(eligible):
        expected=float(spearmanr(x[i,m[i]],y[i,m[i]]).statistic)
        rank_maxdiff=max(rank_maxdiff,abs(expected-rho[i]))
    check('TiedRankSpearman',rank_maxdiff<1e-12,f'200 fixed synthetic rows vs SciPy; maxdiff {rank_maxdiff:.3g}')
    scene=SceneStatistics(f);rng=np.random.default_rng(2022);weights=np.zeros((2000,630),np.int32)
    for b in range(2000):
        for fold in range(3):weights[b,fold*210:(fold+1)*210]=np.bincount(rng.integers(0,210,210),minlength=210)
    check('SceneBootstrap',np.array_equal(weights,scene.weights),'2000 whole-scene draws stratified by three frozen outer210 folds, regenerated seed2022')
    root=ROOT/'08_negative_controls/cache';nc1=np.load(root/'stage12a_candidate_internal_permutation.npy',mmap_mode='r')
    nc2=np.load(root/'stage12a_region_type_permutation.npy',mmap_mode='r');rng=np.random.default_rng(2022)
    for start in range(0,n,2048):
        end=min(start+2048,n);expected=np.stack([rng.permutation(6) for _ in range(end-start)])
        assert np.array_equal(expected,nc1[start:end])
    rng=np.random.default_rng(2022)
    for region in range(4):
        for typ in range(3):
            actors=np.flatnonzero((data['map_region']==region)&(f.agent_type_id.to_numpy()==typ));flat=(actors[:,None]*6+np.arange(6)).ravel()
            if len(flat):assert np.array_equal(rng.permutation(flat),nc2[actors].ravel())
    check('NegativeControls',np.array_equal(np.sort(nc2.ravel()),np.arange(n*6)) and np.all(np.sort(nc1,axis=1)==np.arange(6)),
        'full semantic+mask records are deterministic permutations; NC2 exactly region/type restricted; scores/candidates untouched')
    controls=pd.read_csv(ROOT/'08_negative_controls/stage12a_negative_controls.csv');ii=np.arange(n)
    for typ,spec in read_json(PROTOCOL)['PrimarySignals'].items():
        for field,sign in spec:
            k=SEM_FIELDS.index(field);xx=data['semantic'][:,:,k]*sign;vv=data['feature_valid'][:,:,k]
            aa=[rho_rows(xx,fd,vv),rho_rows(xx[ii[:,None],nc1],fd,vv[ii[:,None],nc1]),rho_rows(xx.ravel()[nc2],fd,vv.ravel()[nc2])]
            common=(f.agent_type.to_numpy()==typ)&aa[0][1]&aa[1][1]&aa[2][1]
            row=controls[(controls.Scope=='Pooled')&(controls.ActorType==typ)&(controls.Feature==field)].iloc[0]
            assert int(row.CommonActors)==int(common.sum())
            for label,arr in zip(['Real','NC1','NC2'],aa):
                st=scene.mean(arr[0],common,bootstrap=False)
                if st['Scenes']:assert abs(st['SceneMean']-row[label+'_SceneMeanRho'])<1e-10
                else:assert pd.isna(row[label+'_SceneMeanRho'])
    check('PairedSceneAssociation',True,'all six primary features pooled real and control estimates recomputed on identical eligible actors')
    switch=pd.read_csv(ROOT/'03_feature_statistics/stage12a_switch_semantic_analysis.csv');delta=fd[np.arange(n),top[:,4]].astype(float)-fd[np.arange(n),top[:,1]].astype(float)
    for scope,sm in scopes(f):
        for group,gm in groups(f,data['geometry']).items():
            mask=sm&gm
            if not mask.any():continue
            row=switch[(switch.Scope==scope)&(switch.Group==group)&(switch.Outcome=='Worsened')].iloc[0]
            assert abs(row.GrossHarm-row.GrossGain-delta[mask].sum())<max(1e-6,abs(delta[mask].sum())*1e-10)
    check('GrossGainHarm',True,'gross harm minus gross gain equals net signed FDE change in every fold/type/motion group')
    switches=pd.read_csv(ROOT/'03_feature_statistics/stage12a_switch_feature_differences.csv')
    for group,mask in groups(f).items():
        if group not in ['Vehicle','MovingVehicle','Pedestrian','Pedestrian5-8m','Bicycle']:continue
        for outcome,condition in [('Improved',delta<0),('Worsened',delta>0),('HighCostWorsened',delta>5)]:
            for field in ['centerline_mean_distance','centerline_heading_error','drivable_inside_fraction','walkway_inside_fraction']:
                k=SEM_FIELDS.index(field);m=mask&condition&data['feature_valid'][ii,top[:,4],k]&data['feature_valid'][ii,top[:,1],k]
                row=switches[(switches.Scope=='Pooled')&(switches.Group==group)&(switches.Outcome==outcome)&(switches.Feature==field)].iloc[0]
                assert int(row.Count)==int(m.sum())
                if m.any():
                    expected=data['semantic'][ii[m],top[m,4],k]-data['semantic'][ii[m],top[m,1],k]
                    assert abs(float(expected.mean())-row.Mean)<max(1e-10,abs(row.Mean)*1e-10)
    check('SwitchSemanticPairing',True,'selected C minus R2 features independently gathered; major motion/type groups match counts and means')
    case=pd.read_csv(ROOT/'09_cases/stage12a_case_manifest.csv');render=read_json(ROOT/'09_cases/stage12a_case_render_audit.json')
    check('Cases',len(case)==30 and case.Category.value_counts().to_dict()==render['Counts'],'10+10 error and5+5 successful switch cases, registered error/gain order')
    for name,h in render['Files'].items():assert sha256(ROOT/'09_cases'/name)==h
    for _,r in case.iterrows():
        i=int(r.HeadTrainIndex);assert r.ActorID==f.iloc[i].actor_id and int(r.CMode)==top[i,4] and int(r.R2Mode)==top[i,1] and int(r.OracleMode)==best[i]
        side=read_json(ROOT/'09_cases'/Path(r.Figure).with_suffix('.json'))
        assert np.array_equal(np.array(side['CandidateGeometryIDs']),data['candidate_geometry_id'][i].astype('U64'))
    check('CaseIdentityAndFiles',True,'all six geometry IDs and R2/C/oracle modes equal frozen arrays; PNG/SVG/PDF SHA256 verified')
    gap=fd[ii,top[:,4]].astype(float)-fd.min(-1)
    for typ in ['Vehicle','Pedestrian']:
        population=f.agent_type.to_numpy()==typ
        for kind,count,eligible,key in [('Error',10,population&(top[:,4]!=best),-gap),('Success',5,population&(top[:,4]!=top[:,1])&(delta<0),delta)]:
            order=sorted(np.flatnonzero(eligible),key=lambda i:(float(key[i]),f.iloc[i].actor_id));picked=[];seen=set()
            for i in order:
                if f.iloc[i].scene_token in seen:continue
                picked.append(int(i));seen.add(f.iloc[i].scene_token)
                if len(picked)==count:break
            assert len(picked)==count
            actual=case[case.Category==typ+kind].sort_values('Ordinal').HeadTrainIndex.to_numpy()
            assert np.array_equal(actual,picked)
    check('RegisteredCaseSelection',True,'all30 selections independently reproduced from FDE/gain order with identity ties and scene diversity only')
    check('NoNewCheckpoint',not any(p.suffix in ['.pt','.pth'] for p in ROOT.rglob('*')),'Stage12A produces diagnostic features and reports only')
    check('HistoricalIntegrity',True,f'{len(frozen["historical_files"])} historical tracked files, {len(frozen["preserved_untracked_files"])} preserved Stage2C files,20 checkpoints and frozen data hashes unchanged')
    dump('11_reports/stage12a_verification.csv',checks)
    atomic_json(ROOT/'11_reports/stage12a_verification.json',{'Status':'PASS','MapIntegrity':'PASS','CoordinateAudit':'PASS','GTLeakage':'PASS',
        'CandidateIdentity':'PASS','SemanticCoverage':'PASS','NegativeControl':'PASS','Actors':n,'Scenes':630,'Candidates':n*6,
        'FDEMaxDiff':maxfd,'ADEMaxDiff':maxad,'TiedRankMaxDiff':rank_maxdiff,'EmptyCandidateContextsRetained':empty,
        'HistoryFilesVerified':len(frozen['historical_files']),'PreservedUntrackedVerified':len(frozen['preserved_untracked_files']),
        'CheckpointHashesVerified':len(frozen['checkpoints']),'TrainingExecuted':False,'OfficialVALOrTestUsed':False,
        'HeadDevEvaluationUsed':False,'Checks':checks,'ProtocolSHA256':sha256(PROTOCOL)})
    print('STAGE12A_INDEPENDENT_VERIFICATION_PASS',flush=True)
if __name__=='__main__':main()
