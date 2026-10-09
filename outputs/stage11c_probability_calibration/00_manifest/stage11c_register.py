"""Register all statistical choices before reading InnerDev model outputs."""
from stage11c_common import *
import subprocess, scipy
def main():
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT,text=True).strip()==BASE
    assert not REG.exists() and not PROTOCOL.exists()
    protocol={
      'Stage':'Stage11C','BaseCommit':BASE,'Branch':'stage11c/ranking-probability-calibration',
      'Method':'one fold-specific global scalar temperature, Vehicle/Pedestrian shared; Bicycle original FoldR2',
      'TrainNeuralNetworks':False,'WriteCheckpoints':False,'NewCalibratorsOrSearches':False,
      'Data':'HeadTrain630, original Stage11B 3 outer210 / inner378 / dev42 splits',
      'FitPartition':'corresponding InnerDev42 only; V/P fixed .5/.5 macro weights',
      'OracleLabel':'lowest-index argmin frozen FDE among six candidates',
      'FitObjective':'macro OracleBestMode NLL: .5 mean Vehicle + .5 mean Pedestrian',
      'Temperature':{'bounds':[1.,1000.],'space':'natural log T','method':'scipy.optimize.minimize_scalar bounded',
        'dtype':'float64','xatol':1e-10,'maxiter':1000,'endpoints_evaluated':True,
        'selection':'minimum objective among interior solution and exact endpoints; exact tie uses smaller T',
        'boundary_rule':'exact endpoint 1000 wins => BOUNDARY_HIT; never expand range'},
      'ProbabilityStorage':'V/P calibrated logits and probabilities FP64; Bicycle separate original FP32 blocks, bitwise unchanged; routed by observed actor type',
      'Precision':'stable FP64 shifted logsumexp for NLL/SoftCE; raw probabilities original Stage11B FP32 promoted exactly for statistics',
      'Top1':'lowest-index argmax of probabilities; require exact equality with raw probabilities and logits',
      'Brier':'sum across six classes (p_k - onehot_oracle_k)^2; unnormalized, range 0..2',
      'ECE':{'bins':15,'edges':'uniform j/15 for j=0..15','intervals':'[j/15,(j+1)/15), last includes 1',
        'event':'selected highest-probability mode equals lowest-index oracle-best FDE mode',
        'weight':'actor/window count','empty_bins':'zero contribution, reliability means left missing',
        'interpretation':'six-candidate oracle-mode classification calibration; not real-world trajectory/intent/collision/risk probability'},
      'Auxiliary':{'SoftCE':'q=softmax(-FDE/1m), -sum q log p',
        'ExpectedFDE':'sum p*FDE','ExpectedRegret':'sum p*(FDE-minFDE)',
        'NormalizedExpectedRegret':'sum p*cost / max(1m,mean six costs) using original Stage11B convention'},
      'Bootstrap':{'replicates':2000,'seed':2022,'unit':'paired whole scene, within each 210-scene outer fold, pooled630',
        'CI_percentiles':[2.5,97.5],'metrics':['OracleBestModeNLL','BrierScore','Top1ModeECE','ExpectedRegret'],
        'ECE':'recompute nonlinear ECE for each replicate from per-scene/bin confidence and oracle-event counts',
        'uncertainty':'descriptive, fixed fitted T/checkpoints; not refit within bootstrap; no independent test claim'},
      'Decision':{'Engineering':'frozen checkpoints, no model training, isolated three T, exact Top1/Bicycle, finite normalized outputs',
        'ClearWorsening':'positive delta with descriptive paired 95% CI entirely >0',
        'UsefulYES':'engineering PASS; legal T; merged V/P NLL both decrease; Brier not clearly worse for both; full ECE/concentration reporting; Top1 exact',
        'Conflict':'both types Brier clearly worse OR both types ECE clearly worse, despite NLL improvement => INCONCLUSIVE',
        'UsefulNO':'no improvement of both type NLLs or engineering failure',
        'ConcentrationReduced':'both type mean pmax and fraction pmax>0.99 decrease',
        'ExpectedRegretTradeoff':'must disclose; not alone a veto for oracle-event calibration',
        'Readiness':'YES and GO iff engineering PASS and CalibrationUseful YES; actual execution always STOP after Stage11C'},
      'Limitations':['InnerDev reused for checkpoint selection and temperature fit',
        'Stage11B OOF already inspected; this is OOF development evidence, not independent testing',
        'single observed GT future; FDE-derived oracle labels do not establish real-world risk probabilities'],
      'Forbidden':['HeadDev70','official VAL150','official test','model training','new semantic graph','Stage12'],
      'Versions':{'numpy':np.__version__,'scipy':scipy.__version__,'torch':torch.__version__,'pandas':pd.__version__},
      'Seed':2022}
    atomic_json(PROTOCOL,protocol)
    historic={}
    files=subprocess.check_output(['git','ls-tree','-r','--name-only',BASE],cwd=PROJECT,text=True).splitlines()
    for p in files: historic[p]=sha256(PROJECT/p)
    preserved={p:sha256(PROJECT/p) for p in subprocess.check_output(['git','ls-files','--others','--exclude-standard'],cwd=PROJECT,text=True).splitlines() if not p.startswith(str(ROOT.relative_to(PROJECT))+'/')}
    old=read_json(S11B/'00_manifest/stage11b_frozen_history.json')
    cps=dict(old['checkpoints']);allcps=read_json(S11B/'04_checkpoints/stage11b_all_frozen.json')
    assert allcps['Status']=='FROZEN_ALL_COMPLETE' and len(allcps['Checkpoints'])==12
    tab=pd.read_csv(S11B/'04_checkpoints/stage11b_checkpoint_manifest.csv')
    for row in allcps['Checkpoints']:
        assert sha256(PROJECT/row['Path'])==row['SHA256']
        match=tab[(tab.Fold==row['Fold'])&(tab.Model==row['Model'])];assert len(match)==1 and match.iloc[0].SHA256==row['SHA256']
        cps[row['Path']]=row['SHA256']
    splitnorm={str(p.relative_to(PROJECT)):sha256(p) for p in (S11B/'02_splits').glob('*.json')}
    data={}
    for p in [S11B/'01_preflight/cache/identities.csv',S11A/'01_identity_audit/cache/candidates.npy',
              *[S8/'01_training/cache'/n for n in ['arg0.npy','arg1.npy','arg2.npy','arg6.npy','fde.npy','ade.npy']],
              *[OOF/n for n in read_json(OOF/'complete.json')['cache_files']]]:
        data[str(p.relative_to(PROJECT))]=sha256(p)
    for n,h in read_json(OOF/'complete.json')['cache_files'].items(): assert data[str((OOF/n).relative_to(PROJECT))]==h
    f=frame();outer=[];sizes=[]
    for fold in (1,2,3):
        s=split(fold);assert [len(s[k]) for k in ('InnerTrain','InnerDev','OuterTest')]==[378,42,210]
        assert not set(s['InnerDev'])&set(s['OuterTest']) and not set(s['InnerDev'])&set(s['InnerTrain'])
        assert set(s['InnerTrain']+s['InnerDev']+s['OuterTest'])==set(f.scene_token)
        outer.extend(s['OuterTest']);sizes.append({'Fold':fold,**{k:int(len(indices(fold,k))) for k in ('InnerTrain','InnerDev','OuterTest')}})
    assert len(set(outer))==630
    frozen={'Status':'PASS','BaseCommit':BASE,'checkpoints':cps,'Stage11BCheckpoints':allcps['Checkpoints'],
       'Stage11BCheckpointManifestSHA256':sha256(S11B/'04_checkpoints/stage11b_checkpoint_manifest.csv'),
       'historical_files':historic,'preserved_untracked_files':preserved,'split_and_normalization':splitnorm,'data_files':data,
       'CandidateCoordinates':'same immutable Stage11A source array; no copying or modification',
       'Actors':len(f),'Scenes':630,'TypeCounts':{t:int((f.agent_type==t).sum()) for t in TYPES},'FoldActors':sizes,
       'OfficialVALTestOrHeadDevOpened':False}
    atomic_json(FROZEN,frozen)
    atomic_json(REG,{'Status':'REGISTERED_BEFORE_FIT','protocol_sha256':sha256(PROTOCOL),
       'requirements_sha256':sha256(ROOT/'00_manifest/stage11c_requirements.txt'),
       'frozen_manifest_sha256':sha256(FROZEN),'FitOrOOFAnalysisStarted':False})
    verify(history=True)
    print('REGISTERED_FROZEN_PASS',len(cps),len(historic),sizes,flush=True)
if __name__=='__main__': main()
