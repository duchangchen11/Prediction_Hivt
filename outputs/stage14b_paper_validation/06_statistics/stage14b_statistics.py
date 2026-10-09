"""Three registered paired contrasts; known and new results clearly separated."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'00_protocol'))
from stage14b_common import *
COMPARISONS=(('G-C','NG-C'),('G-C','Matched-NG-C'),('NG-C','NG-A'))
def main():
    verify(history=True);audit=read_json(ROOT/'05_evaluation/stage14b_identity_audit.json');assert audit['Status']=='PASS'
    dest=ROOT/'05_evaluation/cache'
    for name,h in audit['cache_files'].items():assert sha256(dest/name)==h
    f=pd.read_csv(dest/'stage14b_actor_records.csv',dtype={'future_mask_bits':str})
    old=np.load(S14A/'05_evaluation/cache/stage14a_oof_metrics.npy',mmap_mode='r');new=np.load(dest/'stage14b_matched_metrics.npy',mmap_mode='r')
    fd=np.column_stack((old[:,:,0],new[:,0]));scene_order=[s for k in (1,2,3) for s in sorted(split(k)['OuterTest'])]
    assert len(set(scene_order))==630 and f.scene_token.nunique()==630
    lookup={s:i for i,s in enumerate(scene_order)};codes=f.scene_token.map(lookup).to_numpy()
    rng=np.random.default_rng(2022);weights=np.zeros((2000,630),np.int32)
    for r in range(2000):
        for k in range(3):weights[r,k*210:(k+1)*210]=np.bincount(rng.integers(0,210,210),minlength=210)
    prior_weights=S14A/'06_bootstrap/cache/stage14a_scene_bootstrap_weights.npy'
    assert np.array_equal(weights,np.load(prior_weights,mmap_mode='r'))
    rows=[];foldrows=[];masksets=groups(f)
    for group,keep in masksets.items():
        count=int(keep.sum());den=np.bincount(codes[keep],minlength=630).astype(np.float64);bootden=weights@den
        assert (bootden>0).all()
        sums=np.stack([np.bincount(codes[keep],weights=fd[keep,j],minlength=630) for j in range(5)],-1)
        means=weights@sums/bootden[:,None]
        for a,b in COMPARISONS:
            ai,bi=MODELS.index(a),MODELS.index(b);d=fd[keep,ai]-fd[keep,bi];draw=means[:,ai]-means[:,bi]
            ci=np.quantile(draw,[.025,.975]);primary=group=='Overall';adj=np.quantile(draw,[.05/6,1-.05/6]) if primary else (None,None)
            row=dict(Group=group,Comparison=a+'-'+b,Count=count,Scenes=int((den>0).sum()),DeltaTop1FDE=float(d.mean()),CI95Lower=float(ci[0]),CI95Upper=float(ci[1]),CoPrimary=primary,FamilySize=3 if primary else None,BonferroniCILower=None if adj[0] is None else float(adj[0]),BonferroniCIUpper=None if adj[1] is None else float(adj[1]),ResultOrigin='new capacity contrast' if 'Matched-NG-C' in (a,b) else 'historical Stage14A contrast reused; already known',Scope='internal frozen-predictor ranking OOF; no independent end-to-end claim')
            for k in (1,2,3):
                kk=keep&(f.Fold.to_numpy()==k);v=float((fd[kk,ai]-fd[kk,bi]).mean());row[f'Fold{k}DeltaTop1FDE']=v
                foldrows.append(dict(Group=group,Comparison=a+'-'+b,Fold=k,Count=int(kk.sum()),DeltaTop1FDE=v))
            row['NegativeFolds']=sum(row[f'Fold{k}DeltaTop1FDE']<0 for k in (1,2,3));rows.append(row)
    dump('06_statistics/stage14b_bootstrap_ci.csv',rows);dump('06_statistics/stage14b_fold_comparisons.csv',foldrows)
    primary=next(r for r in rows if r['Group']=='Overall' and r['Comparison']=='G-C-Matched-NG-C')
    supported=primary['DeltaTop1FDE']<0 and primary['BonferroniCIUpper']<0 and primary['NegativeFolds']>=2
    atomic_json(ROOT/'06_statistics/stage14b_capacity_evidence.json',dict(Status='PASS',CapacityAlternativeNotSufficient='SUPPORTED' if supported else 'NOT_ESTABLISHED',RegisteredCapacityComparison=primary,Interpretation='approximate capacity control; does not establish causal effect or eliminate all architecture/optimization differences',Replicates=2000,Seed=2022,Unit='paired whole scene within each fold',FamilySize=3,AdjustedPercentiles=[100*.05/6,100*(1-.05/6)],HistoricalComparisonsAlreadyKnown=True))
    verify(history=True);print('STAGE14B_CAPACITY_BOOTSTRAP_PASS',primary,flush=True)
if __name__=='__main__':main()
