"""Frozen-output oracle-mode calibration; no calibration fitting or model forward."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'00_manifest'))
from stage16_common import *
def main():
 f,m,p,z=frozen_eval();reg=read_json(ROOT/'00_manifest/stage16_supplement_registration.json')
 edges=np.asarray(reg['ECEBins']['Edges']);fds=[];offset=0
 for fold in (1,2,3):
  folder=OLD/f'05_candidate_interface/cache/fold{fold}/OuterTest';fr=pd.read_csv(folder/'targets.csv')
  ids=['scene_token','sample_token','instance_token','node_index','fold','predictor_checkpoint_sha256']
  assert f.iloc[offset:offset+len(fr)][ids].reset_index(drop=True).equals(fr[ids])
  a=np.load(folder/'fde.npy',mmap_mode='r');assert a.shape==(len(fr),6);fds.append(a);offset+=len(fr)
 fd=np.concatenate(fds);oracle=fd.argmin(-1);ties=(fd==fd.min(-1)[:,None]).sum(-1)>1
 assert np.array_equal(fd.min(-1),m[:,0,FIELDS.index('minFDE6')])
 rows=[];bins=[];switch=[]
 for j,name in enumerate(MODELS):
  prob=np.asarray(p[:,j],np.float64);assert np.all(np.isfinite(prob)) and np.allclose(prob.sum(-1),1,atol=1e-6,rtol=0)
  selected=prob.argmax(-1);confidence=prob.max(-1);hit=selected==oracle;op=prob[np.arange(len(f)),oracle]
  # Evaluable only after GT labels were kept outside model inputs.
  assert np.array_equal(hit.astype(float),m[:,j,FIELDS.index('HitRate')]);assert np.allclose(op,m[:,j,FIELDS.index('OracleModeProbability')],atol=1e-7)
  entropy=-(prob*np.log(np.maximum(prob,1e-30))).sum(-1)
  brier=np.sum(prob**2,-1)-2*op+1
  binid=np.minimum(np.searchsorted(edges,confidence,side='right')-1,14)
  for fold in (0,1,2,3):
   for group,gm in groups(f).items():
    keep=gm&((f.fold.to_numpy()==fold) if fold else True);n=int(keep.sum())
    if not n:continue
    ece=0.
    for b in range(15):
     bm=keep&(binid==b);count=int(bm.sum())
     conf=float(confidence[bm].mean()) if count else None;acc=float(hit[bm].mean()) if count else None
     if count:ece+=count/n*abs(conf-acc)
     bins.append({'Fold':fold,'Group':group,'Model':name,'Bin':b,'Lower':edges[b],'Upper':edges[b+1],'Count':count,'MeanConfidence':conf,'OracleHitRate':acc})
    rows.append({'Fold':fold,'Group':group,'Model':name,'Count':n,'Scenes':int(f.loc[keep,'scene_token'].nunique()),'Top1FDE':float(m[keep,j,0].mean()),'Top1Probability':float(confidence[keep].mean()),'PredictionEntropy':float(entropy[keep].mean()),'OracleModeProbability':float(op[keep].mean()),'HitRate':float(hit[keep].mean()),'ECE15':float(ece),'BrierScore':float(brier[keep].mean()),'ConfidenceMinusHitRate':float(confidence[keep].mean()-hit[keep].mean()),'ProbabilityAbove0p95Fraction':float((confidence[keep]>.95).mean()),'WrongAndConfidenceAbove0p95Fraction':float((~hit[keep]&(confidence[keep]>.95)).mean()),'ExactOracleTieCount':int(ties[keep].sum())})
 for a,b in (('G-C','NG-C'),('NG-C','NG-A'),('G-C','G-A')):
  ia,ib=MODELS.index(a),MODELS.index(b);delta=m[:,ia,0]-m[:,ib,0]
  for group,gm in groups(f).items():
   for cls,cm in (('Improved',delta<0),('Worsened',delta>0),('Equal',delta==0)):
    mask=gm&cm;n=int(mask.sum())
    if n:switch.append({'Comparison':a+'-'+b,'Group':group,'Outcome':cls,'Count':n,'MeanDeltaTop1FDE':float(delta[mask].mean()),'MedianDeltaTop1FDE':float(np.median(delta[mask])),'P90AbsoluteDelta':float(np.quantile(np.abs(delta[mask]),.9)),'MeanA_Top1Probability':float(p[mask,ia].max(-1).mean()),'MeanB_Top1Probability':float(p[mask,ib].max(-1).mean())})
 dump(ROOT/'06_source_data/stage16_probability_metrics.csv',rows);dump(ROOT/'06_source_data/stage16_reliability_bins.csv',bins);dump(ROOT/'06_source_data/stage16_switch_probability.csv',switch)
 atomic_json(ROOT/'04_probability/stage16_probability_integrity.json',{'Status':'PASS','Rows':len(f),'Scenes':630,'K':6,'FoldIdentityExact':True,'OracleFDEExact':True,'OriginalHitRateExact':True,'OriginalOracleProbabilityMatched':True,'GTOnlyLabels':True,'CalibrationFits':0,'NoModelForward':True,'ECEBinRegistrationSHA256':sha256(ROOT/'00_manifest/stage16_supplement_registration.json'),'SourceMetricsSHA256':sha256(OLD/'08_evaluation/cache/metrics.npy'),'SourceProbabilitySHA256':sha256(OLD/'08_evaluation/cache/probabilities.npy'),'ExactOracleTieCount':int(ties.sum()),'Method':'fixed15 equal-width top-label ECE, categorical unnormalized K6 Brier; event FDE-oracle selection, not trajectory coverage or calibrated future density'})
 out=pd.DataFrame(rows);print(out[(out.Fold==0)&(out.Group=='Overall')][['Model','Top1FDE','ECE15','BrierScore','Top1Probability','HitRate']].to_string(index=False))
if __name__=='__main__':main()
