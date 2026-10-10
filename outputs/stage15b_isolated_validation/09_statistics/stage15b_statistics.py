"""Registered paired whole-scene bootstrap and fixed three-primary decision family."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'00_manifest'),str(ROOT/'05_candidate_interface'),str(ROOT/'06_rank_training'),str(ROOT/'08_evaluation')]
from stage15b_evaluate import *
COMPARISONS=(('G-C','NG-C'),('G-C','Matched-NG-C'),('NG-C','NG-A'))


def main():
 cache=ROOT/'08_evaluation/cache';audit=read_json(cache/'complete.json');assert audit['Status']=='PASS'
 for name,h in audit['SHA256'].items():assert sha256(cache/name)==h
 f=pd.read_csv(cache/'actor_records.csv');v=np.load(cache/'metrics.npy',mmap_mode='r')
 scene_order=[s for k in (1,2,3) for s in sorted(read_json(PROTOCOL)['folds'][k-1]['parts']['OuterTest'])]
 assert len(set(scene_order))==630;lookup={s:i for i,s in enumerate(scene_order)};codes=f.scene_token.map(lookup).to_numpy()
 rng=np.random.default_rng(2022);weights=np.zeros((2000,630),np.int32)
 for r in range(2000):
  for k in range(3):weights[r,k*210:(k+1)*210]=np.bincount(rng.integers(0,210,210),minlength=210)
 dest=ROOT/'09_statistics/cache';dest.mkdir(parents=True,exist_ok=True);np.save(dest/'bootstrap_scene_weights.npy',weights)
 rows=[];foldrows=[]
 for group,keep in groups(f).items():
  count=int(keep.sum());den=np.bincount(codes[keep],minlength=630).astype(np.float64);bootden=weights@den;assert (bootden>0).all()
  for metric in ('Top1FDE','Top1ADE','HitRate','minFDE6'):
   j=FIELDS.index(metric);x=np.array(v[keep,:,j]);sums=np.stack([np.bincount(codes[keep],weights=x[:,i],minlength=630) for i in range(len(MODELS))],-1)
   bootmeans=weights@sums/bootden[:,None]
   for a,b in COMPARISONS:
    ai,bi=MODELS.index(a),MODELS.index(b);d=x[:,ai]-x[:,bi];draw=bootmeans[:,ai]-bootmeans[:,bi]
    ci=np.quantile(draw,[.025,.975]);primary=group=='Overall' and metric=='Top1FDE';adj=np.quantile(draw,[.05/6,1-.05/6]) if primary else (None,None)
    row={'Group':group,'Comparison':a+'-'+b,'Metric':metric,'Count':count,'Scenes':int((den>0).sum()),'Delta':float(d.mean()),
         'CI95Lower':float(ci[0]),'CI95Upper':float(ci[1]),'CoPrimary':primary,'FamilySize':3 if primary else None,
         'BonferroniCILower':None if adj[0] is None else float(adj[0]),'BonferroniCIUpper':None if adj[1] is None else float(adj[1]),
         'Replicates':2000,'Seed':2022,'Unit':'paired whole scenes independently210 within each fold','Scope':'new internal scene-isolated end-to-end CV; conditional on trained models'}
    # Corresponding fold-specific scene intervals use the same draw weights.
    for k in (1,2,3):
     kk=keep&(f.fold.to_numpy()==k);delta=float((v[kk,ai,j]-v[kk,bi,j]).mean());row[f'Fold{k}Delta']=delta
     sl=slice((k-1)*210,k*210);dd=(weights[:,sl]@(sums[sl,ai]-sums[sl,bi]))/(weights[:,sl]@den[sl]);fc=np.quantile(dd,[.025,.975])
     foldrows.append({'Group':group,'Comparison':a+'-'+b,'Metric':metric,'Fold':k,'Count':int(kk.sum()),'Delta':delta,'CI95Lower':float(fc[0]),'CI95Upper':float(fc[1])})
    row['NegativeFolds']=sum(row[f'Fold{k}Delta']<0 for k in (1,2,3));rows.append(row)
 dump(ROOT/'stage15b_bootstrap_ci.csv',rows);dump(ROOT/'09_statistics/stage15b_fold_comparisons.csv',foldrows)
 decisions={}
 for name,c in read_json(PROTOCOL)['decisions'].items():
  row=next(r for r in rows if r['CoPrimary'] and r['Comparison']==c['Contrast']);supported=row['Delta']<0 and row['BonferroniCIUpper']<0 and row['NegativeFolds']>=2
  decisions[name]={'Decision':'SUPPORTED' if supported else 'NOT_SUPPORTED','RegisteredRule':c['Rule'],'Evidence':row}
 atomic_json(ROOT/'09_statistics/stage15b_decisions.json',{'Status':'COMPLETE','Decisions':decisions,'HistoricalExposureLimitation':True,'Uncertainty':'fixed three trained folds, not seed retraining variability','WeightsSHA256':sha256(dest/'bootstrap_scene_weights.npy')})
 print('STATISTICAL_DECISIONS', {k:v['Decision'] for k,v in decisions.items()},flush=True)

if __name__=='__main__':
 torch.set_num_threads(4);main()
