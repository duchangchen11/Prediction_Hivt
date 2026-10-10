"""Report every registered initialization, never select a seed from Outer scores."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'00_manifest'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from stage16_common import *
from stage16_write_analysis import table
def main():
 assert read_json(ROOT/'03_seed_stability/stage16_seed_evaluation_integrity.json')['Status']=='PASS'
 d=pd.read_csv(ROOT/'06_source_data/stage16_seed_metrics.csv');s=pd.read_csv(ROOT/'06_source_data/stage16_seed_summary.csv');c=pd.read_csv(ROOT/'06_source_data/stage16_seed_contrasts.csv');gate=read_json(ROOT/'03_seed_stability/stage16_all_frozen.json')
 reg=read_json(ROOT/'00_manifest/stage16_supplement_registration.json');hours=sum(x['Seconds'] for x in gate['Results'])/3600
 direction=[]
 for (group,cmp),a in c[c.Fold!=0].groupby(['Group','Comparison'],sort=False):
  direction.append({'Group':group,'Comparison':cmp,'FoldInitializationPairs':len(a),'NegativePairs':int((a.DeltaTop1FDE<0).sum()),'PositivePairs':int((a.DeltaTop1FDE>0).sum()),'EqualPairs':int((a.DeltaTop1FDE==0).sum()),'UnweightedMeanPairDelta':float(a.DeltaTop1FDE.mean())})
 dump(ROOT/'06_source_data/stage16_seed_direction_summary.csv',direction)
 record={'Status':'COMPLETE','FreshRankHeadFits':24,'FreshPredictorFits':0,'MeasuredHeadFitHours':hours,'OriginalReplicatesReused':12,'HeadInitializationsPerFoldAndModel':3,'FixedSamplerSeedByFold':[2022,2122,2222],'RegisteredInitSeedsByFold':[[2022,3022,4022],[2122,3122,4122],[2222,3222,4222]],'PredictorCrossSeedRobustnessEvaluated':False,'Stage15BPreregisteredDecisionsRevised':False,'DirectionSummary':direction}
 atomic_json(ROOT/'03_seed_stability/stage16_seed_decision.json',record)
 text=f'''# Stage16 ranking-head initialization stability

Completed all24 preregistered supplementary fits; reused the twelve original NG-A/NG-C/G-C/Matched-NG-C heads as initialization replicate0. No HiVT model was trained. The three Stage15B predictors, candidate coordinates, partitions, normalization and samefold R2 Bicycle route remained frozen.

This is **ranking-head initialization sensitivity conditional on fixed predictors and fixed data ordering**, not predictor cross-seed stability. The supplement was registered in commit `9e5ed32d8f5802ca49ce9e31b71854065fe6d0b7` before new Outer scores were calculated. All24 new heads were selected on InnerDev and globally frozen before unified Outer evaluation. The original Stage15B conclusions and primary bootstrap family remain unchanged.

## Registered seed plan

| Fold | Original init | New init1 | New init2 | Frozen predictor / sampler seed |
|---|---|---|---|---|
| 1 | 2022 | 3022 | 4022 | 2022 |
| 2 | 2122 | 3122 | 4122 | 2122 |
| 3 | 2222 | 3222 | 4222 | 2222 |

Only node/graph/adapter initialization seeds change. All models in a fold share the same per-epoch permutation and continuous1024-target carry, with exact original order SHA prefixes checked. AdamW0.001, weight_decay0.0001,FP32,microbatch128×8,max50 epochs,patience5, original Loss A/C and strict original relative Vehicle/Pedestrian InnerDev Sdev are unchanged. Shared constructors retain exact zero residual output at step0. Random initialization checks use zero optimizer steps; fitting uses InnerTrain378, selection InnerDev42, never Outer210/HeadDev70/official VAL. No seed was dropped, selected by Outer, or averaged into a new inference model.

Estimated before fitting: {reg['TimingEstimateHours']:.3f} head-fit GPU-reservation hours from original runs, {reg['EpochCapUpperEstimateHours']:.3f} hours at50-epoch caps. Actual24 fit wall times sum to **{hours:.3f} hours**, including Dev evaluation/I/O/checkpoint work, not pure CUDA kernel time. Peak allocated training memory across new jobs: {max(x['PeakGPUMemoryBytes'] for x in gate['Results'])/2**20:.1f}MiB. Existing RTX3080 environment was reused; no package or data installation occurred.

## Each initialization: Overall Top1FDE, meters

'''+table(d[d.Group=='Overall'],['Fold','Model','InitializationReplicate','InitializationSeed','Count','Top1FDE'])+'\n\n'
 text+='## Mean and sample standard deviation across three initializations\n\n'+table(s[s.Group=='Overall'],['Fold','Model','Initializations','CountPerInitialization','MeanTop1FDE','SDTop1FDE','MinTop1FDE','MaxTop1FDE'])+'\n\n'
 text+='Fold0 denotes actor-window-weighted pooling of the three fixedfold predictions, grouping corresponding replicate indices as registered. Its SD is across those **three pooled configurations**, not nine independently trained predictors; it depends on this predetermined cross-fold pairing. `stage16_seed_all_fold_combinations.csv` additionally enumerates all27 choices of the three recorded initializations acrossfolds, without selecting any configuration. Fold means/SD and paired directions are the clearer evidence of initialization sensitivity. Min/max columns describe variation and are not a checkpoint/seed choice.\n\n'
 text+='## Paired direction by fold and initialization\n\n'+table(c[c.Group=='Overall'],['Fold','InitializationReplicate','Comparison','DeltaTop1FDE','Direction'])+'\n\n'
 text+='## Direction counts: nine fold×initialization pairs per comparison\n\n'+table(pd.DataFrame(direction)[lambda x:x.Group.isin(['Overall','Vehicle','Pedestrian','MovingVehicle'])])+'\n\n'
 text+='Nine pairs reuse three predictors/scenes; they are not nine independent scientific replications. These counts and means are descriptive, without new significance claims or a replacement for scene-cluster uncertainty. Positive rows must remain visible even when the pooled mean favors the method.\n\n'
 text+='## Observed Overall directions\n\n'
 for row in direction:
  if row['Group']=='Overall':
   text+=f"{row['Comparison']}: {row['NegativePairs']} of {row['FoldInitializationPairs']} fold-initialization pairs favor the first model, {row['PositivePairs']} favor the second, and {row['EqualPairs']} tie. "
 text+='This count describes the recorded conditional sensitivity sample; it does not establish population significance or predictor-seed robustness.\n\n'
 for group in ('Vehicle','Pedestrian','MovingVehicle'):
  text+='## '+group+' initialization spread\n\n'+table(s[s.Group==group],['Fold','Model','CountPerInitialization','MeanTop1FDE','SDTop1FDE'])+'\n\n'
 text+='''## Limits and reproducibility

Candidate geometry and minFDE6 cannot change under these head-only fits. New outputs retain identity alignment and bitwise samefold R2 Bicycle scores/selections. The evaluation audit records those checks and all24 chosen checkpoint SHA256. Peak allocation excludes CUDA reserved memory and does not represent full-system predictor deployment memory.

Training source: `03_seed_stability/stage16_train_heads.py`; it ports the original optimizer loop into the Stage16 output scope while reusing original model classes and loss function bodies read-only. Registration includes code hashes, candidate manifest SHA, normalization SHA, predictor SHA and frozen R2 references. Every initialization has its full curve, batch-order digest, configuration, selected checkpoint SHA and measured resource record. New checkpoints/last-state recovery files remain local and never overwrite Stage15B checkpoints. Evaluation entry `stage16_evaluate_seeds.py` refuses to read Outer arrays before all24 heads freeze.

`06_source_data/stage16_seed_metrics.csv`, `stage16_seed_summary.csv`, `stage16_seed_contrasts.csv` and `stage16_seed_direction_summary.csv` provide every fold/group value. Main seven-model figures still use original frozen Stage15B results; supplementary S2 plots head-initialization spread. The smallest positive/negative changes should not be inflated into new scientific claims. Three initializations are a limited sensitivity sample. Predictor training, batch-order stochasticity, different candidate generators, external benchmarks and pristine held-out research generalization remain untested by this supplement.
'''
 (ROOT/'stage16_seed_stability.md').write_text(text)
 sci=ROOT/'stage16_scientific_summary.md';original=sci.read_text().split('\n## Completed initialization supplement')[0]
 original+='\n## Completed initialization supplement\n\n'+table(pd.DataFrame(direction)[lambda x:x.Group=='Overall'])+'\n\nAll24 new head fits and the12 original replicates are disclosed in the seed report. This is conditional initialization evidence; mixed directions are retained and do not revise the original preregistered Stage15B decisions.\n'
 exceptions=c[(c.Fold!=0)&(c.Group=='Overall')&(c.DeltaTop1FDE>0)]
 if len(exceptions):
  original+='\nOverall contrasts with a positive difference:\n\n'+table(exceptions,['Fold','InitializationReplicate','Comparison','DeltaTop1FDE'])+'\n\nThe capacity comparison is therefore not uniformly favorable under initialization changes. This limits a universal superiority claim while leaving the original preregistered conditional bootstrap result intact. No initialization is selected or discarded using these outcomes.\n'
 dist=pd.read_csv(ROOT/'06_source_data/stage16_selected_error_distributions.csv');original+='\n## MovingVehicle error distribution\n\n'+table(dist[(dist.Group=='MovingVehicle')&dist.Model.isin(['R0','NG-C','G-C'])],['Model','Count','MedianTop1FDE','P90Top1FDE','P99Top1FDE','MedianOracleMinFDE6','P90OracleMinFDE6'])+'\n\nAll distances are meters and distributions use the frozen actor-window denominator, not independent trajectories. The long upper tail remains after ranking.\n';sci.write_text(original)
 print('Seed report complete; actual fit hours',hours)
if __name__=='__main__':main()
