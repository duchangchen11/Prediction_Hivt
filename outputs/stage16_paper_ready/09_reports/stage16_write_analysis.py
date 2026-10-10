"""Reproducible probability and historical-scope reports from frozen source tables."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'00_manifest'))
from stage16_common import *
def table(frame,cols=None,digits=6):
 d=frame if cols is None else frame[cols]
 def fmt(v):return f'{v:.{digits}f}' if isinstance(v,(float,np.floating)) else str(v)
 return '\n'.join(['| '+' | '.join(d.columns)+' |','| '+' | '.join(['---']*len(d.columns))+' |',*['| '+' | '.join(fmt(v) for v in row)+' |' for row in d.itertuples(index=False,name=None)]])
def probability():
 d=pd.read_csv(ROOT/'06_source_data/stage16_probability_metrics.csv');p=d[(d.Fold==0)&(d.Group=='Overall')]
 a=read_json(ROOT/'04_probability/stage16_probability_integrity.json')
 text='''# Stage16 probability quality analysis

Frozen Stage15B probabilities show a selection–calibration tradeoff: Loss C improves selected endpoint error while producing substantially more concentrated and overconfident **oracle-mode selection** scores. This does not modify any Stage15B checkpoint, loss or preregistered conclusion.

Dataset: nuScenes official TRAIN scenes; custom internal three-fold scene-isolated CV, 630 Outer scenes and 260,151 full-horizon actor windows, K=6, nominal6s/12samples. This is not official nuScenes test evaluation. All comparisons use identical candidate coordinates and identities, with Bicycle routed to the samefold frozen R2 in NG/G/Matched models. Sources are the frozen probability/logit/metric arrays and each fold's detached candidate FDE labels.

## Event and measures

For actor-window i, the categorical label is `y_i=argmin_k endpointFDE(i,k)` over original candidate indices; exact ties select the lowest index. The event for top-label calibration is `argmax_k p_i(k)==y_i`, identical to Stage15B HitRate. It is an **offline candidate-oracle agreement event**, not probability of a naturally defined maneuver, within-2m prediction success, trajectory coverage, safety, or continuous future density.

Top1Probability=max(p); PredictionEntropy=-sum(p log p), in nats; OracleModeProbability=p[y]. ECE uses 15 equal-width bins fixed before calculation, [lower,upper) with 1 included in the last bin: sum(bin_count/N)*abs(mean_confidence-bin_event_rate). BrierScore=mean(sum_k(p_k-onehot(y)_k)^2), range0–2, without division by K. Empty bins contribute zero. These are actor-window-weighted descriptive summaries; overlapping windows are not independent repetitions. No calibration parameters/temperature/bin count were fitted or selected on OuterTest. No model forward was needed.

## Overall results

'''+table(p,['Model','Count','Top1FDE','Top1Probability','PredictionEntropy','OracleModeProbability','HitRate','ECE15','BrierScore'])+'\n\n'
 text+='## Concentration and wrong confident selections\n\n'+table(p,['Model','ConfidenceMinusHitRate','ProbabilityAbove0p95Fraction','WrongAndConfidenceAbove0p95Fraction'])+'\n\n'
 text+='The “wrong and confident” fraction uses all actor windows as denominator. Fractions are dimensionless; FDE is meters. Positive confidence-minus-hit denotes mean overconfidence, while ECE also captures varying bin direction. Loss A models are generally underconfident on this particular oracle label; they are not automatically calibrated simply because their ECE is smaller.\n\n'
 for group in ('Vehicle','Pedestrian','MovingVehicle','Bicycle'):
  g=d[(d.Fold==0)&(d.Group==group)]
  text+='## '+group+'\n\n'+table(g,['Model','Count','Top1FDE','Top1Probability','HitRate','ECE15','BrierScore'])+'\n\n'
 text+='## Fold check\n\n'+table(d[(d.Fold!=0)&(d.Group=='Overall')&d.Model.isin(['NG-A','NG-C','G-A','G-C'])],['Fold','Model','Count','Top1FDE','PredictionEntropy','Top1Probability','HitRate','ECE15','BrierScore'])+'\n\n'
 text+=f'''Exact endpoint-oracle ties occur in {a['ExactOracleTieCount']:,} actor windows. Near-equivalent static trajectories and arbitrary candidate identities limit interpretation of a unique one-hot oracle as a true categorical future outcome. This limitation does not erase the observed concentration but prevents calling these scores calibrated trajectory probabilities.

## Interpretation

C is linear expected normalized regret on the probability simplex, `sum_k p_k*(FDE_k-minFDE)/max(1m,mean regret)`, averaged over actors. It is a decision-risk objective rather than a strictly proper categorical likelihood loss; concentrating on the lowest-risk candidate is compatible with that objective. The probability analysis is an empirical association, not a causal decomposition of loss versus architecture.

G-C improves pooled Top1FDE over G-A and NG-C while its ECE/Brier are worse. NG-C similarly improves FDE over NG-A while ECE/Brier worsen. Thus the paper should present the softmax as **normalized ranking scores**, disclose overconfidence, and avoid a calibrated uncertainty claim. Keep existing Loss C unchanged. Future calibration would require separate training/development data and new registration; none was performed.

Reproduction: `04_probability/stage16_probability.py`. Full group/fold tables, fixed-bin counts and switch-conditioned summaries are in `06_source_data/stage16_probability_metrics.csv`, `stage16_reliability_bins.csv`, and `stage16_switch_probability.csv`. Integrity and source SHA are in `04_probability/stage16_probability_integrity.json`. Supplementary reliability figure uses the same bins. Results remain conditional on the frozen three predictors/18 heads; they do not assess predictor retraining variability.
'''
 (ROOT/'stage16_probability_analysis.md').write_text(text)
def scientific():
 a=pd.read_csv(PROJECT/'outputs/stage14a_paper_graph_ablation/05_evaluation/stage14a_ablation_metrics.csv');b=pd.read_csv(PROJECT/'outputs/stage14b_paper_validation/05_evaluation/stage14b_capacity_metrics.csv');c=pd.read_csv(OLD/'stage15b_end_to_end_metrics.csv');c=c[c.Fold=='Pooled']
 allrows=[]
 for stage,d,scope in [('Stage14A',a,'old TRAIN700 predictor; ranking-only OOF'),('Stage14B',b,'same old frozen predictor; capacity control'),('Stage15B',c,'new fold-isolated predictors and heads; internal CV')]:
  for _,x in d.iterrows():allrows.append({'Stage':stage,'Scope':scope,**{k:x[k] for k in ('Group','Model','Count','Top1FDE','Top1ADE','minFDE6','HitRate')}})
 full=pd.DataFrame(allrows);dump(ROOT/'06_source_data/stage16_historical_comparison.csv',allrows)
 ci=pd.read_csv(OLD/'stage15b_bootstrap_ci.csv');main=ci[(ci.Group=='Overall')&(ci.Metric=='Top1FDE')]
 text='''# Stage16 scientific summary

The two implemented methods remain supported by the frozen **internal scene-isolated CV** comparisons: candidate-conditioned future interaction messages and normalized regret-aware mode ranking. They reduce selected-mode error without changing the six candidate trajectories. Stage15B resolves predictor–Outer training overlap; it does not create a previously untouched research confirmation set, establish literature novelty, or constitute official nuScenes test performance.

## Evidence scope: Stage14A / Stage14B / Stage15B

Stage14A held one TRAIN700-trained Stage5A predictor fixed. Its ranking heads were scene OOF, but all630 evaluated scenes were in predictor training. Stage14B audited this limitation, designed scheme A, and added the 24,001-parameter target-only capacity control on the same old candidates; it did not retrain HiVT. Stage15B fitted three fresh 650,403-parameter Stage5A generators on378 InnerTrain scenes each;42 InnerDev selected checkpoints. Both predictor and ranker fitting, normalization and model selection excluded eachfold210 Outer scenes and70 quarantined HeadDev scenes. Outer union is630 scenes/260,151 actor windows. History-derived features and predicted futures enter the scorer; GT futures only provide detached training labels, Dev selection and offline evaluation.

Official VAL150 historically informed predictor selection and method development. The TRAIN scenes and earlier internal results also informed development. Therefore even scene-isolated retraining is an internal validation of the frozen procedure, not pristine independent scientific confirmation. Scheme A uses predictor training-in candidates for the head's InnerTrain; it does not remove all train-in versus inference candidate distribution shift. Predictors differ across folds in both training scenes and seeds, so these folds cannot isolate predictor-seed variance.

## Pooled Overall metrics, meters

'''+table(full[(full.Group=='Overall')&full.Model.isin(['NG-A','NG-C','G-A','G-C','Matched-NG-C'])],['Stage','Model','Count','Top1FDE','Top1ADE','minFDE6','HitRate'])+'\n\n'
 text+='## Frozen Stage15B primary evidence\n\n'+table(main,['Comparison','Delta','BonferroniCILower','BonferroniCIUpper','NegativeFolds'])+'\n\n'
 text+='''Each negative delta favors the first model. All three frozen co-primary comparisons retain their original support decision: 2000 paired whole-scene bootstrap draws, seed2022,210 sampled scenes independently in each fold, actor-window-weighted aggregate; family3 Bonferroni98.333% intervals. These intervals condition on trained models. Stage14A family4 and Stage14B family3 intervals are not interchangeable; old and new estimates must not be pooled as independent replications because they reuse scenes and development history.

G-C−NG-C Overall is −0.055221m in Stage14A/14B versus −0.045382m in Stage15B; graph direction is retained with a smaller absolute gain. NG-C−NG-A is −0.126209m before versus −0.069239m after isolation. G-C−Matched-NG-C is −0.058842m before versus −0.022523m after isolation. Matching active parameter count retains evidence beyond this particular capacity control, but cannot prove all capacity/optimization confounding is eliminated: matching counts does not match function class or training dynamics. The nominal65-parameter difference is0.2701%, and the common scalar score bias is softmax-unidentifiable in all heads.

The isolated common oracle minFDE6 is1.266202m, versus1.258135m on the old generator. Different predictor geometry and Dev selection prevent attributing the between-stage difference solely to isolation. The original graph/loss ablations and capacity control remain historical analyses; Stage15B supplies the principal end-to-end internal CV evidence.

## MovingVehicle absolute error

MovingVehicle uses t0 `vehicle.moving`, not a GT-speed-filtered selection. N=41,728. G-C Top1FDE=10.210756m, versus NG-C10.470631m and oracle5.343627m. Its graph gain −0.259876m has descriptive95% scene CI[−0.364893,−0.173174] and negative direction in all three folds; this secondary group is outside the registered primary family. The oracle's large absolute error means fixed-candidate quality is itself limiting; ranking cannot close missing-candidate error. Residual selection gap remains4.867129m. Six-second horizon and heterogeneous moving scenes are relevant context, but these summaries do not establish the cause of the large errors. Report distributions/median/tails from frozen source data rather than hiding them behind an Overall mean dominated by parked vehicles.

## Pedestrian modest/uneven gain

N=66,145. G-C1.330484m vs NG-C1.334883m gives −0.004399m; descriptive95% scene CI[−0.008611,−0.000027], with fold3 slightly positive (+0.000107m). The old graph gain was −0.012584m. NG-C−NG-A Pedestrian CI crosses zero and fold3 is worse (+0.009289m). Do not describe uniformly strong multi-type gains. Bicycle uses identical samefold R2 selections across all NG/G/Matched variants; its result validates routing consistency, not an interaction-graph improvement.

## Concentration and costly mode switches

Loss C's pooled G-C entropy=0.132957nats and Top1Probability=0.942559, compared with G-A entropy1.459550 and probability0.299848. The dedicated probability report establishes a tradeoff against the explicitly defined endpoint-oracle label, using fixed-bin ECE and categorical Brier; no calibration was fitted. These are ranking scores and must not be sold as well-calibrated maneuver probabilities.

G-C vs NG-C improves23,914 actor windows, worsens19,956 and ties216,281. In MovingVehicle,7,813 improve and5,718 worsen; in Pedestrian,6,577 improve and5,756 worsen. Four real-coordinate examples deliberately include extreme improvements and failures. They illustrate mode selection under identical geometry and are not representative prevalence estimates. High confidence can coexist with a costly wrong switch; switching summaries and harm tails remain disclosed. No causal claim follows from one case.

## Compute burden

Frozen Stage15B CUDA head-only forward for1024 normalized cached targets is approximately G-C10.3–10.4ms, NG-C4.7–4.8ms, Matched6.5ms and R2≈2.0ms on RTX3080. This excludes predictor, feature construction, transfers and Bicycle route. The engineering full-system benchmark for16 fixed InnerDev windows measures G-C0.3012/0.2103/0.1726s in folds1/2/3, versus R20.2809/0.1853/0.1455s; G-C adds20–27ms relative to R2 in that path. Actor counts differ byfold (1014/392/163 current actors); these are not comparable scene-independent deployment FPS or optimized real-time guarantees. Full timings include predictor, ego conversion, CPU feature construction, normalization, transfer, score head and route, but exclude raw I/O/audit hashing. Benchmark peak memory includes resident pools/models.

Per-fold total inference parameters: R0 650,403; R2 651,076; NG658,501; G675,142; Matched675,077, including Bicycle R2 where applicable. The generator is frozen during head optimization; actual head trainable counts are673/7,425/24,066/24,001. No ensemble of the three folds is claimed.

## Paper framing and remaining boundaries

Keep the paper focused on fixed-candidate selection, the future interaction graph and normalized regret-aware ranking. Describe the adapted Stage5A generator and R2 routing honestly as existing components; NG means no **post-prediction** graph message, not an interaction-free HiVT backbone. Show common oracle geometry, paired scene uncertainty, grouped harms, initialization sensitivity and calibration limitations. No third module, changed loss, predictor retraining, new checkpoint choice or Outer-based tuning is introduced.

Two external scoring-component adaptations were reviewed, not evaluated; their architecture/loss/source differences are explicit. The seven current baselines support controlled mechanism claims, but a reproduced external scorer or another genuinely comparable external prediction baseline remains a reviewer-facing limitation. Cross-protocol literature numbers are excluded. An official submission would additionally need official targets/split, global-coordinate export, supported K and official metrics; current custom K6 three-type endpoint protocol is not official test.

Supplementary seed results are reported separately after all24 new heads freeze; they measure only initialization sensitivity conditional on fixed predictors and fixed sample order, and cannot overwrite Stage15B's preregistered outcomes. Paper-readiness is a bounded internal-evidence package, subject to brain-AI review, literature novelty appraisal and journal selection.
'''
 contribution=pd.read_csv(ROOT/'06_source_data/stage16_error_contribution.csv')
 text+='\n## Actor-weighted composition of the Overall change\n\n'+table(contribution[contribution.Comparison=='G-C-NG-C'],['ExclusiveGroup','Count','FractionOfOverall','MeanDeltaTop1FDE','WeightedContributionMeters'])+'\n\n'
 text+='The four groups are disjoint and exhaustive; OtherVehicle means Vehicle without the t0 moving attribute, including stopped, parked and other observed states. WeightedContributionMeters equals group fraction times group mean paired FDE difference. These signed contributions sum to the frozen Overall G-C minus NG-C delta. This is an arithmetic decomposition of the actor-window-weighted mean, not a causal attribution or separate statistical test. The MovingVehicle contribution is large despite its smaller window fraction; it does not remove the large remaining absolute errors. The full source table also records the capacity and loss contrasts.\n'
 (ROOT/'stage16_scientific_summary.md').write_text(text)
if __name__=='__main__':probability();scientific();print('probability and scientific reports generated from frozen sources')
