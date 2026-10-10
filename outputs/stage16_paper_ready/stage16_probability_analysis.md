# Stage16 probability quality analysis

Frozen Stage15B probabilities show a selection–calibration tradeoff: Loss C improves selected endpoint error while producing substantially more concentrated and overconfident **oracle-mode selection** scores. This does not modify any Stage15B checkpoint, loss or preregistered conclusion.

Dataset: nuScenes official TRAIN scenes; custom internal three-fold scene-isolated CV, 630 Outer scenes and 260,151 full-horizon actor windows, K=6, nominal6s/12samples. This is not official nuScenes test evaluation. All comparisons use identical candidate coordinates and identities, with Bicycle routed to the samefold frozen R2 in NG/G/Matched models. Sources are the frozen probability/logit/metric arrays and each fold's detached candidate FDE labels.

## Event and measures

For actor-window i, the categorical label is `y_i=argmin_k endpointFDE(i,k)` over original candidate indices; exact ties select the lowest index. The event for top-label calibration is `argmax_k p_i(k)==y_i`, identical to Stage15B HitRate. It is an **offline candidate-oracle agreement event**, not probability of a naturally defined maneuver, within-2m prediction success, trajectory coverage, safety, or continuous future density.

Top1Probability=max(p); PredictionEntropy=-sum(p log p), in nats; OracleModeProbability=p[y]. ECE uses 15 equal-width bins fixed before calculation, [lower,upper) with 1 included in the last bin: sum(bin_count/N)*abs(mean_confidence-bin_event_rate). BrierScore=mean(sum_k(p_k-onehot(y)_k)^2), range0–2, without division by K. Empty bins contribute zero. These are actor-window-weighted descriptive summaries; overlapping windows are not independent repetitions. No calibration parameters/temperature/bin count were fitted or selected on OuterTest. No model forward was needed.

## Overall results

| Model | Count | Top1FDE | Top1Probability | PredictionEntropy | OracleModeProbability | HitRate | ECE15 | BrierScore |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R0 | 260151 | 2.372741 | 0.239331 | 1.630392 | 0.223405 | 0.451830 | 0.213904 | 0.761162 |
| R2 | 260151 | 2.346574 | 0.277057 | 1.514176 | 0.248595 | 0.455017 | 0.178037 | 0.737966 |
| NG-A | 260151 | 2.375782 | 0.281256 | 1.496853 | 0.250995 | 0.439276 | 0.158690 | 0.737091 |
| NG-C | 260151 | 2.306543 | 0.879821 | 0.269722 | 0.452412 | 0.464492 | 0.419706 | 0.935351 |
| G-A | 260151 | 2.361641 | 0.299848 | 1.459550 | 0.259091 | 0.440198 | 0.147497 | 0.731744 |
| G-C | 260151 | 2.261161 | 0.942559 | 0.132957 | 0.466208 | 0.470369 | 0.477292 | 0.988606 |
| Matched-NG-C | 260151 | 2.283684 | 0.923259 | 0.175243 | 0.462890 | 0.467475 | 0.460852 | 0.969263 |

## Concentration and wrong confident selections

| Model | ConfidenceMinusHitRate | ProbabilityAbove0p95Fraction | WrongAndConfidenceAbove0p95Fraction |
| --- | --- | --- | --- |
| R0 | -0.212499 | 0.000000 | 0.000000 |
| R2 | -0.177959 | 0.000000 | 0.000000 |
| NG-A | -0.158020 | 0.000000 | 0.000000 |
| NG-C | 0.415329 | 0.604760 | 0.292419 |
| G-A | -0.140350 | 0.000027 | 0.000008 |
| G-C | 0.472190 | 0.785655 | 0.391519 |
| Matched-NG-C | 0.455784 | 0.717153 | 0.351796 |

The “wrong and confident” fraction uses all actor windows as denominator. Fractions are dimensionless; FDE is meters. Positive confidence-minus-hit denotes mean overconfidence, while ECE also captures varying bin direction. Loss A models are generally underconfident on this particular oracle label; they are not automatically calibrated simply because their ECE is smaller.

## Vehicle

| Model | Count | Top1FDE | Top1Probability | HitRate | ECE15 | BrierScore |
| --- | --- | --- | --- | --- | --- | --- |
| R0 | 191026 | 2.731294 | 0.242616 | 0.489389 | 0.248128 | 0.756912 |
| R2 | 191026 | 2.698962 | 0.276309 | 0.491310 | 0.215159 | 0.734113 |
| NG-A | 191026 | 2.741044 | 0.281923 | 0.466460 | 0.185404 | 0.733941 |
| NG-C | 191026 | 2.648431 | 0.881711 | 0.497199 | 0.384530 | 0.870735 |
| G-A | 191026 | 2.720192 | 0.302536 | 0.464895 | 0.171827 | 0.728903 |
| G-C | 191026 | 2.588150 | 0.949902 | 0.502497 | 0.447408 | 0.926881 |
| Matched-NG-C | 191026 | 2.616192 | 0.935545 | 0.501680 | 0.433865 | 0.910885 |

## Pedestrian

| Model | Count | Top1FDE | Top1Probability | HitRate | ECE15 | BrierScore |
| --- | --- | --- | --- | --- | --- | --- |
| R0 | 66145 | 1.353283 | 0.229713 | 0.341356 | 0.113294 | 0.773705 |
| R2 | 66145 | 1.346393 | 0.279167 | 0.348054 | 0.068887 | 0.749472 |
| NG-A | 66145 | 1.339737 | 0.279467 | 0.357911 | 0.078761 | 0.746526 |
| NG-C | 66145 | 1.334883 | 0.901468 | 0.368312 | 0.533156 | 1.131229 |
| G-A | 66145 | 1.344339 | 0.293063 | 0.366059 | 0.073813 | 0.740045 |
| G-C | 66145 | 1.330484 | 0.951283 | 0.376128 | 0.575232 | 1.178534 |
| Matched-NG-C | 66145 | 1.338082 | 0.916840 | 0.367103 | 0.549737 | 1.148658 |

## MovingVehicle

| Model | Count | Top1FDE | Top1Probability | HitRate | ECE15 | BrierScore |
| --- | --- | --- | --- | --- | --- | --- |
| R0 | 41728 | 10.874690 | 0.224987 | 0.214101 | 0.011260 | 0.823780 |
| R2 | 41728 | 10.724894 | 0.239868 | 0.221338 | 0.018565 | 0.824845 |
| NG-A | 41728 | 10.863803 | 0.269351 | 0.248179 | 0.021217 | 0.812448 |
| NG-C | 41728 | 10.470631 | 0.874510 | 0.222153 | 0.652370 | 1.390336 |
| G-A | 41728 | 10.714765 | 0.334557 | 0.273989 | 0.060568 | 0.802017 |
| G-C | 41728 | 10.210756 | 0.943385 | 0.234087 | 0.709338 | 1.451459 |
| Matched-NG-C | 41728 | 10.324218 | 0.929852 | 0.227138 | 0.702714 | 1.446703 |

## Bicycle

| Model | Count | Top1FDE | Top1Probability | HitRate | ECE15 | BrierScore |
| --- | --- | --- | --- | --- | --- | --- |
| R0 | 2980 | 2.016767 | 0.242202 | 0.496309 | 0.254107 | 0.755143 |
| R2 | 2980 | 1.957885 | 0.278165 | 0.502685 | 0.224858 | 0.729593 |
| NG-A | 2980 | 1.957885 | 0.278165 | 0.502685 | 0.224858 | 0.729593 |
| NG-C | 2980 | 1.957885 | 0.278165 | 0.502685 | 0.224858 | 0.729593 |
| G-A | 2980 | 1.957885 | 0.278165 | 0.502685 | 0.224858 | 0.729593 |
| G-C | 2980 | 1.957885 | 0.278165 | 0.502685 | 0.224858 | 0.729593 |
| Matched-NG-C | 2980 | 1.957885 | 0.278165 | 0.502685 | 0.224858 | 0.729593 |

## Fold check

| Fold | Model | Count | Top1FDE | PredictionEntropy | Top1Probability | HitRate | ECE15 | BrierScore |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | NG-A | 82673 | 2.222825 | 1.380953 | 0.312726 | 0.449470 | 0.136745 | 0.700207 |
| 2 | NG-A | 83711 | 2.517525 | 1.511270 | 0.272733 | 0.413840 | 0.144956 | 0.751859 |
| 3 | NG-A | 93767 | 2.384100 | 1.586171 | 0.261117 | 0.452995 | 0.193070 | 0.756427 |
| 1 | NG-C | 82673 | 2.193321 | 0.093970 | 0.959817 | 0.512695 | 0.452209 | 0.926961 |
| 2 | NG-C | 83711 | 2.365157 | 0.213951 | 0.907181 | 0.414020 | 0.499245 | 1.049106 |
| 3 | NG-C | 93767 | 2.354042 | 0.474470 | 0.784864 | 0.467051 | 0.322229 | 0.841192 |
| 1 | G-A | 82673 | 2.283581 | 1.326849 | 0.338781 | 0.482370 | 0.157756 | 0.696600 |
| 2 | G-A | 83711 | 2.463382 | 1.490426 | 0.285477 | 0.394608 | 0.120529 | 0.747664 |
| 3 | G-A | 93767 | 2.339635 | 1.548986 | 0.278352 | 0.443717 | 0.167810 | 0.748518 |
| 1 | G-C | 82673 | 2.145866 | 0.089428 | 0.961898 | 0.515416 | 0.451629 | 0.924103 |
| 2 | G-C | 83711 | 2.335707 | 0.114374 | 0.950025 | 0.423433 | 0.532718 | 1.092646 |
| 3 | G-C | 93767 | 2.296264 | 0.187925 | 0.918841 | 0.472554 | 0.450870 | 0.952594 |

Exact endpoint-oracle ties occur in 0 actor windows. Near-equivalent static trajectories and arbitrary candidate identities limit interpretation of a unique one-hot oracle as a true categorical future outcome. This limitation does not erase the observed concentration but prevents calling these scores calibrated trajectory probabilities.

## Interpretation

C is linear expected normalized regret on the probability simplex, `sum_k p_k*(FDE_k-minFDE)/max(1m,mean regret)`, averaged over actors. It is a decision-risk objective rather than a strictly proper categorical likelihood loss; concentrating on the lowest-risk candidate is compatible with that objective. The probability analysis is an empirical association, not a causal decomposition of loss versus architecture.

G-C improves pooled Top1FDE over G-A and NG-C while its ECE/Brier are worse. NG-C similarly improves FDE over NG-A while ECE/Brier worsen. Thus the paper should present the softmax as **normalized ranking scores**, disclose overconfidence, and avoid a calibrated uncertainty claim. Keep existing Loss C unchanged. Future calibration would require separate training/development data and new registration; none was performed.

Reproduction: `04_probability/stage16_probability.py`. Full group/fold tables, fixed-bin counts and switch-conditioned summaries are in `06_source_data/stage16_probability_metrics.csv`, `stage16_reliability_bins.csv`, and `stage16_switch_probability.csv`. Integrity and source SHA are in `04_probability/stage16_probability_integrity.json`. Supplementary reliability figure uses the same bins. Results remain conditional on the frozen three predictors/18 heads; they do not assess predictor retraining variability.
