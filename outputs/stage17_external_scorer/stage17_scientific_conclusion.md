# Stage17 — scientific conclusion

G-C has lower pooled Top1FDE than the adapted TNT scorer: Adapted TNT Overall Top1FDE=2.421995 m, G-C=2.261161 m, NG-C=2.306543 m. G-C−TNT=-0.160834 m;95% descriptive paired scene-cluster CI[-0.191230,-0.133289]. Fold differences=-0.064171/-0.239325/-0.175988 m (all three folds). The descriptive interval excludes zero in the observed pooled direction.

NG-C−TNT=-0.115452 m;95% descriptive CI[-0.139082,-0.092697]. Complete participant/fold/metric estimates, including any degradation and null direction, appear in [comparison](stage17_external_comparison.csv) and [bootstrap](stage17_bootstrap_comparison.csv). No result is used to revise G-C or the original Stage15B decisions.

| Group | Model | Count | Top1FDE | Top1ADE | HitRate | minFDE6 |
| --- | --- | --- | --- | --- | --- | --- |
| Overall | R0 | 260151 | 2.372741 | 1.056338 | 0.451830 | 1.266202 |
| Overall | R2 | 260151 | 2.346574 | 1.044426 | 0.455017 | 1.266202 |
| Overall | NG-A | 260151 | 2.375782 | 1.064636 | 0.439276 | 1.266202 |
| Overall | NG-C | 260151 | 2.306543 | 1.020819 | 0.464492 | 1.266202 |
| Overall | G-A | 260151 | 2.361641 | 1.063987 | 0.440198 | 1.266202 |
| Overall | G-C | 260151 | 2.261161 | 0.999894 | 0.470369 | 1.266202 |
| Overall | Matched-NG-C | 260151 | 2.283684 | 1.009464 | 0.467475 | 1.266202 |
| Overall | Adapted TNT Scoring | 260151 | 2.421995 | 1.083085 | 0.489427 | 1.266202 |
| Vehicle | R0 | 191026 | 2.731294 | 1.198128 | 0.489389 | 1.390441 |
| Vehicle | R2 | 191026 | 2.698962 | 1.183550 | 0.491310 | 1.390441 |
| Vehicle | NG-A | 191026 | 2.741044 | 1.212796 | 0.466460 | 1.390441 |
| Vehicle | NG-C | 191026 | 2.648431 | 1.153161 | 0.497199 | 1.390441 |
| Vehicle | G-A | 191026 | 2.720192 | 1.209955 | 0.464895 | 1.390441 |
| Vehicle | G-C | 191026 | 2.588150 | 1.125391 | 0.502497 | 1.390441 |
| Vehicle | Matched-NG-C | 191026 | 2.616192 | 1.137288 | 0.501680 | 1.390441 |
| Vehicle | Adapted TNT Scoring | 191026 | 2.795167 | 1.233333 | 0.519756 | 1.390441 |
| Pedestrian | R0 | 66145 | 1.353283 | 0.652980 | 0.341356 | 0.914364 |
| Pedestrian | R2 | 66145 | 1.346393 | 0.649446 | 0.348054 | 0.914364 |
| Pedestrian | NG-A | 66145 | 1.339737 | 0.644469 | 0.357911 | 0.914364 |
| Pedestrian | NG-C | 66145 | 1.334883 | 0.644362 | 0.368312 | 0.914364 |
| Pedestrian | G-A | 66145 | 1.344339 | 0.650122 | 0.366059 | 0.914364 |
| Pedestrian | G-C | 66145 | 1.330484 | 0.642264 | 0.376128 | 0.914364 |
| Pedestrian | Matched-NG-C | 66145 | 1.338082 | 0.645543 | 0.367103 | 0.914364 |
| Pedestrian | Adapted TNT Scoring | 66145 | 1.360134 | 0.655141 | 0.400076 | 0.914364 |
| MovingVehicle | R0 | 41728 | 10.874690 | 4.687916 | 0.214101 | 5.343627 |
| MovingVehicle | R2 | 41728 | 10.724894 | 4.619718 | 0.221338 | 5.343627 |
| MovingVehicle | NG-A | 41728 | 10.863803 | 4.722059 | 0.248179 | 5.343627 |
| MovingVehicle | NG-C | 41728 | 10.470631 | 4.470499 | 0.222153 | 5.343627 |
| MovingVehicle | G-A | 41728 | 10.714765 | 4.681714 | 0.273989 | 5.343627 |
| MovingVehicle | G-C | 41728 | 10.210756 | 4.350547 | 0.234087 | 5.343627 |
| MovingVehicle | Matched-NG-C | 41728 | 10.324218 | 4.398571 | 0.227138 | 5.343627 |
| MovingVehicle | Adapted TNT Scoring | 41728 | 11.111135 | 4.823664 | 0.240462 | 5.343627 |

The shared candidate oracle minFDE6 remains1.266202 m; scoring cannot repair missing candidate geometry. HitRate means exact selected-mode agreement with the endpoint-FDE oracle, not a2m threshold. MovingVehicle is the original **t0 vehicle.moving** group, not a future-GT motion filter. All original hard and high-error samples remain included.

**A metric tradeoff must also be reported:** TNT Overall HitRate=0.489427, higher than G-C=0.470369 and NG-C=0.464492, despite its worse mean Top1FDE and Top1ADE. More frequent exact oracle-mode matches do not guarantee smaller average selection error. This aggregate frequency/error contrast is not evidence of a causal mechanism; no probability-calibration claim is made from it. Do not present only FDE as though TNT loses on every metric.

TNT’s soft target ranks **maximum full-path squared distance**, while comparison metrics include endpoint-FDE and the frozen project loss/selection definitions. Context is independently learned by the immutable predictor, not a newly fitted encoder. TNT’s context/whole-path information differs from G-C’s explicit predicted neighbor edges; these data do not identify a causal mechanism or prove that a full TNT system would behave similarly.

Pure TNT scores all types. [Bicycle-route sensitivity](06_evaluation/stage17_bicycle_route_sensitivity.csv) exposes the frozenR2 option separately: TNT+routed-R2 Overall Top1FDE=2.420710 m; G-C minus this sensitivity result=-0.159549 m. Bicycle primary TNT FDE=2.070084 m versus the G-C/R2 route=1.957885 m (n=2,980). This does not replace the primary result or conceal poor Bicycle scores. Vehicle/Pedestrian/MovingVehicle estimates are invariant to this route.

This comparison was added **after Stage15B results were known**. It is supplementary and outside the original confirmatory family3. The2000 paired bootstrap draws cluster by scene and independently resample210 scenes per fold; intervals are descriptive and conditional on fixed predictors/scorers. There is no new multiplicity-adjusted confirmatory claim. Historical method-development exposure, SchemeA training-in candidate use for head training, and lack of inner OOF remain unchanged limitations. Internal scene-isolated CV is not an untouched confirmation set or official nuScenes test result.

The baseline uses actual pinned unofficial TNT scoring code and paper/component loss; its context,horizon,candidate count and joint-training setting are adapted. The active full-repository BCE difference and missing redistribution license are disclosed. Source stays local; published code contains loader/provenance, not third-party class text. No new network innovation experiment follows this stage.
