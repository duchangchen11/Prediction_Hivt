# Stage15B candidate provenance

| Fold | Role | Scenes | ActorWindows | Windows | ContextActors | PartialTargetsExcluded | EmptySupervisionWindowsExcluded | PredictorSHA |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | InnerTrain | 378 | 159151 | 9137 | 247584 | 73196 | 7 | 23bf3629c54eb86110965be97d38d02e7350a2b45a8568e2b831b0d9e6d632fd |
| 1 | InnerDev | 42 | 18327 | 1017 | 29610 | 9357 | 5 | 23bf3629c54eb86110965be97d38d02e7350a2b45a8568e2b831b0d9e6d632fd |
| 1 | OuterTest | 210 | 82673 | 5041 | 135083 | 43464 | 20 | 23bf3629c54eb86110965be97d38d02e7350a2b45a8568e2b831b0d9e6d632fd |
| 2 | InnerTrain | 378 | 157356 | 9103 | 249623 | 76707 | 32 | dadf139ef4d5ca4009a22de3dda42f3cd91f62a149d2c4114e922856100729c8 |
| 2 | InnerDev | 42 | 19084 | 1018 | 30572 | 9458 | 0 | dadf139ef4d5ca4009a22de3dda42f3cd91f62a149d2c4114e922856100729c8 |
| 2 | OuterTest | 210 | 83711 | 5074 | 132082 | 39852 | 0 | dadf139ef4d5ca4009a22de3dda42f3cd91f62a149d2c4114e922856100729c8 |
| 3 | InnerTrain | 378 | 148900 | 9104 | 240144 | 75388 | 20 | b1f7c7e8200255a998d65676a42b2b431ab5444df70ffd3822eb5349ba168ab4 |
| 3 | InnerDev | 42 | 17484 | 1011 | 27021 | 7928 | 0 | b1f7c7e8200255a998d65676a42b2b431ab5444df70ffd3822eb5349ba168ab4 |
| 3 | OuterTest | 210 | 93767 | 5080 | 145112 | 42701 | 12 | b1f7c7e8200255a998d65676a42b2b431ab5444df70ffd3822eb5349ba168ab4 |

Each fold has one role-separated streamed cache shared by all six heads: scene/sample/instance/node identity, fold/seed/checkpoint SHA, original logits/probabilities, all6 trajectories, original timestamps, ego frame, history/candidate masks and separate complete supervised masks/labels. No sixfold cache duplication. All current actors remain graph context, including partial-future actors; only predetermined full-horizon eligible targets are scored. No difficult case exclusions. Empty-supervision windows contain no qualifying target and are transparently counted.

Feature computation uses only observable history/type/candidates/logits/probabilities. GT is separate label data. Full graph/R2 normalization is fitted only on corresponding InnerTrain; `stage15b_foldN_normalization.json` includes source lists, counts and SHA. InnerTrain candidates are predictor training-in under Scheme A; this is disclosed rather than presented as inner OOF. Outer labels/errors are cached without aggregate evaluation or fitting access; a read guard forbids the Outer cache during head optimization.

Small per-fold provenance manifests retain full local cache/context SHA/size/identity lists. Large `.pt`/`.npy`/identity CSV caches stay local.
