# Stage16 ranking-head initialization stability

Completed all24 preregistered supplementary fits; reused the twelve original NG-A/NG-C/G-C/Matched-NG-C heads as initialization replicate0. No HiVT model was trained. The three Stage15B predictors, candidate coordinates, partitions, normalization and samefold R2 Bicycle route remained frozen.

This is **ranking-head initialization sensitivity conditional on fixed predictors and fixed data ordering**, not predictor cross-seed stability. The supplement was registered in commit `9e5ed32d8f5802ca49ce9e31b71854065fe6d0b7` before new Outer scores were calculated. All24 new heads were selected on InnerDev and globally frozen before unified Outer evaluation. The original Stage15B conclusions and primary bootstrap family remain unchanged.

## Registered seed plan

| Fold | Original init | New init1 | New init2 | Frozen predictor / sampler seed |
|---|---|---|---|---|
| 1 | 2022 | 3022 | 4022 | 2022 |
| 2 | 2122 | 3122 | 4122 | 2122 |
| 3 | 2222 | 3222 | 4222 | 2222 |

Only node/graph/adapter initialization seeds change. All models in a fold share the same per-epoch permutation and continuous1024-target carry, with exact original order SHA prefixes checked. AdamW0.001, weight_decay0.0001,FP32,microbatch128×8,max50 epochs,patience5, original Loss A/C and strict original relative Vehicle/Pedestrian InnerDev Sdev are unchanged. Shared constructors retain exact zero residual output at step0. Random initialization checks use zero optimizer steps; fitting uses InnerTrain378, selection InnerDev42, never Outer210/HeadDev70/official VAL. No seed was dropped, selected by Outer, or averaged into a new inference model.

Estimated before fitting: 1.541 head-fit GPU-reservation hours from original runs, 7.025 hours at50-epoch caps. Actual24 fit wall times sum to **2.073 hours**, including Dev evaluation/I/O/checkpoint work, not pure CUDA kernel time. Peak allocated training memory across new jobs: 148.2MiB. Existing RTX3080 environment was reused; no package or data installation occurred.

## Each initialization: Overall Top1FDE, meters

| Fold | Model | InitializationReplicate | InitializationSeed | Count | Top1FDE |
| --- | --- | --- | --- | --- | --- |
| 1 | NG-A | 0 | 2022 | 82673 | 2.222825 |
| 1 | NG-C | 0 | 2022 | 82673 | 2.193321 |
| 1 | G-C | 0 | 2022 | 82673 | 2.145866 |
| 1 | Matched-NG-C | 0 | 2022 | 82673 | 2.188395 |
| 1 | NG-A | 1 | 3022 | 82673 | 2.217458 |
| 1 | NG-C | 1 | 3022 | 82673 | 2.186080 |
| 1 | G-C | 1 | 3022 | 82673 | 2.138802 |
| 1 | Matched-NG-C | 1 | 3022 | 82673 | 2.194088 |
| 1 | NG-A | 2 | 4022 | 82673 | 2.214632 |
| 1 | NG-C | 2 | 4022 | 82673 | 2.203344 |
| 1 | G-C | 2 | 4022 | 82673 | 2.153149 |
| 1 | Matched-NG-C | 2 | 4022 | 82673 | 2.197440 |
| 2 | NG-A | 0 | 2122 | 83711 | 2.517525 |
| 2 | NG-C | 0 | 2122 | 83711 | 2.365157 |
| 2 | G-C | 0 | 2122 | 83711 | 2.335707 |
| 2 | Matched-NG-C | 0 | 2122 | 83711 | 2.351003 |
| 2 | NG-A | 1 | 3122 | 83711 | 2.491795 |
| 2 | NG-C | 1 | 3122 | 83711 | 2.353441 |
| 2 | G-C | 1 | 3122 | 83711 | 2.311875 |
| 2 | Matched-NG-C | 1 | 3122 | 83711 | 2.370522 |
| 2 | NG-A | 2 | 4122 | 83711 | 2.499976 |
| 2 | NG-C | 2 | 4122 | 83711 | 2.351110 |
| 2 | G-C | 2 | 4122 | 83711 | 2.316739 |
| 2 | Matched-NG-C | 2 | 4122 | 83711 | 2.362462 |
| 3 | NG-A | 0 | 2222 | 93767 | 2.384100 |
| 3 | NG-C | 0 | 2222 | 93767 | 2.354042 |
| 3 | G-C | 0 | 2222 | 93767 | 2.296264 |
| 3 | Matched-NG-C | 0 | 2222 | 93767 | 2.307598 |
| 3 | NG-A | 1 | 3222 | 93767 | 2.351429 |
| 3 | NG-C | 1 | 3222 | 93767 | 2.298902 |
| 3 | G-C | 1 | 3222 | 93767 | 2.285834 |
| 3 | Matched-NG-C | 1 | 3222 | 93767 | 2.279722 |
| 3 | NG-A | 2 | 4222 | 93767 | 2.422651 |
| 3 | NG-C | 2 | 4222 | 93767 | 2.326753 |
| 3 | G-C | 2 | 4222 | 93767 | 2.272109 |
| 3 | Matched-NG-C | 2 | 4222 | 93767 | 2.279104 |
| 0 | NG-A | 0 | fold-specific; see fold rows | 260151 | 2.375782 |
| 0 | NG-C | 0 | fold-specific; see fold rows | 260151 | 2.306543 |
| 0 | G-C | 0 | fold-specific; see fold rows | 260151 | 2.261161 |
| 0 | Matched-NG-C | 0 | fold-specific; see fold rows | 260151 | 2.283684 |
| 0 | NG-A | 1 | fold-specific; see fold rows | 260151 | 2.354021 |
| 0 | NG-C | 1 | fold-specific; see fold rows | 260151 | 2.280598 |
| 0 | G-C | 1 | fold-specific; see fold rows | 260151 | 2.247489 |
| 0 | Matched-NG-C | 1 | fold-specific; see fold rows | 260151 | 2.281726 |
| 0 | NG-A | 2 | fold-specific; see fold rows | 260151 | 2.381427 |
| 0 | NG-C | 2 | fold-specific; see fold rows | 260151 | 2.295373 |
| 0 | G-C | 2 | fold-specific; see fold rows | 260151 | 2.248666 |
| 0 | Matched-NG-C | 2 | fold-specific; see fold rows | 260151 | 2.279975 |

## Mean and sample standard deviation across three initializations

| Fold | Model | Initializations | CountPerInitialization | MeanTop1FDE | SDTop1FDE | MinTop1FDE | MaxTop1FDE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | NG-A | 3 | 82673 | 2.218305 | 0.004162 | 2.214632 | 2.222825 |
| 1 | NG-C | 3 | 82673 | 2.194248 | 0.008669 | 2.186080 | 2.203344 |
| 1 | G-C | 3 | 82673 | 2.145939 | 0.007174 | 2.138802 | 2.153149 |
| 1 | Matched-NG-C | 3 | 82673 | 2.193308 | 0.004573 | 2.188395 | 2.197440 |
| 2 | NG-A | 3 | 83711 | 2.503099 | 0.013146 | 2.491795 | 2.517525 |
| 2 | NG-C | 3 | 83711 | 2.356569 | 0.007528 | 2.351110 | 2.365157 |
| 2 | G-C | 3 | 83711 | 2.321440 | 0.012592 | 2.311875 | 2.335707 |
| 2 | Matched-NG-C | 3 | 83711 | 2.361329 | 0.009808 | 2.351003 | 2.370522 |
| 3 | NG-A | 3 | 93767 | 2.386060 | 0.035652 | 2.351429 | 2.422651 |
| 3 | NG-C | 3 | 93767 | 2.326566 | 0.027570 | 2.298902 | 2.354042 |
| 3 | G-C | 3 | 93767 | 2.284736 | 0.012115 | 2.272109 | 2.296264 |
| 3 | Matched-NG-C | 3 | 93767 | 2.288808 | 0.016276 | 2.279104 | 2.307598 |
| 0 | NG-A | 3 | 260151 | 2.370410 | 0.014471 | 2.354021 | 2.381427 |
| 0 | NG-C | 3 | 260151 | 2.294171 | 0.013014 | 2.280598 | 2.306543 |
| 0 | G-C | 3 | 260151 | 2.252439 | 0.007577 | 2.247489 | 2.261161 |
| 0 | Matched-NG-C | 3 | 260151 | 2.281795 | 0.001855 | 2.279975 | 2.283684 |

Fold0 denotes actor-window-weighted pooling of the three fixedfold predictions, grouping corresponding replicate indices as registered. Its SD is across those **three pooled configurations**, not nine independently trained predictors; it depends on this predetermined cross-fold pairing. `stage16_seed_all_fold_combinations.csv` additionally enumerates all27 choices of the three recorded initializations acrossfolds, without selecting any configuration. Fold means/SD and paired directions are the clearer evidence of initialization sensitivity. Min/max columns describe variation and are not a checkpoint/seed choice.

## Paired direction by fold and initialization

| Fold | InitializationReplicate | Comparison | DeltaTop1FDE | Direction |
| --- | --- | --- | --- | --- |
| 1 | 0 | G-C-NG-C | -0.047455 | Improved |
| 1 | 0 | G-C-Matched-NG-C | -0.042529 | Improved |
| 1 | 0 | NG-C-NG-A | -0.029504 | Improved |
| 1 | 1 | G-C-NG-C | -0.047278 | Improved |
| 1 | 1 | G-C-Matched-NG-C | -0.055286 | Improved |
| 1 | 1 | NG-C-NG-A | -0.031378 | Improved |
| 1 | 2 | G-C-NG-C | -0.050194 | Improved |
| 1 | 2 | G-C-Matched-NG-C | -0.044291 | Improved |
| 1 | 2 | NG-C-NG-A | -0.011288 | Improved |
| 2 | 0 | G-C-NG-C | -0.029450 | Improved |
| 2 | 0 | G-C-Matched-NG-C | -0.015297 | Improved |
| 2 | 0 | NG-C-NG-A | -0.152368 | Improved |
| 2 | 1 | G-C-NG-C | -0.041566 | Improved |
| 2 | 1 | G-C-Matched-NG-C | -0.058647 | Improved |
| 2 | 1 | NG-C-NG-A | -0.138354 | Improved |
| 2 | 2 | G-C-NG-C | -0.034371 | Improved |
| 2 | 2 | G-C-Matched-NG-C | -0.045723 | Improved |
| 2 | 2 | NG-C-NG-A | -0.148866 | Improved |
| 3 | 0 | G-C-NG-C | -0.057778 | Improved |
| 3 | 0 | G-C-Matched-NG-C | -0.011335 | Improved |
| 3 | 0 | NG-C-NG-A | -0.030058 | Improved |
| 3 | 1 | G-C-NG-C | -0.013068 | Improved |
| 3 | 1 | G-C-Matched-NG-C | 0.006112 | Worsened |
| 3 | 1 | NG-C-NG-A | -0.052526 | Improved |
| 3 | 2 | G-C-NG-C | -0.054644 | Improved |
| 3 | 2 | G-C-Matched-NG-C | -0.006994 | Improved |
| 3 | 2 | NG-C-NG-A | -0.095898 | Improved |
| 0 | 0 | G-C-NG-C | -0.045382 | Improved |
| 0 | 0 | G-C-Matched-NG-C | -0.022523 | Improved |
| 0 | 0 | NG-C-NG-A | -0.069239 | Improved |
| 0 | 1 | G-C-NG-C | -0.033110 | Improved |
| 0 | 1 | G-C-Matched-NG-C | -0.034238 | Improved |
| 0 | 1 | NG-C-NG-A | -0.073423 | Improved |
| 0 | 2 | G-C-NG-C | -0.046706 | Improved |
| 0 | 2 | G-C-Matched-NG-C | -0.031309 | Improved |
| 0 | 2 | NG-C-NG-A | -0.086054 | Improved |

## Direction counts: nine fold×initialization pairs per comparison

| Group | Comparison | FoldInitializationPairs | NegativePairs | PositivePairs | EqualPairs | UnweightedMeanPairDelta |
| --- | --- | --- | --- | --- | --- | --- |
| Overall | G-C-NG-C | 9 | 9 | 0 | 0 | -0.041756 |
| Overall | G-C-Matched-NG-C | 9 | 8 | 1 | 0 | -0.030443 |
| Overall | NG-C-NG-A | 9 | 9 | 0 | 0 | -0.076693 |
| Vehicle | G-C-NG-C | 9 | 9 | 0 | 0 | -0.054862 |
| Vehicle | G-C-Matched-NG-C | 9 | 8 | 1 | 0 | -0.038134 |
| Vehicle | NG-C-NG-A | 9 | 9 | 0 | 0 | -0.101327 |
| Pedestrian | G-C-NG-C | 9 | 7 | 2 | 0 | -0.005688 |
| Pedestrian | G-C-Matched-NG-C | 9 | 9 | 0 | 0 | -0.009661 |
| Pedestrian | NG-C-NG-A | 9 | 7 | 2 | 0 | -0.011324 |
| MovingVehicle | G-C-NG-C | 9 | 9 | 0 | 0 | -0.240039 |
| MovingVehicle | G-C-Matched-NG-C | 9 | 8 | 1 | 0 | -0.164015 |
| MovingVehicle | NG-C-NG-A | 9 | 9 | 0 | 0 | -0.408298 |

Nine pairs reuse three predictors/scenes; they are not nine independent scientific replications. These counts and means are descriptive, without new significance claims or a replacement for scene-cluster uncertainty. Positive rows must remain visible even when the pooled mean favors the method.

## Observed Overall directions

G-C-NG-C: 9 of 9 fold-initialization pairs favor the first model, 0 favor the second, and 0 tie. G-C-Matched-NG-C: 8 of 9 fold-initialization pairs favor the first model, 1 favor the second, and 0 tie. NG-C-NG-A: 9 of 9 fold-initialization pairs favor the first model, 0 favor the second, and 0 tie. This count describes the recorded conditional sensitivity sample; it does not establish population significance or predictor-seed robustness.

## Vehicle initialization spread

| Fold | Model | CountPerInitialization | MeanTop1FDE | SDTop1FDE |
| --- | --- | --- | --- | --- |
| 1 | NG-A | 61556 | 2.542338 | 0.005077 |
| 1 | NG-C | 61556 | 2.513788 | 0.010840 |
| 1 | G-C | 61556 | 2.452877 | 0.008654 |
| 1 | Matched-NG-C | 61556 | 2.511784 | 0.006909 |
| 2 | NG-A | 60451 | 2.905007 | 0.020996 |
| 2 | NG-C | 60451 | 2.711399 | 0.007082 |
| 2 | G-C | 60451 | 2.664114 | 0.015084 |
| 2 | Matched-NG-C | 60451 | 2.715500 | 0.014188 |
| 3 | NG-A | 69019 | 2.750331 | 0.048055 |
| 3 | NG-C | 69019 | 2.668508 | 0.037521 |
| 3 | G-C | 69019 | 2.612116 | 0.015910 |
| 3 | Matched-NG-C | 69019 | 2.616226 | 0.022825 |
| 0 | NG-A | 191026 | 2.732255 | 0.020779 |
| 0 | NG-C | 191026 | 2.632224 | 0.017316 |
| 0 | G-C | 191026 | 2.577258 | 0.009449 |
| 0 | Matched-NG-C | 191026 | 2.613986 | 0.002076 |

## Pedestrian initialization spread

| Fold | Model | CountPerInitialization | MeanTop1FDE | SDTop1FDE |
| --- | --- | --- | --- | --- |
| 1 | NG-A | 20250 | 1.264314 | 0.001780 |
| 1 | NG-C | 20250 | 1.252887 | 0.004936 |
| 1 | G-C | 20250 | 1.240815 | 0.002994 |
| 1 | Matched-NG-C | 20250 | 1.255137 | 0.002484 |
| 2 | NG-A | 22108 | 1.408312 | 0.008290 |
| 2 | NG-C | 22108 | 1.382875 | 0.009246 |
| 2 | G-C | 22108 | 1.379153 | 0.006470 |
| 2 | Matched-NG-C | 22108 | 1.389683 | 0.005062 |
| 3 | NG-A | 23787 | 1.352038 | 0.005703 |
| 3 | NG-C | 23787 | 1.354929 | 0.000247 |
| 3 | G-C | 23787 | 1.353660 | 0.001633 |
| 3 | Matched-NG-C | 23787 | 1.357790 | 0.002834 |
| 0 | NG-A | 66145 | 1.343990 | 0.004165 |
| 0 | NG-C | 66145 | 1.333030 | 0.001621 |
| 0 | G-C | 66145 | 1.327634 | 0.002564 |
| 0 | Matched-NG-C | 66145 | 1.337023 | 0.001431 |

## MovingVehicle initialization spread

| Fold | Model | CountPerInitialization | MeanTop1FDE | SDTop1FDE |
| --- | --- | --- | --- | --- |
| 1 | NG-A | 12793 | 10.602074 | 0.021116 |
| 1 | NG-C | 12793 | 10.458468 | 0.054661 |
| 1 | G-C | 12793 | 10.193693 | 0.036741 |
| 1 | Matched-NG-C | 12793 | 10.440673 | 0.035128 |
| 2 | NG-A | 14211 | 10.759873 | 0.125864 |
| 2 | NG-C | 14211 | 10.026051 | 0.019343 |
| 2 | G-C | 14211 | 9.821949 | 0.085107 |
| 2 | Matched-NG-C | 14211 | 10.059750 | 0.069192 |
| 3 | NG-A | 14724 | 11.057716 | 0.214212 |
| 3 | NG-C | 14724 | 10.710250 | 0.182596 |
| 3 | G-C | 14724 | 10.459010 | 0.064460 |
| 3 | Matched-NG-C | 14724 | 10.466273 | 0.103969 |
| 0 | NG-A | 41728 | 10.816591 | 0.100539 |
| 0 | NG-C | 41728 | 10.400046 | 0.076622 |
| 0 | G-C | 41728 | 10.160710 | 0.044496 |
| 0 | Matched-NG-C | 41728 | 10.319978 | 0.014412 |

## Limits and reproducibility

Candidate geometry and minFDE6 cannot change under these head-only fits. New outputs retain identity alignment and bitwise samefold R2 Bicycle scores/selections. The evaluation audit records those checks and all24 chosen checkpoint SHA256. Peak allocation excludes CUDA reserved memory and does not represent full-system predictor deployment memory.

Training source: `03_seed_stability/stage16_train_heads.py`; it ports the original optimizer loop into the Stage16 output scope while reusing original model classes and loss function bodies read-only. Registration includes code hashes, candidate manifest SHA, normalization SHA, predictor SHA and frozen R2 references. Every initialization has its full curve, batch-order digest, configuration, selected checkpoint SHA and measured resource record. New checkpoints/last-state recovery files remain local and never overwrite Stage15B checkpoints. Evaluation entry `stage16_evaluate_seeds.py` refuses to read Outer arrays before all24 heads freeze.

`06_source_data/stage16_seed_metrics.csv`, `stage16_seed_summary.csv`, `stage16_seed_contrasts.csv` and `stage16_seed_direction_summary.csv` provide every fold/group value. Main seven-model figures still use original frozen Stage15B results; supplementary S2 plots head-initialization spread. The smallest positive/negative changes should not be inflated into new scientific claims. Three initializations are a limited sensitivity sample. Predictor training, batch-order stochasticity, different candidate generators, external benchmarks and pristine held-out research generalization remain untested by this supplement.
