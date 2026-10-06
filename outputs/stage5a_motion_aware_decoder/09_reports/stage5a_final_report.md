# Stage5A: Motion-Aware Heterogeneous Residual Decoder

【Motivation】

Stage3B TypeEmbedding remains SUPPORTED and is the sole formal backbone baseline. Stage4A always-on relation bias and Stage4F necessity-gated relation bias remain technically PASS but scientifically NOT SUPPORTED. Stage4F-D GateCompensation remains MIXED: pooled attenuation coexists with a localized Layer2/Head3 exception. These results motivate stopping attention-logit relation-bias changes and testing actor-type/history-dependent trajectory decoding.

【Current Decoder】

The actual frozen decoder is an MLPDecoder. pi = pi(concat(local, global)); h = aggr_embed(concat(global, local)); loc = loc(h); scale = ELU(scale(h)) + 1 + min_scale. There is no mode query, decoder cross-attention or Transformer decoder. The pi branch retains its original structure and embedding inputs.

【Method】

The original LocalEncoder, TypeEmbedding fusion, GlobalInteractor and multihead_proj are retained. c is six-dimensional: type one-hot [V,P,B] plus log1p(recent displacement), log1p(history net displacement) and log1p(history path length). Padding gaps are skipped and successive valid observations are connected in time; fewer than two valid observations produce zero motion features.

r = softmax(Linear(16,2)(ReLU(Linear(6,16)(c)))). Each actor shares its two routing weights across all six modes. Two unnamed experts use Linear(64,16) → ReLU → Linear(16,64). Δh[k,i] = Σₑ r[i,e] Aₑ(h[k,i]); h′ = h + Δh; the original shared loc and scale heads receive h′. Residual scale is one. The experts do not receive future labels or neighbor statistics, and they do not directly modify pi.

| Model | Parameters |
| --- | --- |
| Stage3B | 646001 |
| Stage5A | 650403 |
| Additional | 4402 |

【No Future Leakage】

The condition function accepts only [N,5,2] historical positions, [N,5] historical padding and actor types. Perturbing future trajectory, future masks/padding, GT endpoint, future labels, future times, ego future and target mask changes neither condition nor routing (maximum difference zero). The real-batch output comparison also has zero difference. Future-displacement bins are used only after inference for offline analysis.

【Initialization】

Seed 2022 reproduces canonical Stage3B step0 from scratch. Every shared parameter and buffer is bitwise equal. Both expert final Linear layers have zero weight/bias; the router final layer is also zero, so r=[0.5,0.5] and Δh=0. No trained Stage3B checkpoint is loaded for optimization.

Neutral real-TRAIN-batch differences: raw_prediction=0.0; mode_logits=0.0; mode_prob=0.0. All required shape, finite, broadcast and padding tests pass. All six new gradient modules are positive within ten updates. Tiny fixed-scale training uses the frozen six TRAIN windows, covers 11 vehicle / 12 pedestrian / 6 bicycle targets and moving/low-history-motion V/P; all three regression diagnostics decrease. Mean absolute residual=0.212388; maximum router deviation from half=0.302796. Tiny is discarded before formal optimization.

【Training】

One formal from-scratch model, seed2022; official TRAIN700/VAL150, test unused; Th=5, Tf=12, K=6; batch16, embed64, heads8, global layers3, dropout0.1, local radius50m, AdamW weight_decay1e-4. Fixed-scale warm-up executes exactly5000 updates at LR0.001. NLL restores this experiment's own warm-up best model/optimizer/RNG, then uses the unchanged original learnable Laplace NLL at LR0.0001. Python, NumPy, torch and CUDA RNG states are saved. Complete VAL150 occurs every500 updates, strict overall full-horizon minFDE6 selection and NLL patience5; global hard limit21000. There is no extra loss, oversampling, class weighting, expert balancing or hyperparameter search.

Warm-up best source step=3500; NLL updates=6500; best global step=9000; final executed global step=11500; stop reason=patience_5. Training code commit=1b15908ab6daa8a800a45dcf243ab406e4869542. Best checkpoint SHA256=88fe3feb59b7e830ec1e40aa917484386e0d2799d0f6116f410adda2d97fdce7. Fresh reload reconciles all metrics to the selected checkpoint within preregistered tolerances. NaN=0, Inf=0.

【Main Results】

All main values use full-horizon actor-window means. minADE6 is ADE of the best-FDE mode, minFDE6 the smallest endpoint error, MR6 endpoint error>2m, Top1 argmax predicted probability, NLL original best-summed-L2 mode and valid-time coordinate density mean.

| Group | Model | Count | minADE6 | minFDE6 | MR6 | Top1ADE6 | Top1FDE6 | NLL |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| overall | Stage3B | 54990 | 0.666406 | 1.343260 | 0.163102 | 1.196715 | 2.687984 | -0.705393 |
| overall | Stage5A | 54990 | 0.665786 | 1.337996 | 0.163757 | 1.204789 | 2.680611 | -0.668057 |
| vehicle | Stage3B | 42332 | 0.708347 | 1.431914 | 0.165761 | 1.343296 | 3.050323 | -0.876745 |
| vehicle | Stage5A | 42332 | 0.707392 | 1.424436 | 0.166045 | 1.361097 | 3.053786 | -0.829353 |
| pedestrian | Stage3B | 12002 | 0.525708 | 1.043875 | 0.154891 | 0.699869 | 1.457706 | -0.099274 |
| pedestrian | Stage5A | 12002 | 0.526518 | 1.047327 | 0.157640 | 0.680031 | 1.424744 | -0.098604 |
| bicycle | Stage3B | 656 | 0.534082 | 1.099908 | 0.141768 | 0.827986 | 1.814865 | -0.737320 |
| bicycle | Stage5A | 656 | 0.528919 | 1.077948 | 0.128049 | 0.718954 | 1.576419 | -0.678123 |

【Vehicle Motion】

| Group | Count | Stage3B_minFDE6 | Stage5A_minFDE6 | Delta_minFDE6 |
| --- | --- | --- | --- | --- |
| vehicle.moving | 10461 | 4.791532 | 4.765633 | -0.025899 |
| vehicle.stopped | 5599 | 0.868888 | 0.863684 | -0.005204 |
| vehicle.parked | 25198 | 0.146415 | 0.146192 | -0.000223 |
| unknown | 1074 | 1.803797 | 1.793712 | -0.010085 |
| Vehicle >5m | 9744 | 5.579887 | 5.561047 | -0.018840 |

【Pedestrian Motion Bins】

Frozen endpoint bins are [0,1), [1,2), [2,5), [5,10), [10,20) meters. <5m and >5m use strict inequalities and remain offline only.

| Group | Count | Stage3B_minFDE6 | Stage5A_minFDE6 | Delta_minFDE6 |
| --- | --- | --- | --- | --- |
| Pedestrian <5m | 4421 | 0.535423 | 0.560457 | 0.025034 |
| Pedestrian >5m | 7581 | 1.340389 | 1.331254 | -0.009135 |
| Pedestrian 0-1m | 3192 | 0.167310 | 0.172325 | 0.005015 |
| Pedestrian 1-2m | 313 | 0.975018 | 1.059186 | 0.084169 |
| Pedestrian 2-5m | 916 | 1.667979 | 1.742568 | 0.074589 |
| Pedestrian 5-10m | 7321 | 1.312738 | 1.301926 | -0.010812 |
| Pedestrian 10-20m | 260 | 2.118974 | 2.157070 | 0.038097 |

【Bootstrap】

Stage3B and Stage5A match scene/sample/instance/node/type/motion/future-mask/GT-SHA exactly: 54,990 full +30,037 partial =85,027 actor-windows across150 VAL scenes and3603 supervised windows. The primary endpoint is overall full-horizon minFDE6. Paired scene-cluster percentile bootstrap uses1000 replicates and seed2022, pooling all actor-window deltas in each set of resampled whole scenes. Negative E−B favors Stage5A. These intervals condition on the two selected checkpoints and do not measure variation across training seeds or correct for checkpoint selection on the same VAL split. Secondary subgroup intervals are descriptive; no multiplicity correction is applied. Overlapping windows and nested motion groups are not independent observations. Bicycle has limited unique instances/scenes and receives cautious interpretation.

| Group | Metric | Delta | CI lower | CI upper | Count | Scenes |
| --- | --- | --- | --- | --- | --- | --- |
| overall | minADE6 | -0.000620 | -0.004485 | 0.003131 | 54990 | 150 |
| overall | minFDE6 | -0.005265 | -0.012193 | 0.001745 | 54990 | 150 |
| overall | Top1FDE6 | -0.007373 | -0.079163 | 0.068343 | 54990 | 150 |
| vehicle | minADE6 | -0.000954 | -0.005916 | 0.003803 | 42332 | 150 |
| vehicle | minFDE6 | -0.007477 | -0.016383 | 0.001423 | 42332 | 150 |
| pedestrian | minADE6 | 0.000810 | -0.001811 | 0.003357 | 12002 | 121 |
| pedestrian | minFDE6 | 0.003452 | -0.002741 | 0.009500 | 12002 | 121 |
| pedestrian | Top1FDE6 | -0.032962 | -0.049876 | -0.015046 | 12002 | 121 |
| bicycle | minFDE6 | -0.021960 | -0.081926 | 0.045106 | 656 | 47 |
| vehicle.moving | minFDE6 | -0.025899 | -0.057338 | 0.009777 | 10461 | 137 |
| vehicle.stopped | minFDE6 | -0.005204 | -0.017646 | 0.009713 | 5599 | 103 |
| vehicle.parked | minFDE6 | -0.000223 | -0.001734 | 0.000967 | 25198 | 131 |
| unknown | minFDE6 | -0.010085 | -0.043866 | 0.019659 | 1074 | 61 |
| Vehicle >5m | minFDE6 | -0.018840 | -0.053432 | 0.017683 | 9744 | 137 |
| Pedestrian <5m | minFDE6 | 0.025034 | 0.014010 | 0.037737 | 4421 | 100 |
| Pedestrian >5m | minFDE6 | -0.009135 | -0.014814 | -0.003316 | 7581 | 110 |
| Pedestrian 0-1m | minFDE6 | 0.005015 | -0.000893 | 0.011238 | 3192 | 86 |
| Pedestrian 1-2m | minFDE6 | 0.084169 | 0.033216 | 0.144043 | 313 | 49 |
| Pedestrian 2-5m | minFDE6 | 0.074589 | 0.045294 | 0.110397 | 916 | 65 |
| Pedestrian 5-10m | minFDE6 | -0.010812 | -0.015369 | -0.006682 | 7321 | 110 |
| Pedestrian 10-20m | minFDE6 | 0.038097 | -0.069147 | 0.171648 | 260 | 29 |

【Router Behavior】

Routing is actor-level and shared across six modes. Entropy uses natural logarithms. Dominance is argmax; exact ties are separately reported. Group tables below use full-horizon targets; the collapse check additionally covers all current-valid context actors.

| Group | Count | r1_mean | r2_mean | r1_median | r2_median | entropy_mean | expert1_dominant_rate | expert2_dominant_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| overall | 54990 | 0.367687 | 0.632313 | 0.279184 | 0.720816 | 0.576186 | 0.161393 | 0.838607 |
| vehicle | 42332 | 0.380786 | 0.619214 | 0.270722 | 0.729278 | 0.564430 | 0.206085 | 0.793915 |
| pedestrian | 12002 | 0.322873 | 0.677127 | 0.344407 | 0.655593 | 0.617508 | 0.003583 | 0.996417 |
| bicycle | 656 | 0.342332 | 0.657668 | 0.263603 | 0.736397 | 0.578744 | 0.164634 | 0.835366 |
| vehicle.moving | 10461 | 0.700778 | 0.299222 | 0.780153 | 0.219847 | 0.516529 | 0.798490 | 0.201510 |
| vehicle.stopped | 5599 | 0.276745 | 0.723255 | 0.256126 | 0.743874 | 0.582060 | 0.010716 | 0.989284 |
| vehicle.parked | 25198 | 0.271274 | 0.728726 | 0.256711 | 0.743288 | 0.580204 | 0.003413 | 0.996587 |
| unknown | 1074 | 0.375719 | 0.624281 | 0.271755 | 0.728245 | 0.569002 | 0.209497 | 0.790503 |
| Vehicle >5m | 9744 | 0.729322 | 0.270678 | 0.798206 | 0.201794 | 0.502945 | 0.851704 | 0.148296 |
| Pedestrian <5m | 4421 | 0.247273 | 0.752727 | 0.229386 | 0.770614 | 0.554635 | 0.000226 | 0.999774 |
| Pedestrian >5m | 7581 | 0.366961 | 0.633039 | 0.369142 | 0.630858 | 0.654175 | 0.005540 | 0.994460 |
| Pedestrian 0-1m | 3192 | 0.230367 | 0.769633 | 0.223255 | 0.776745 | 0.538176 | 0.000000 | 1.000000 |
| Pedestrian 1-2m | 313 | 0.270555 | 0.729445 | 0.271244 | 0.728756 | 0.578513 | 0.003195 | 0.996805 |
| Pedestrian 2-5m | 916 | 0.298228 | 0.701772 | 0.305720 | 0.694280 | 0.603829 | 0.000000 | 1.000000 |
| Pedestrian 5-10m | 7321 | 0.364741 | 0.635259 | 0.367317 | 0.632683 | 0.653340 | 0.000683 | 0.999317 |
| Pedestrian 10-20m | 260 | 0.429488 | 0.570512 | 0.420788 | 0.579212 | 0.677669 | 0.142308 | 0.857692 |

Router collapse=NO. Preregistered criterion: the same expert receives probability>0.9 in>95% of all current-valid actor-windows. Context population=91092. The learned experts retain neutral numerical names; a routing association alone does not prove a static/dynamic specialization.

| Group | Feature | Spearman_rho | Count |
| --- | --- | --- | --- |
| overall | log1p_recent_displacement | 0.927962 | 54990 |
| overall | log1p_net_displacement | 0.935145 | 54990 |
| overall | log1p_path_length | 0.947196 | 54990 |
| vehicle | log1p_recent_displacement | 0.963822 | 42332 |
| vehicle | log1p_net_displacement | 0.978804 | 42332 |
| vehicle | log1p_path_length | 0.995592 | 42332 |
| pedestrian | log1p_recent_displacement | 0.991221 | 12002 |
| pedestrian | log1p_net_displacement | 0.943003 | 12002 |
| pedestrian | log1p_path_length | 0.943927 | 12002 |

【Expert Diversity】

Both expert outputs are evaluated on the same hidden h. Norms and cosine are computed per mode then averaged over all six modes. Relative difference is RMS(A1−A2)/max(RMS(A1),RMS(A2),1e-12). Numerical cosine is zero when a vector is zero; this convention and the zero-output rate are reported. Before training, near-identity was defined as relative difference<1e-3 and cosine>0.999; functional collapse means this holds for>95% of current-valid actor-windows, or both outputs are numerically zero for>95%. This is a descriptive implementation criterion, not a training objective.

| Group | Count | expert1_norm_mean | expert2_norm_mean | expert_cosine_mean | expert_relative_difference_mean | residual_norm_mean | near_identical_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| overall | 54990 | 4.642152 | 24.570323 | -0.417899 | 1.095736 | 14.209821 | 0.000000 |
| vehicle | 42332 | 4.459539 | 23.513870 | -0.415825 | 1.095999 | 13.070938 | 0.000000 |
| pedestrian | 12002 | 5.280543 | 28.371746 | -0.426224 | 1.094466 | 18.245872 | 0.000000 |
| bicycle | 656 | 4.746462 | 23.193921 | -0.399444 | 1.101965 | 13.859971 | 0.000000 |
| vehicle.moving | 10461 | 3.816802 | 30.849864 | -0.486283 | 1.065846 | 7.934489 | 0.000000 |
| vehicle.stopped | 5599 | 4.665682 | 21.115440 | -0.393279 | 1.105847 | 14.744547 | 0.000000 |
| vehicle.parked | 25198 | 4.679290 | 21.000659 | -0.391628 | 1.106330 | 14.822893 | 0.000000 |
| unknown | 1074 | 4.489514 | 23.527714 | -0.414791 | 1.095973 | 13.272132 | 0.000000 |
| Vehicle >5m | 9744 | 3.749568 | 31.381828 | -0.492243 | 1.063764 | 7.336186 | 0.000000 |
| Pedestrian <5m | 4421 | 5.177302 | 23.441586 | -0.405096 | 1.109253 | 17.043789 | 0.000000 |
| Pedestrian >5m | 7581 | 5.340750 | 31.246859 | -0.438545 | 1.085843 | 18.946890 | 0.000000 |
| Pedestrian 0-1m | 3192 | 5.125973 | 22.205783 | -0.397323 | 1.112645 | 16.622053 | 0.000000 |
| Pedestrian 1-2m | 313 | 5.231063 | 25.103037 | -0.418408 | 1.104280 | 17.645696 | 0.000000 |
| Pedestrian 2-5m | 916 | 5.337797 | 27.180284 | -0.427633 | 1.099131 | 18.307747 | 0.000000 |
| Pedestrian 5-10m | 7321 | 5.345271 | 31.193423 | -0.438539 | 1.086068 | 18.988194 | 0.000000 |
| Pedestrian 10-20m | 260 | 5.213437 | 32.751495 | -0.438702 | 1.079507 | 17.783842 | 0.000000 |

Expert functional collapse=NO.

【Efficiency】

500 paired measurements alternate B/E and E/B on identical preloaded GPU batches;20 warm-up batches. CUDA events cover synchronized model forward, excluding I/O, H2D and the external clone but including both models' internal clone. An independent one-GPU-model pass over all226 VAL batches measures peak CUDA allocation. These are batch-forward latency measurements on the available GPU, not a deployment throughput claim.

| Model | parameters | mean_inference_ms | median_inference_ms | std_inference_ms | peak_CUDA_memory_MiB |
| --- | --- | --- | --- | --- | --- |
| Stage3B | 646001 | 58.717283 | 54.049921 | 27.790081 | 1735.671387 |
| Stage5A | 650403 | 58.389866 | 55.528578 | 22.093861 | 1735.690918 |

Mean inference overhead=-0.557616%; median overhead=2.735724%. GPU=NVIDIA GeForce RTX 3080; torch=2.5.1+cu124; CUDA=12.4.

Interpret small latency differences within the observed timing dispersion; this single-device measurement includes runtime noise.

【Qualitative Cases】

Four deterministic, purposive matched cases show moving vehicle, parked/stopped vehicle, moving pedestrian and degradation. The same actor/GT/map/frame/axis limits are used across Stage3B and Stage5A. Original twelve future observations are displayed without interpolation or smoothing. History=gray circles, GT=black squares, Best-FDE=blue circles, Top1=orange dashed triangles. Best-FDE uses GT and is distinct from predicted mode ranking. Replayed predictions use the original16-window VAL batch, match best/top1 mode IDs and reproduce CSV metrics within1e-4m. Examples do not estimate population effects.

| name | case_kind | agent_type | delta_FDE_m |
| --- | --- | --- | --- |
| stage5a_case_moving_vehicle | moving vehicle improvement | vehicle | -7.094152 |
| stage5a_case_parked_stopped_vehicle | parked/stopped vehicle | vehicle | -7.038580 |
| stage5a_case_moving_pedestrian | moving pedestrian | pedestrian | -1.797105 |
| stage5a_case_degradation | degradation case | vehicle | 11.011261 |

【Scientific Decision】

Motion-Aware Heterogeneous Decoder = **NOT SUPPORTED**. Stage5A = **PASS**. Ready Reliability = **NO**.

Important motion groups with reliable FDE gains: Pedestrian 5-10m. Reliable major-class harm guard=False. The guard conservatively treats any Vehicle/Pedestrian FDE95% CI lower>0 as harm; no post-hoc effect-size threshold is introduced. SUPPORTED requires overall CI upper<0 plus at least one reliable important motion gain and no class harm/collapse. PARTIAL requires overall CI spanning0 plus at least two such motion gains and no class harm/collapse. Remaining outcomes are NOT SUPPORTED.

Primary overall minFDE6 delta=-0.005265 m, 95% CI=[-0.012193473418844558, 0.0017449905652007114]; reliable important motion gains=1. Pedestrian <5m delta=0.025034 m, 95% CI=[0.014009639016743963, 0.03773673898898576]. Localized improvement and degradation must be read together with the primary interval, rather than inferring broad motion-adaptation benefit from routing separation alone.

All earlier scientific conclusions remain frozen. Only one Stage5A formal model was trained. Type-only/motion-only routing ablations, additional seeds, attention changes and Reliability are not executed. STOP: any next experiment requires the next explicit user instruction.

【Artifacts】

All new files reside in outputs/stage5a_motion_aware_decoder. Existing scene shards and checkpoints are referenced read-only; no dataset is copied or reprocessed. Code, configuration, small audit/report/table files, the complete training curve and figures are versioned. Raw actor/router records, per-validation scene-detailed JSON, logs and checkpoints remain local; counts, schemas and SHA256 are recorded as applicable. The artifact inventory excludes its own hash and transient Git-upload payloads. Old Stage2C redraw files are preserved and not submitted.
