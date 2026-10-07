【Motivation】

Stage7A remains NOT_SUPPORTED / PaperUsableSemantic=NO / ReadyStage7B=NO. Frozen Stage7C classified RELEVANCE_MISALLOCATION and recommended score-only actor-conditioned semantic attention. This is a candidate explanation with localized descriptive evidence, not established causality. Stage7D tests forecasting performance and attention allocation independently; Stage3B is the sole primary baseline.

【Architecture】

Stage7D has one change: a 14→32→8 ReLU MLP adds a per-head bias to the canonical lane→actor score immediately before softmax. Its inputs are nine frozen Stage7A semantic features, the original target-actor-rotated relative XY divided by 50, and the existing actor type one-hot. Its final linear layer starts at exactly zero. Total parameters are 646,745: 744 more than Stage3B (+0.115170%).

Canonical Stage3B construction finishes first. All original AL tensors are copied into the isolated AL implementation, AA/temporal modules are retained, and post-constructor CPU RNG is restored. No trained checkpoint initializes Stage7D. Geometry embeddings, Q, K, V, update/gate, FFN, global interactor and decoder keep the original equations. Semantics only enter the score. The neutral audit also perturbs semantic inputs with nonzero bias weights and requires exact K/V equality.

The Python attribute is named `semantic_score_mlp`: the canonical optimizer partitions parameters by their full names and treats any name containing `bias` as no-decay. Naming the parent module with that substring would make its linear weights appear in both partitions. The attribute name preserves the unchanged optimizer: linear weights receive 1e-4 decay and linear biases receive zero decay. This changes no architecture or hyperparameter.

The extra score contains an actor/head-specific constant component, but a constant across an actor's incoming lanes cancels in softmax. Large absolute bias therefore does not by itself demonstrate a useful semantic effect. Mechanism conclusions use paired GT relevance and turn-consistency measurements after checkpoint selection, independently of forecasting support.

Initialization/identity audits use controlled deterministic CUDA kernels and `CUBLAS_WORKSPACE_CONFIG=:4096:8` to distinguish instrumentation differences from scatter nondeterminism. Tiny and formal training retain the original non-deterministic kernel policy. Tiny weights are discarded; the formal run starts from the fresh canonical seed2022 constructor.


Frozen nine semantic fields and original categorical zeros are reused. Th=5, Tf=12, K=6, width64, eight heads, temporal4/global3, dropout0.1, radius50m. No semantic residual, decoder modification, reliability or dynamic signal input. Geometry-only K/V describes direct forward dependence: learned shared weights can differ after separately training Stage7D.

【Neutral Initialization】

PASS. Shared parameters and buffers: exact equality. raw_prediction/mode_logits/mode_prob maximum differences: {'raw_prediction': 0.0, 'mode_logits': 0.0, 'mode_prob': 0.0}. Original attention score difference: 0.0. Bias starts at exactly0. Six real TRAIN graphs; deterministic CUDA only for controlled equality. Nonzero-bias semantic perturbation K/V differences: {'lin_k': 0.0, 'lin_v': 0.0}.

【Data Integrity】

PASS: 100 random TRAIN and100 random VAL windows, seed2022; every original field bitwise equal, only lane_semantic added. Audited Stage7A map metadata reused, no map/trajectory/edge regeneration. Official TRAIN700/VAL150; test unused. Fresh evaluation:150 scenes,3603 supervised windows,54990 full-horizon +30037 partial =85027 targets; prediction NaN=0/Inf=0. Full-horizon semantic groups and signed GT turns are identity/GT-hash checked frozen offline sidecars, never branch inputs.

【Training】

Fresh seed2022 canonical initialization; no trained baseline weights. Fixed six TRAIN graphs cover V/P/B,moving,left/right connectors,controls,crosswalks; tiny200 updates PASS, first20 mean loss=3.468379, last20=2.392088, bias and attention become finite nonzero. Both MLP layers have finite nonzero gradients within10 updates, shared modules connected.

Formal Protocol1: AdamW, decay1e-4,natural taxonomy distribution,batch16. Fixed warmup5000 at LR.001, own best source step=3500. Model/optimizer/Python/NumPy/Torch/CUDA RNG restored at transition; LR alone becomes.0001, canonical NLL sampler resets. NLL updates=15000, final global=20000, best global=17500, stop=patience_5. Every500 fullVAL, strict overall full-horizon minFDE selection,patience5,hard21000. No subgroup/Top1 checkpoint selection or tuning. Training source commit=ad43f821606de97d1878e95f8849da31c42c3c5a.

Every validation logs exact absolute-bias statistics over all retained VAL edges/heads, including context actors; maximum observed |bias|=10.873121; all finite. Final mean/median/p95/max |bias|=1.195475/0.716368/4.537684/6.386813. Large absolute score biases alone cannot establish mechanism because actor/head constants cancel in softmax.

【Main Forecasting Results】

| Group | Model | Horizon | Count | minADE6 | minFDE6 | MR6 | Top1ADE6 | Top1FDE6 | NLL | independent_minADE6 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Overall | Stage3B | full_horizon | 54990 | 0.666406 | 1.343260 | 0.163102 | 1.196715 | 2.687984 | -0.705393 | 0.629952 |
| Overall | Stage7D | full_horizon | 54990 | 0.663083 | 1.333902 | 0.156610 | 1.215770 | 2.712028 | -0.665086 | 0.624144 |
| Vehicle | Stage3B | full_horizon | 42332 | 0.708347 | 1.431914 | 0.165761 | 1.343296 | 3.050323 | -0.876745 | 0.665897 |
| Vehicle | Stage7D | full_horizon | 42332 | 0.715331 | 1.447114 | 0.164533 | 1.371148 | 3.086795 | -0.827510 | 0.670122 |
| Pedestrian | Stage3B | full_horizon | 12002 | 0.525708 | 1.043875 | 0.154891 | 0.699869 | 1.457706 | -0.099274 | 0.509812 |
| Pedestrian | Stage7D | full_horizon | 12002 | 0.486217 | 0.948811 | 0.129395 | 0.690859 | 1.445362 | -0.089527 | 0.468692 |
| Bicycle | Stage3B | full_horizon | 656 | 0.534082 | 1.099908 | 0.141768 | 0.827986 | 1.814865 | -0.737320 | 0.508489 |
| Bicycle | Stage7D | full_horizon | 656 | 0.527410 | 1.073824 | 0.143293 | 0.792746 | 1.702708 | -0.714105 | 0.501250 |
| Vehicle >5m | Stage3B | full_horizon | 9744 | 2.647145 | 5.579887 | 0.676416 | 5.177695 | 12.105361 | 1.718378 | 2.496167 |
| Vehicle >5m | Stage7D | full_horizon | 9744 | 2.659471 | 5.625346 | 0.670259 | 5.292910 | 12.272857 | 1.699790 | 2.494172 |
| Pedestrian <5m | Stage3B | full_horizon | 4421 | 0.296053 | 0.535423 | 0.067858 | 0.461395 | 0.967858 | -0.707604 | 0.276884 |
| Pedestrian <5m | Stage7D | full_horizon | 4421 | 0.301700 | 0.551557 | 0.071477 | 0.468916 | 0.989135 | -0.696343 | 0.283543 |
| Pedestrian >5m | Stage3B | full_horizon | 7581 | 0.659635 | 1.340389 | 0.205646 | 0.838939 | 1.743370 | 0.255486 | 0.645647 |
| Pedestrian >5m | Stage7D | full_horizon | 7581 | 0.593821 | 1.180477 | 0.163171 | 0.820289 | 1.711420 | 0.264349 | 0.576665 |

Full and partial metrics are in the source tables. minADE uses the best-FDE candidate, MR is endpoint>2m, Top1 is argmax mode probability; NLL retains the original best summed-L2 mode/valid-coordinate definition. Stage3B frozen reference Overall minFDE=1.343260 /Top1FDE=2.687984; Stage7A=1.350546/2.749482. Historical scientific decisions remain unchanged.

【Moving Vehicle】

| Group | Model | Horizon | Count | minADE6 | minFDE6 | MR6 | Top1ADE6 | Top1FDE6 | NLL | independent_minADE6 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vehicle.moving | Stage3B | full_horizon | 10461 | 2.328189 | 4.791532 | 0.597075 | 4.656183 | 10.734859 | 1.336461 | 2.190683 |
| vehicle.moving | Stage7D | full_horizon | 10461 | 2.343754 | 4.838301 | 0.592295 | 4.716816 | 10.799879 | 1.289128 | 2.193951 |
| vehicle.stopped | Stage3B | full_horizon | 5599 | 0.350035 | 0.868888 | 0.087158 | 0.508323 | 1.313399 | -1.227037 | 0.332388 |
| vehicle.stopped | Stage7D | full_horizon | 5599 | 0.357574 | 0.876474 | 0.085908 | 0.520810 | 1.332714 | -1.182316 | 0.340262 |
| vehicle.parked | Stage3B | full_horizon | 25198 | 0.109823 | 0.146415 | 0.004088 | 0.155198 | 0.248490 | -1.716679 | 0.101524 |
| vehicle.parked | Stage7D | full_horizon | 25198 | 0.113252 | 0.150683 | 0.003969 | 0.170522 | 0.271523 | -1.626285 | 0.105652 |
| unknown | Stage3B | full_horizon | 1074 | 0.841131 | 1.803797 | 0.167598 | 1.302886 | 2.992323 | -0.901342 | 0.794011 |
| unknown | Stage7D | full_horizon | 1074 | 0.845052 | 1.807718 | 0.175047 | 1.385482 | 3.155450 | -0.853626 | 0.790812 |

Vehicle>5m uses the frozen GT endpoint displacement threshold and is offline evaluation only. Moving is the original t0 taxonomy annotation, not a new speed threshold.

【Turning Vehicle】

| Group | Model | Horizon | Count | minADE6 | minFDE6 | MR6 | Top1ADE6 | Top1FDE6 | NLL | independent_minADE6 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IntersectionVehicle20 | Stage3B | full_horizon | 32085 | 0.858295 | 1.760552 | 0.204021 | 1.639053 | 3.740200 | -0.606982 | 0.808250 |
| IntersectionVehicle20 | Stage7D | full_horizon | 32085 | 0.865994 | 1.778350 | 0.202587 | 1.666668 | 3.773869 | -0.574914 | 0.812731 |
| NearTurnConnector20 | Stage3B | full_horizon | 24949 | 0.939652 | 1.976263 | 0.218526 | 1.752410 | 4.010570 | -0.573041 | 0.890091 |
| NearTurnConnector20 | Stage7D | full_horizon | 24949 | 0.949533 | 1.986333 | 0.217844 | 1.770473 | 4.028333 | -0.535299 | 0.893430 |
| NearTrafficControl20 | Stage3B | full_horizon | 31778 | 0.851623 | 1.748630 | 0.200957 | 1.611249 | 3.686621 | -0.641969 | 0.801533 |
| NearTrafficControl20 | Stage7D | full_horizon | 31778 | 0.859370 | 1.764527 | 0.199415 | 1.635728 | 3.713864 | -0.604907 | 0.805700 |
| TurningVehicle_GT | Stage3B | full_horizon | 1663 | 5.890031 | 14.452465 | 0.971137 | 7.662410 | 18.571997 | 2.456115 | 5.711947 |
| TurningVehicle_GT | Stage7D | full_horizon | 1663 | 5.877629 | 14.488232 | 0.972339 | 7.999931 | 19.205390 | 2.429763 | 5.707812 |
| GT-left | Stage3B | full_horizon | 831 | 5.586522 | 13.962548 | 0.979543 | 7.308599 | 17.970814 | 2.397509 | 5.406809 |
| GT-left | Stage7D | full_horizon | 831 | 5.579507 | 13.981018 | 0.978339 | 7.676271 | 18.710244 | 2.350929 | 5.402233 |
| GT-right | Stage3B | full_horizon | 832 | 6.193176 | 14.941793 | 0.962740 | 8.015796 | 19.172457 | 2.514649 | 6.016718 |
| GT-right | Stage7D | full_horizon | 832 | 6.175393 | 14.994837 | 0.966346 | 8.323201 | 19.699941 | 2.508503 | 6.013023 |

TurningVehicle_GT remains1663, GT-left831/GT-right832. Endpoint displacement>5m, first/last1s secants>0.5m, absolute wrapped heading change>20°. Intersection/NearTurn/NearControl strict20m whole-centerline context definitions are reused without changes. These masks are never used for training or selection.

【Attention Relevance】

| Group | Model | Count | GTRelevantMass2m | GTRelevantMass4m | GTNearestAttentionWeight | GTNearestAttentionRank | Top1LaneGTDistance | Top3MinGTDistance | Top5MinGTDistance | AttentionEntropy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Overall | Stage3B | 54990 | 0.045253 | 0.074175 | 0.008225 | 116.997127 | 25.772559 | 21.273979 | 19.067218 | 4.593523 |
| Overall | Stage7A | 54990 | 0.042974 | 0.069612 | 0.007339 | 124.043190 | 29.581856 | 25.722232 | 23.146611 | 4.773848 |
| Overall | Stage7D | 54990 | 0.031779 | 0.058692 | 0.006931 | 135.248500 | 28.476286 | 24.503039 | 22.257334 | 4.257747 |
| Vehicle | Stage3B | 42332 | 0.056767 | 0.090513 | 0.009297 | 103.634296 | 25.194898 | 21.066805 | 18.990861 | 4.565671 |
| Vehicle | Stage7A | 42332 | 0.053629 | 0.084946 | 0.008419 | 114.266465 | 29.212699 | 25.440601 | 22.876728 | 4.717238 |
| Vehicle | Stage7D | 42332 | 0.038626 | 0.069480 | 0.007634 | 130.780521 | 28.920971 | 25.057205 | 22.878498 | 4.159576 |
| Pedestrian | Stage3B | 12002 | 0.005967 | 0.018403 | 0.004581 | 163.958424 | 27.831637 | 22.097839 | 19.451943 | 4.684897 |
| Pedestrian | Stage7A | 12002 | 0.006213 | 0.016628 | 0.003642 | 160.123813 | 30.915811 | 26.795493 | 24.212129 | 4.969251 |
| Pedestrian | Stage7D | 12002 | 0.008197 | 0.021669 | 0.004523 | 150.996001 | 26.997487 | 22.704472 | 20.175780 | 4.601084 |
| vehicle.moving | Stage3B | 10461 | 0.210087 | 0.319280 | 0.009351 | 107.609263 | 10.460720 | 6.752240 | 4.915718 | 4.349927 |
| vehicle.moving | Stage7A | 10461 | 0.201039 | 0.305948 | 0.008191 | 110.782669 | 11.174833 | 8.606306 | 7.035317 | 4.666489 |
| vehicle.moving | Stage7D | 10461 | 0.142273 | 0.245319 | 0.006128 | 130.327072 | 14.784719 | 11.762426 | 9.942319 | 4.330640 |
| Vehicle >5m | Stage3B | 9744 | 0.230394 | 0.348545 | 0.009281 | 114.750616 | 9.263441 | 5.960911 | 4.281574 | 4.323981 |
| Vehicle >5m | Stage7A | 9744 | 0.222357 | 0.335697 | 0.008573 | 110.251437 | 9.600451 | 7.288504 | 5.901975 | 4.638025 |
| Vehicle >5m | Stage7D | 9744 | 0.157594 | 0.270062 | 0.006246 | 126.430829 | 13.353505 | 10.606979 | 8.889173 | 4.309049 |
| TurningVehicle_GT | Stage3B | 1663 | 0.176139 | 0.270324 | 0.006987 | 122.110944 | 10.655373 | 6.761460 | 5.000285 | 4.623712 |
| TurningVehicle_GT | Stage7A | 1663 | 0.171935 | 0.258281 | 0.007893 | 103.394768 | 9.819686 | 7.499242 | 6.108371 | 4.791150 |
| TurningVehicle_GT | Stage7D | 1663 | 0.134648 | 0.227607 | 0.005445 | 120.052014 | 13.131451 | 10.182686 | 8.231582 | 4.638620 |
| GT-left | Stage3B | 831 | 0.177179 | 0.276476 | 0.006907 | 97.544525 | 10.676990 | 6.448043 | 4.861738 | 4.627615 |
| GT-left | Stage7A | 831 | 0.141940 | 0.223084 | 0.007323 | 109.774368 | 12.266369 | 9.560345 | 7.959187 | 4.821665 |
| GT-left | Stage7D | 831 | 0.111999 | 0.197778 | 0.004616 | 134.709386 | 14.865950 | 11.797778 | 9.664493 | 4.677491 |
| GT-right | Stage3B | 832 | 0.175100 | 0.264179 | 0.007066 | 146.647837 | 10.633781 | 7.074500 | 5.138666 | 4.619815 |
| GT-right | Stage7A | 832 | 0.201894 | 0.293436 | 0.008463 | 97.022837 | 7.375944 | 5.440616 | 4.259779 | 4.760672 |
| GT-right | Stage7D | 832 | 0.157269 | 0.257401 | 0.006272 | 105.412260 | 11.399037 | 8.569535 | 6.800394 | 4.599796 |

Stage7C post-softmax observer reused;100 random VAL batches PASS, all three output differences<1e-6, dropout disabled, model state unchanged. Baseline/Stage7A attention ledger is frozen. GT relevance is future polyline-to-current-line-segment distance, not segment-start distance. Primary mass population is Vehicle; Moving is prespecified secondary. Nearest rank considers all current graph segments, zero for nonedges, average ties; tie/coverage limitations match Stage7C. Entropy/Top-K distance have missing values for actors with zero incoming edges; paired finite denominators are explicitly recorded in source tables. Attention sums are checked per actor/head. Captured-forward metrics reproduce fresh evaluation within declared numerical tolerance.

Vehicle relevant mass decreases by -0.018142,95%CI=[-0.021368,-0.015067]; Moving relevant mass also decreases reliably. The lower entropy accompanies lower GT relevance and does not establish better selection.

【Turn Consistency】

| Group | Model | Count | CorrectTurnMass | OppositeTurnMass | StraightMass | UnknownTurnMass | TopConnectorMatchRate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TurningVehicle_GT | Stage3B | 1663 | 0.167866 | 0.138249 | 0.308012 | 0.000599 | 0.445580 |
| TurningVehicle_GT | Stage7A | 1663 | 0.136127 | 0.104972 | 0.443093 | 0.001302 | 0.268190 |
| TurningVehicle_GT | Stage7D | 1663 | 0.122745 | 0.105000 | 0.494437 | 0.000727 | 0.117859 |
| GT-left | Stage3B | 831 | 0.178235 | 0.145078 | 0.308653 | 0.001111 | 0.451264 |
| GT-left | Stage7A | 831 | 0.131880 | 0.112424 | 0.452378 | 0.002448 | 0.181709 |
| GT-left | Stage7D | 831 | 0.107182 | 0.123280 | 0.505708 | 0.001313 | 0.068592 |
| GT-right | Stage3B | 832 | 0.157510 | 0.131428 | 0.307372 | 0.000089 | 0.439904 |
| GT-right | Stage7A | 832 | 0.140368 | 0.097530 | 0.433820 | 0.000156 | 0.354567 |
| GT-right | Stage7D | 832 | 0.138291 | 0.086742 | 0.483180 | 0.000142 | 0.167067 |

Highest-attention connector direction match is an offline explanatory match rate, not prediction accuracy. NO_CONNECTOR counts as a nonmatch in the primary denominator. Correct/opposite labels use frozen signed GT heading, and straight/unknown mass are reported separately.

Turning correct-turn mass decreases by -0.045121,95%CI=[-0.057393,-0.033948]; connector match decreases by -0.327721,CI=[-0.387807,-0.270521]. Opposite-turn mass also decreases, while straight mass increases. Neither prespecified mechanism improvement leg is supported; Stage7D does not restore the Stage3B attention pattern. These are descriptive comparisons, not causal explanations of errors.

【Bias Interpretation】

| ActorGroup | SemanticGroup | Edges | mean_bias | mean_abs_bias |
| --- | --- | --- | --- | --- |
| Overall | ordinary lane | 11058260 | 0.399136 | 0.597789 |
| Overall | connector | 15749325 | 1.275207 | 1.615135 |
| Overall | turn_left | 3542494 | 0.697291 | 1.465105 |
| Overall | turn_straight | 8935040 | 1.639723 | 1.729093 |
| Overall | turn_right | 3266648 | 0.907426 | 1.467500 |
| Overall | turn_unknown | 5143 | -0.337520 | 0.746903 |
| Overall | traffic_light_controlled | 8978039 | 1.585581 | 1.741264 |
| Overall | stop_sign_controlled | 1751472 | -0.148804 | 0.856655 |
| Overall | other_control | 10880159 | 1.286053 | 1.688116 |
| Overall | crosswalk_intersects | 10586942 | 1.620055 | 1.953485 |
| Vehicle | ordinary lane | 7853562 | 0.471862 | 0.670540 |
| Vehicle | connector | 11198690 | 1.361664 | 1.717357 |
| Vehicle | turn_left | 2541236 | 0.715884 | 1.550002 |
| Vehicle | turn_straight | 6373237 | 1.786187 | 1.856695 |
| Vehicle | turn_right | 2279524 | 0.898198 | 1.516331 |
| Vehicle | turn_unknown | 4693 | -0.347967 | 0.758275 |
| Vehicle | traffic_light_controlled | 6371931 | 1.704623 | 1.852357 |
| Vehicle | stop_sign_controlled | 1264301 | -0.162528 | 0.940543 |
| Vehicle | other_control | 7734925 | 1.336819 | 1.767570 |
| Vehicle | crosswalk_intersects | 7786241 | 1.670924 | 2.027103 |
| Pedestrian | ordinary lane | 3019447 | 0.227178 | 0.424088 |
| Pedestrian | connector | 4334529 | 1.062395 | 1.363769 |
| Pedestrian | turn_left | 948709 | 0.652550 | 1.249163 |
| Pedestrian | turn_straight | 2437851 | 1.269871 | 1.409417 |
| Pedestrian | turn_right | 947779 | 0.939191 | 1.361213 |
| Pedestrian | turn_unknown | 190 | 0.013805 | 0.656970 |
| Pedestrian | traffic_light_controlled | 2462099 | 1.299403 | 1.476804 |
| Pedestrian | stop_sign_controlled | 467096 | -0.099029 | 0.638939 |
| Pedestrian | other_control | 2994863 | 1.163469 | 1.494849 |
| Pedestrian | crosswalk_intersects | 2668408 | 1.475630 | 1.746845 |
| Bicycle | ordinary lane | 185251 | 0.118802 | 0.344784 |
| Bicycle | connector | 216106 | 1.063406 | 1.359700 |
| Bicycle | turn_left | 52549 | 0.605921 | 1.258112 |
| Bicycle | turn_straight | 123952 | 1.383125 | 1.455463 |
| Bicycle | turn_right | 39345 | 0.676888 | 1.198659 |
| Bicycle | turn_unknown | 260 | -0.405693 | 0.607355 |
| Bicycle | traffic_light_controlled | 144009 | 1.211073 | 1.347202 |
| Bicycle | stop_sign_controlled | 20075 | -0.442620 | 0.639165 |
| Bicycle | other_control | 150371 | 1.116163 | 1.450293 |
| Bicycle | crosswalk_intersects | 132293 | 1.539209 | 1.788670 |

Eight signed head means and absolute means are in stage7d_bias_statistics.csv. Groups overlap. Empty groups have undefined means and do not imply prediction NaNs. Observed type-conditioned means mix different positions; the additional controlled table holds semantic pattern and relative XY fixed, varying only actor type across observed patterns and five fixed XY positions. Maximum raw type response difference=1.698262; maximum type difference in semantic-versus-zero contrast=1.437795. These describe network response, not a physical causal effect or independent evidence of forecasting gain.

【Bootstrap】

| Group | Metric | Count | Stage3B | Stage7D | Delta | CI_lower | CI_upper | Scenes | NonemptyScenes | Replicates | Seed | Multiplicity |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Overall | minFDE6 | 54990 | 1.343260 | 1.333902 | -0.009358 | -0.021728 | 0.002292 | 150 | 150 | 1000 | 2022 | primary |
| Vehicle | minFDE6 | 42332 | 1.431914 | 1.447114 | 0.015201 | 0.002333 | 0.028877 | 150 | 150 | 1000 | 2022 | unadjusted secondary |
| Pedestrian | minFDE6 | 12002 | 1.043875 | 0.948811 | -0.095065 | -0.122432 | -0.072928 | 150 | 121 | 1000 | 2022 | unadjusted secondary |
| vehicle.moving | minFDE6 | 10461 | 4.791532 | 4.838301 | 0.046769 | -0.004718 | 0.101980 | 150 | 137 | 1000 | 2022 | unadjusted secondary |
| Vehicle >5m | minFDE6 | 9744 | 5.579887 | 5.625346 | 0.045459 | -0.009084 | 0.099506 | 150 | 137 | 1000 | 2022 | unadjusted secondary |
| IntersectionVehicle20 | minFDE6 | 32085 | 1.760552 | 1.778350 | 0.017798 | 0.001654 | 0.034963 | 150 | 150 | 1000 | 2022 | unadjusted secondary |
| NearTurnConnector20 | minFDE6 | 24949 | 1.976263 | 1.986333 | 0.010070 | -0.007138 | 0.027233 | 150 | 147 | 1000 | 2022 | unadjusted secondary |
| NearTrafficControl20 | minFDE6 | 31778 | 1.748630 | 1.764527 | 0.015897 | -0.000687 | 0.033047 | 150 | 147 | 1000 | 2022 | unadjusted secondary |
| TurningVehicle_GT | minFDE6 | 1663 | 14.452465 | 14.488232 | 0.035768 | -0.013638 | 0.098855 | 150 | 106 | 1000 | 2022 | unadjusted secondary |
| Overall | Top1FDE6 | 54990 | 2.687984 | 2.712028 | 0.024044 | -0.000388 | 0.050639 | 150 | 150 | 1000 | 2022 | unadjusted secondary |
| Vehicle | GTRelevantMass2m | 42332 | 0.056767 | 0.038626 | -0.018142 | -0.021368 | -0.015067 | 150 | 150 | 1000 | 2022 | unadjusted secondary |
| vehicle.moving | GTRelevantMass2m | 10461 | 0.210087 | 0.142273 | -0.067814 | -0.073559 | -0.061897 | 150 | 137 | 1000 | 2022 | unadjusted secondary |
| Vehicle | GTNearestAttentionRank | 42332 | 103.634296 | 130.780521 | 27.146225 | 22.163619 | 32.536084 | 150 | 150 | 1000 | 2022 | unadjusted secondary |
| Vehicle | AttentionEntropy | 41974 | 4.565671 | 4.159576 | -0.406095 | -0.442043 | -0.367617 | 150 | 150 | 1000 | 2022 | unadjusted secondary |
| TurningVehicle_GT | CorrectTurnMass | 1663 | 0.167866 | 0.122745 | -0.045121 | -0.057393 | -0.033948 | 150 | 106 | 1000 | 2022 | unadjusted secondary |
| TurningVehicle_GT | TopConnectorMatchRate | 1663 | 0.445580 | 0.117859 | -0.327721 | -0.387807 | -0.270521 | 150 | 106 | 1000 | 2022 | unadjusted secondary |
| TurningVehicle_GT | OppositeTurnMass | 1663 | 0.138249 | 0.105000 | -0.033249 | -0.040738 | -0.025990 | 150 | 106 | 1000 | 2022 | unadjusted secondary |

Paired whole-scene bootstrap:150 official VAL scenes,1000 draws,seed2022, actor-window sums/counts pooled for each draw,percentile95CI. Delta=Stage7D−Stage3B. Forecasting improvement has negative delta; attention mass/match improvement has positive delta. Primary Overall minFDE is distinct from exploratory unadjusted overlapping subgroup/mechanism comparisons. No multiple-comparison correction or seed variance estimate. Per-scene sufficient statistics are archived.

【Efficiency】

| model | parameters | additional_parameters | parameter_increase_percent | paired_measurements | mean_forward_ms | median_forward_ms | peak_CUDA_allocated_MiB | peak_forward_increment_MiB | device | batch_size | unique_VAL_batches | benchmark |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Stage3B | 646001 | 0 | 0.000000 | 500 | 35.892893 | 31.186640 | 254.273926 | 231.224609 | NVIDIA GeForce RTX 3080 | 16 | 8 | CUDA events; synchronized; alternating order; both models resident; baseline original fields and Stage7D augmented fields share original tensors; data loading excluded; warmed8batches3pairs |
| Stage7D | 646745 | 744 | 0.115170 | 500 | 36.289583 | 31.662592 | 257.874512 | 234.825195 | NVIDIA GeForce RTX 3080 | 16 | 8 | CUDA events; synchronized; alternating order; both models resident; baseline original fields and Stage7D augmented fields share original tensors; data loading excluded; warmed8batches3pairs |

500 paired forward measurements, alternating model order,eight fixed VAL batches,batch16; warmed3 paired passes per batch,CUDA events and synchronization. Data loading excluded; both models resident. Baseline receives original fields and shares original tensors with semantic input. Absolute and incremental peak allocated memory are reported; hardware/timing environment limits generalization.

【Attention-Error Association】

| Group | X | Y | SpearmanR | p_value | Count | Interpretation |
| --- | --- | --- | --- | --- | --- | --- |
| Overall | DeltaRelevantMass | DeltaMinFDE | -0.042901 | 0.000000 | 54990 | descriptive association; clustered observations; p-value does not replace scene bootstrap |
| Overall | DeltaRelevantMass | DeltaTop1FDE | -0.019873 | 0.000003 | 54990 | descriptive association; clustered observations; p-value does not replace scene bootstrap |
| Overall | DeltaGTNearestAttentionRank | DeltaMinFDE | 0.020689 | 0.000001 | 54990 | descriptive association; clustered observations; p-value does not replace scene bootstrap |
| Overall | DeltaGTNearestAttentionRank | DeltaTop1FDE | 0.012794 | 0.002698 | 54990 | descriptive association; clustered observations; p-value does not replace scene bootstrap |
| Vehicle | DeltaRelevantMass | DeltaMinFDE | -0.027817 | 0.000000 | 42332 | descriptive association; clustered observations; p-value does not replace scene bootstrap |
| Vehicle | DeltaRelevantMass | DeltaTop1FDE | -0.002035 | 0.675484 | 42332 | descriptive association; clustered observations; p-value does not replace scene bootstrap |
| Vehicle | DeltaGTNearestAttentionRank | DeltaMinFDE | 0.008125 | 0.094574 | 42332 | descriptive association; clustered observations; p-value does not replace scene bootstrap |
| Vehicle | DeltaGTNearestAttentionRank | DeltaTop1FDE | -0.005960 | 0.220138 | 42332 | descriptive association; clustered observations; p-value does not replace scene bootstrap |
| Pedestrian | DeltaRelevantMass | DeltaMinFDE | -0.031703 | 0.000513 | 12002 | descriptive association; clustered observations; p-value does not replace scene bootstrap |
| Pedestrian | DeltaRelevantMass | DeltaTop1FDE | -0.066172 | 0.000000 | 12002 | descriptive association; clustered observations; p-value does not replace scene bootstrap |
| Pedestrian | DeltaGTNearestAttentionRank | DeltaMinFDE | 0.018851 | 0.038910 | 12002 | descriptive association; clustered observations; p-value does not replace scene bootstrap |
| Pedestrian | DeltaGTNearestAttentionRank | DeltaTop1FDE | 0.042222 | 0.000004 | 12002 | descriptive association; clustered observations; p-value does not replace scene bootstrap |
| vehicle.moving | DeltaRelevantMass | DeltaMinFDE | 0.001403 | 0.885877 | 10461 | descriptive association; clustered observations; p-value does not replace scene bootstrap |
| vehicle.moving | DeltaRelevantMass | DeltaTop1FDE | -0.048500 | 0.000001 | 10461 | descriptive association; clustered observations; p-value does not replace scene bootstrap |
| vehicle.moving | DeltaGTNearestAttentionRank | DeltaMinFDE | 0.028978 | 0.003035 | 10461 | descriptive association; clustered observations; p-value does not replace scene bootstrap |
| vehicle.moving | DeltaGTNearestAttentionRank | DeltaTop1FDE | -0.027436 | 0.005012 | 10461 | descriptive association; clustered observations; p-value does not replace scene bootstrap |
| Vehicle >5m | DeltaRelevantMass | DeltaMinFDE | 0.000094 | 0.992565 | 9744 | descriptive association; clustered observations; p-value does not replace scene bootstrap |
| Vehicle >5m | DeltaRelevantMass | DeltaTop1FDE | -0.042488 | 0.000027 | 9744 | descriptive association; clustered observations; p-value does not replace scene bootstrap |
| Vehicle >5m | DeltaGTNearestAttentionRank | DeltaMinFDE | 0.021101 | 0.037259 | 9744 | descriptive association; clustered observations; p-value does not replace scene bootstrap |
| Vehicle >5m | DeltaGTNearestAttentionRank | DeltaTop1FDE | -0.023096 | 0.022613 | 9744 | descriptive association; clustered observations; p-value does not replace scene bootstrap |
| TurningVehicle_GT | DeltaRelevantMass | DeltaMinFDE | -0.081923 | 0.000826 | 1663 | descriptive association; clustered observations; p-value does not replace scene bootstrap |
| TurningVehicle_GT | DeltaRelevantMass | DeltaTop1FDE | -0.041618 | 0.089764 | 1663 | descriptive association; clustered observations; p-value does not replace scene bootstrap |
| TurningVehicle_GT | DeltaGTNearestAttentionRank | DeltaMinFDE | 0.072279 | 0.003186 | 1663 | descriptive association; clustered observations; p-value does not replace scene bootstrap |
| TurningVehicle_GT | DeltaGTNearestAttentionRank | DeltaTop1FDE | 0.021937 | 0.371311 | 1663 | descriptive association; clustered observations; p-value does not replace scene bootstrap |
| TurningVehicle_GT | DeltaCorrectTurnMass | DeltaMinFDE | -0.104754 | 0.000019 | 1663 | descriptive association; clustered observations; p-value does not replace scene bootstrap |
| GT-left | DeltaRelevantMass | DeltaMinFDE | -0.093316 | 0.007106 | 831 | descriptive association; clustered observations; p-value does not replace scene bootstrap |
| GT-left | DeltaRelevantMass | DeltaTop1FDE | -0.038296 | 0.270156 | 831 | descriptive association; clustered observations; p-value does not replace scene bootstrap |
| GT-left | DeltaGTNearestAttentionRank | DeltaMinFDE | 0.051465 | 0.138250 | 831 | descriptive association; clustered observations; p-value does not replace scene bootstrap |
| GT-left | DeltaGTNearestAttentionRank | DeltaTop1FDE | 0.029310 | 0.398763 | 831 | descriptive association; clustered observations; p-value does not replace scene bootstrap |
| GT-left | DeltaCorrectTurnMass | DeltaMinFDE | 0.000704 | 0.983838 | 831 | descriptive association; clustered observations; p-value does not replace scene bootstrap |
| GT-right | DeltaRelevantMass | DeltaMinFDE | 0.019908 | 0.566359 | 832 | descriptive association; clustered observations; p-value does not replace scene bootstrap |
| GT-right | DeltaRelevantMass | DeltaTop1FDE | 0.007541 | 0.828058 | 832 | descriptive association; clustered observations; p-value does not replace scene bootstrap |
| GT-right | DeltaGTNearestAttentionRank | DeltaMinFDE | -0.019238 | 0.579487 | 832 | descriptive association; clustered observations; p-value does not replace scene bootstrap |
| GT-right | DeltaGTNearestAttentionRank | DeltaTop1FDE | -0.033750 | 0.330895 | 832 | descriptive association; clustered observations; p-value does not replace scene bootstrap |
| GT-right | DeltaCorrectTurnMass | DeltaMinFDE | -0.095877 | 0.005645 | 832 | descriptive association; clustered observations; p-value does not replace scene bootstrap |

Spearman and p-values are descriptive only, with clustered/overlapping observations. They do not replace scene bootstrap or support causality. Checkpoint selection did not use these associations.

【Case Studies】

```json
{
  "cases": [
    {
      "case": "case_turning_improvement",
      "title": "Turning vehicle improvement",
      "scene_token": "e7ef871f77f44331aefdebc24ec034b7",
      "sample_token": "2f5de0aeca704127925cf8490ff5a21d",
      "instance_token": "5dc88cb5574141fb8781eda219ffdd50",
      "candidate_count": 887,
      "DeltaMinFDE": -2.5582048892974854,
      "DeltaRelevantMass": 0.0,
      "DeltaCorrectTurnMass": 0.0,
      "DeltaOppositeTurnMass": 0.0
    },
    {
      "case": "case_moving_improvement",
      "title": "Moving vehicle improvement",
      "scene_token": "d01e7279da2649ef896dc42f6b9ee7ab",
      "sample_token": "10103847ab7d4b1db6c1c046c9e38097",
      "instance_token": "307059ee04af43b58cf7b9d540bd85ae",
      "candidate_count": 5319,
      "DeltaMinFDE": -12.765020370483397,
      "DeltaRelevantMass": -0.127237603261015,
      "DeltaCorrectTurnMass": null,
      "DeltaOppositeTurnMass": null
    },
    {
      "case": "case_degradation",
      "title": "Failure / degradation",
      "scene_token": "6a24a80e2ea3493c81f5fdb9fe78c28a",
      "sample_token": "b7e99fb76ffa4e339880fa0a233e8f27",
      "instance_token": "5859747aae214c83999a764fb30494ec",
      "candidate_count": 4724,
      "DeltaMinFDE": 15.73544788360596,
      "DeltaRelevantMass": 0.1676521574894672,
      "DeltaCorrectTurnMass": null,
      "DeltaOppositeTurnMass": null
    }
  ],
  "count": 3,
  "distinct_identities": true,
  "training": false
}
```

Three prespecified extreme-delta illustrations: turning improvement,moving improvement,degradation,distinct actor identities. Every panel shares actor,GT,map,axis limits, and attention color scale. Both model best-FDE trajectories appear with semantic turn colors and separate control/crosswalk markers. Cases illustrate possibilities and cannot establish their prevalence or the overall decision.

The turning improvement has zero relevant/correct-turn mass in both models for its retained edges. The moving improvement accompanies lower relevant mass; the degradation accompanies higher relevant mass. These examples show that relevance shifts and error shifts need not follow a monotonic relationship for individual actors.

【Limitations】

Static map semantics only; no dynamic light states. Turn labels derive from audited centerlines and fixed20°/150° rules. GT relevance/turning/endpoint groups are offline explanatory definitions unavailable to model inputs. One training seed; checkpoint selection and analysis use official VAL, with no independent test claim. The same early-stop protocol can select different actual update counts; this is not an equal-compute comparison. Attention rank ties, segment density, local-edge truncation and empty incoming sets constrain interpretation. Secondary groups overlap and their CIs are unadjusted. Semantic bias changes selection directly, while shared geometry-only representations can change indirectly through training. The bias also consumes relative geometry and actor type, so performance differences do not isolate semantic information from those additional score inputs. No reliability fusion was executed.

【Scientific Decision】

```text
Stage7D = NOT_SUPPORTED
AttentionMechanism = NOT_SUPPORTED
PaperUsableSemantic = NO
SemanticRoute = STOP
ReadySemanticReranking = NO
```

Registered performance and mechanism rules are evaluated separately. SUPPORTED uses the inherited marked-harm guard (>5% relative AND reliable worsening); TARGETED_SUPPORTED rejects any reliable Vehicle/Pedestrian worsening and permits at most0.5% overall point increase with CI including0. Mechanism PARTIAL means exactly one of the two prespecified improvement legs. No result alters these rules.

Overall delta=-0.009358,95%CI=[-0.021728,+0.002292]. Vehicle delta=+0.015201,CI=[+0.002333,+0.028877]. Pedestrian delta=-0.095065,CI=[-0.122432,-0.072928]. Reliably improved prespecified targeted groups=[]; reliable Vehicle/Pedestrian harms=['Vehicle']. These, rather than case examples or attention changes, determine forecasting support.

Semantic modeling experiments Stage7A + Stage7D did not produce reliable forecasting gains under the frozen protocol. This decision concerns the prespecified overall/priority-vehicle support rules; any Pedestrian improvement is reported above and does not override those rules. Stop the semantic architecture route; the paper primary model returns to Stage6A R2. R2 is not executed in this stage.

STOP. Stage7E, Stage7B and R2 remain unexecuted. Await review; no merge to main.
