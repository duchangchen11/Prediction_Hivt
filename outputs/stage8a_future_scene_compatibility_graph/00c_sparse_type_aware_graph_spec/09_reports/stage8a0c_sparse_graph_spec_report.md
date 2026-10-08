# Stage8A-0C sparse type-aware semantic graph final specification audit

## Scope and frozen history

Stage8A-0 remains NOT_READY. Stage8A-0B remains MIXED / coverage FAIL / context STRONG / NOT_READY. This stage specifies sparse inputs and changes the user-supplied engineering support gates before any graph training. These thresholds are neither accuracy thresholds nor significance/paper performance claims. No prediction performance was evaluated.

## Deterministic selector and fixed quotas

Six entity types only. Vehicle quotas lane3/connector3/drivable1/carpark1; pedestrian walkway3/crossing2/drivable1; bicycle lane3/connector3/drivable1. Within each type sort true geometry distance then lexical token. Motion state never changes the selector. No standalone stop line, traffic-light point, nearest fallback or learned NULL token. Real multipart polygons are preserved component by component; Boston’s invalid walkway component stays excluded.

## Feature contract

Actor-mode15D and directed interaction17D are imported directly from the unchanged Stage8A-0 pure module. Map-node18D = type onehot6 + frozen lane semantic9 + local tangent sin/cos/valid3. Map-edge11D names and units are frozen in the manifest. Polygon nine lane bits and tangent/heading fields are exact zero. There is no area, priority, right-of-way or dynamic signal feature. Lane tangent uses the true nearest complete-centerline point, preserving repeated-source-vertex handling. Strict inside/intersection features use OR over original valid multipart components, avoiding invalid collection topology operations without repair.

## Empty-map behavior

Synthetic far-away candidates return zero selected entities. Both actual empty tensors and all-masked padding produce exact zero64D, even after setting encoder biases to nonzero. A patched Tensor.softmax verifies the empty-map code path never calls softmax. Mixed empty/nonempty rows remain finite. LayerNorm(h_node+m_int+0) is valid. No empty target is deleted.

## Entity retention and semantic presence

All authorized matched types preserve presence at100%. Retention is entity-weighted selected/uncapped and naturally falls when a type exceeds its fixed quota. Unauthorized types have no eligible entities and undefined retention/presence ratios; they do not enter the graph.

| Split | ActorGroup | UncappedEntities | SelectedEntities | RetentionRate | PresencePreservationRate |
| --- | --- | --- | --- | --- | --- |
| train | Vehicle | 7840956 | 5459730 | 69.6309% | 100.0000% |
| train | Pedestrian | 822165 | 771671 | 93.8584% | 100.0000% |
| train | Bicycle | 117024 | 82862 | 70.8077% | 100.0000% |
| val | Vehicle | 1605388 | 1067460 | 66.4923% | 100.0000% |
| val | Pedestrian | 132761 | 128086 | 96.4786% | 100.0000% |
| val | Bicycle | 21545 | 17587 | 81.6291% | 100.0000% |

## GlobalTop8 versus quota

The primary comparison uses the same actor-authorized pool. All-six GlobalTop8, including secondary contexts, is separately descriptive. Quota never loses an authorized present type; no performance conclusion follows.

| Split | Group | Selector | MeanSelectedEntities | MeanDistinctEntityTypes | LanePresence | ConnectorPresence | DrivablePresence | CarparkPresence | CrosswalkPresence | WalkwayPresence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | Overall | GlobalTop8 | 3.988776278217304 | 2.208770417866028 | 0.5562599467972031 | 0.4167519864867194 | 0.7822167065974915 | 0.19589028503139885 | 0.06251730814531373 | 0.19513418480790112 |
| train | Overall | TypeAwareQuota | 3.6278234540450787 | 2.2285410598043103 | 0.5717025469546282 | 0.4170817748820748 | 0.7851997403060023 | 0.19690550470839008 | 0.06251730814531373 | 0.19513418480790112 |
| train | Pedestrian | GlobalTop8 | 1.8719092374525286 | 1.6318156152381993 | 0.0 | 0.0 | 0.6107938763057476 | 0.0 | 0.24774368642022532 | 0.7732780525122265 |
| train | Pedestrian | TypeAwareQuota | 1.7569442547107093 | 1.6318156152381993 | 0.0 | 0.0 | 0.6107938763057476 | 0.0 | 0.24774368642022532 | 0.7732780525122265 |
| train | Bicycle | GlobalTop8 | 5.032982791586998 | 2.193329084342469 | 0.8360951773953686 | 0.7210006373486297 | 0.6362332695984704 | 0.0 | 0.0 | 0.0 |
| train | Bicycle | TypeAwareQuota | 4.400998512853198 | 2.235234756745273 | 0.877151051625239 | 0.7210006373486297 | 0.6370830677714043 | 0.0 | 0.0 | 0.0 |
| train | MovingVehicle | GlobalTop8 | 7.299832603198348 | 2.8221284325248424 | 0.8987961676817324 | 0.8981194572069665 | 0.9707091213448731 | 0.054503686291270434 | 0.0 | 0.0 |
| train | MovingVehicle | TypeAwareQuota | 6.31064572425829 | 2.9251878761975996 | 0.9777932115254478 | 0.8985682231007587 | 0.9882751006161627 | 0.06055134095523026 | 0.0 | 0.0 |
| val | Overall | GlobalTop8 | 4.071816087773534 | 2.190610414014669 | 0.5560223070861369 | 0.4297811723343638 | 0.7604140146693338 | 0.22967812329514456 | 0.053530945020306725 | 0.16118385160938353 |
| val | Overall | TypeAwareQuota | 3.6768291204461416 | 2.2137722010062437 | 0.5730890464933018 | 0.4305146390252773 | 0.7653906770928047 | 0.23006304176516942 | 0.053530945020306725 | 0.16118385160938353 |
| val | Pedestrian | GlobalTop8 | 1.8435955118591345 | 1.6193967672054657 | 0.0 | 0.0 | 0.6356301727489863 | 0.0 | 0.24526467810920402 | 0.7385019163472755 |
| val | Pedestrian | TypeAwareQuota | 1.7786757762595122 | 1.6193967672054657 | 0.0 | 0.0 | 0.6356301727489863 | 0.0 | 0.24526467810920402 | 0.7385019163472755 |
| val | Bicycle | GlobalTop8 | 4.854420731707317 | 2.2558434959349594 | 0.838160569105691 | 0.7146849593495935 | 0.7029979674796748 | 0.0 | 0.0 | 0.0 |
| val | Bicycle | TypeAwareQuota | 4.468241869918699 | 2.28760162601626 | 0.8678861788617886 | 0.7146849593495935 | 0.7050304878048781 | 0.0 | 0.0 | 0.0 |
| val | MovingVehicle | GlobalTop8 | 7.450100372813306 | 2.8474970525443712 | 0.9074339610617214 | 0.9310614026702355 | 0.9648217187649364 | 0.04417997004747794 | 0.0 | 0.0 |
| val | MovingVehicle | TypeAwareQuota | 6.428958353248574 | 2.946786476754931 | 0.9804352675015136 | 0.9311888602109423 | 0.9894050919287513 | 0.04575725711372399 | 0.0 | 0.0 |

## Sparse coverage and zero-map rates

Coverage is MapNonEmptyRate, without fallback. All current-valid and full-horizon targets remain in their natural populations. Full-horizon TRAIN/VAL results follow; complete current-valid results are in the CSV.

| Split | Group | Actors | Candidates | MapNonEmptyRate | ZeroMapRate | MeanEntitiesPerMode | MedianEntities | P95Entities | MaxEntities |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | Overall | 290085 | 1740510 | 89.1159% | 10.8841% | 3.6278234540450787 | 3.0 | 7.0 | 8 |
| train | Vehicle | 213745 | 1282470 | 88.6942% | 11.3058% | 4.25719899880699 | 4.0 | 7.0 | 8 |
| train | Pedestrian | 73202 | 439212 | 90.2009% | 9.7991% | 1.7569442547107093 | 2.0 | 3.0 | 6 |
| train | Bicycle | 3138 | 18828 | 92.5271% | 7.4729% | 4.400998512853198 | 5.0 | 7.0 | 7 |
| train | MovingVehicle | 46795 | 280770 | 99.0042% | 0.9958% | 6.31064572425829 | 7.0 | 7.0 | 8 |
| train | StoppedVehicle | 25939 | 155634 | 97.7017% | 2.2983% | 5.795616639037743 | 7.0 | 7.0 | 8 |
| train | ParkedVehicle | 136238 | 817428 | 83.1901% | 16.8099% | 3.2407245653439816 | 3.0 | 7.0 | 8 |
| train | UnknownVehicle | 4773 | 28638 | 95.7714% | 4.2286% | 4.778126964173476 | 5.0 | 7.0 | 8 |
| val | Overall | 54990 | 329940 | 86.0772% | 13.9228% | 3.6768291204461416 | 4.0 | 7.0 | 8 |
| val | Vehicle | 42332 | 253992 | 84.6523% | 15.3477% | 4.202730794670698 | 5.0 | 7.0 | 8 |
| val | Pedestrian | 12002 | 72012 | 90.9932% | 9.0068% | 1.7786757762595122 | 2.0 | 4.0 | 5 |
| val | Bicycle | 656 | 3936 | 88.0843% | 11.9157% | 4.468241869918699 | 5.0 | 7.0 | 7 |
| val | MovingVehicle | 10461 | 62766 | 99.1970% | 0.8030% | 6.428958353248574 | 7.0 | 7.0 | 8 |
| val | StoppedVehicle | 5599 | 33594 | 97.1543% | 2.8457% | 5.658927189379056 | 6.0 | 7.0 | 8 |
| val | ParkedVehicle | 25198 | 151188 | 75.6118% | 24.3882% | 2.937891896182237 | 3.0 | 7.0 | 8 |
| val | UnknownVehicle | 1074 | 6444 | 89.9131% | 10.0869% | 4.602731222842955 | 6.0 | 7.0 | 8 |

## Engineering support gates

Every gate applies independently in TRAIN and VAL full-horizon modes. The new thresholds were explicitly provided before graph training and are registered before this audit. They do not retroactively change the0B decision.

| Split | Group | Metric | Value | Threshold | PASS |
| --- | --- | --- | --- | --- | --- |
| train | MovingVehicle | RouteCenterlineCoverage | 0.9869929123481853 | 0.95 | True |
| train | Vehicle | MapNonEmptyRate | 0.886942384617184 | 0.8 | True |
| train | ParkedVehicle | MapNonEmptyRate | 0.8319007912623497 | 0.7 | True |
| train | Pedestrian | MapNonEmptyRate | 0.9020085972150124 | 0.85 | True |
| train | Pedestrian | SemanticSpecificCoverage | 0.8205718422993907 | 0.7 | True |
| train | Bicycle | MapNonEmptyRate | 0.9252708731676227 | 0.8 | True |
| val | MovingVehicle | RouteCenterlineCoverage | 0.9900742440174617 | 0.95 | True |
| val | Vehicle | MapNonEmptyRate | 0.8465227251252008 | 0.8 | True |
| val | ParkedVehicle | MapNonEmptyRate | 0.7561182104399821 | 0.7 | True |
| val | Pedestrian | MapNonEmptyRate | 0.9099316780536577 | 0.85 | True |
| val | Pedestrian | SemanticSpecificCoverage | 0.7864383713825474 | 0.7 | True |
| val | Bicycle | MapNonEmptyRate | 0.8808434959349594 | 0.8 | True |

## Graph size

Verified maxima: vehicle48, pedestrian36, bicycle42 mode-map edges per target; mode-mode<=288 and unchanged50m/8-actor neighbor selection. ZeroMapTargetRate means all six modes lack map context; it differs from per-mode ZeroMapRate.

| Split | Group | Targets | MeanModeMapEdges | P95ModeMapEdges | MaxModeMapEdges | MeanModeModeEdges | P95ModeModeEdges | ZeroMapTargetRate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | Overall | 290085 | 21.766940724270473 | 42.0 | 48 | 272.9446196804385 | 288.0 | 0.08442008376855059 |
| train | Vehicle | 213745 | 25.54319399284194 | 42.0 | 48 | 270.7278486046457 | 288.0 | 0.08577510585042925 |
| train | Pedestrian | 73202 | 10.541665528264256 | 20.0 | 30 | 279.43646348460425 | 288.0 | 0.08208792109505204 |
| train | Bicycle | 3138 | 26.405991077119186 | 42.0 | 42 | 272.50095602294454 | 288.0 | 0.04652644996813257 |
| train | MovingVehicle | 46795 | 37.86387434554974 | 42.0 | 48 | 254.93339031947858 | 288.0 | 0.007821348434661823 |
| train | StoppedVehicle | 25939 | 34.773699834226456 | 42.0 | 48 | 268.3685569991133 | 288.0 | 0.019507305601603762 |
| train | ParkedVehicle | 136238 | 19.44434739206389 | 42.0 | 48 | 276.5881178525815 | 288.0 | 0.12714514305847122 |
| train | UnknownVehicle | 4773 | 28.668761785040854 | 43.0 | 48 | 271.12759270898806 | 288.0 | 0.02933165723863398 |
| val | Overall | 54990 | 22.06097472267685 | 42.0 | 48 | 269.0644844517185 | 288.0 | 0.11854882705946536 |
| val | Vehicle | 42332 | 25.21638476802419 | 42.0 | 48 | 266.4188793347822 | 288.0 | 0.13011433430974204 |
| val | Pedestrian | 12002 | 10.672054657557073 | 22.0 | 30 | 278.01166472254624 | 288.0 | 0.07973671054824195 |
| val | Bicycle | 656 | 26.809451219512194 | 42.0 | 42 | 276.0914634146341 | 288.0 | 0.08231707317073171 |
| val | MovingVehicle | 10461 | 38.57375011949144 | 42.0 | 48 | 248.13880126182966 | 288.0 | 0.004684064620973138 |
| val | StoppedVehicle | 5599 | 33.95356313627433 | 42.0 | 48 | 265.17449544561526 | 288.0 | 0.025183068405072333 |
| val | ParkedVehicle | 25198 | 17.62735137709342 | 42.0 | 48 | 275.1318358599889 | 288.0 | 0.20747678387173585 |
| val | UnknownVehicle | 1074 | 27.616387337057727 | 42.0 | 44 | 246.53631284916202 | 288.0 | 0.08379888268156424 |

## Static models and inference memory

No optimizer, backward, fitting, tiny overfit or checkpoint. G1 is bitwise identical to the old static prototype; shared node encoder/LayerNorm/head initialization is identical across variants. The new map message uses44D concatenated input and map attention93D input. Final64→32→1 layer weight/bias are exact zero.

| Variant | Parameters | InputDims | HiddenDim | MaxMapEdges | MaxInteractionEdges | NeutralOutputMaxDiff | EstimatedMemoryMiB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| G1 | 24066 | Node15; Interaction17; MapNode18; MapEdge11 | 64 | 0 | 288 | 0.0 | 117.33740234375 |
| G2 | 20546 | Node15; Interaction17; MapNode18; MapEdge11 | 64 | 48 | 0 | 0.0 | 73.6484375 |
| G3 | 37187 | Node15; Interaction17; MapNode18; MapEdge11 | 64 | 48 | 288 | 0.0 | 117.38916015625 |

Batch128 input bytes: 3644416; inference budget maximum 117.389160 MiB, using allowed static CUDA peak plus fixed32MiB margin. This excludes the frozen predictor and retrieval construction. Training activation/optimizer memory is unmeasured and must be measured in separately authorized Stage8A-1 tiny work.

## Candidate, coordinates, leakage and semantics

200 sampled windows (100 TRAIN/100 VAL, seed2022) freshly audited all graph features. Candidate/logit/probability maxdiff=0. Poisoned GT=NaN and inverted future/target masks leave IDs/types/distances, map-node18/map-edge11, actor-mode15 and interaction17 features bitwise unchanged. Full-horizon masks only determine offline populations. Semantic edges checked: 106398; lane9 equals frozen Stage7A, polygon9=0, type onehot exact. Global/local geometry-distance discrepancy max 6.8212102633e-13 m (<1e-5). Candidate roundtrip max 2.84217094304e-13 m; polygon roundtrip max 9.09494701773e-13 m. All1283 source and historical graph cache SHAs were reverified. Selector import audit finds no evaluation/error/ranking-label/best-mode imports.

## Selector determinism

Three independent runs used worker counts1/2/3, chunk sizes1/4/7, ascending/reverse/shuffled window and candidate order, and completion-order gathering. All200 sampled windows’ IDs/types/distances/counts/masks and token ordering are bitwise identical.

| Run | Workers | ChunkWindows | WindowOrdering | CandidateOrdering | Windows | Candidates | SelectorSHA256 | BitwiseIdentical |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 1 | ascending | ascending | 200 | 31896 | ac498ede3fe0747495c5dc489477fd1cf21d221f00ab36e7d374cb6f7bb36775 | PASS |
| 2 | 2 | 4 | reverse | reverse | 200 | 31896 | ac498ede3fe0747495c5dc489477fd1cf21d221f00ab36e7d374cb6f7bb36775 | PASS |
| 3 | 3 | 7 | seed2022_shuffle | seed2022_shuffle | 200 | 31896 | ac498ede3fe0747495c5dc489477fd1cf21d221f00ab36e7d374cb6f7bb36775 | PASS |

## Frozen contract and decision

The final manifest lists taxonomy, actor-primary mapping, all feature names, quotas, radii, empty-map behavior, neighbors, all thresholds and relevant source/code hashes. Hidden64 and ranking temperature1m are frozen. It must remain unchanged in Stage8A-1 unless a concrete code bug is identified.

SparseMapSelector = **PASS**

SemanticPresencePreservation = **PASS**

SparseCoverage = **PASS**

CandidateIdentity = **PASS**

Coordinate = **PASS**

NoFutureLeakage = **PASS**

SelectorDeterminism = **PASS**

MemoryFeasible = **YES**

GraphSpecFrozen = **YES**

Stage8A_0C = **READY**

ReadyStage8A1 = **YES**

**STOP. No automatic G1/G2/G3 training. Await scientific review.**
