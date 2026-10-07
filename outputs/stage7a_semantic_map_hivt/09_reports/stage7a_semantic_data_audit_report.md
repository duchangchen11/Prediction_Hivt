# Stage7A-0 semantic map data audit

Audit only. No model training, architecture change or predictor/R2 forward pass. New work remains in this independent Stage7A root.

【Frozen V1】

Frozen Stage6A commit: `e43347c2c2cd356775f4e0bb59fcf7dbda76f2b5`. Annotated tag `paper-v1-stage6a-r2` targets exactly this commit; tag object `bd70aeb7d61ec624f2559438b9ef5670d8c1e564`. Message: `Freeze paper V1: Stage5A + future interaction reliability R2`. The pre-tag workspace was clean after a path-scoped stash of the five unrelated Stage2C redraw files; all five were restored byte-identically and remain unsubmitted.

Tag remote status: **BLOCKED_LOCAL_GIT_AUTHENTICATION**. Tag is local; GitHub tag push remains blocked by absent local Git credentials. This delivery limitation does not change map-data feasibility. The branch can be published via the authorized GitHub connector.

| Reference | path | sha256 |
| --- | --- | --- |
| Stage5A_checkpoint | outputs/stage5a_motion_aware_decoder/07_checkpoints/stage5a_best_overall_minfde.pt | 88fe3feb59b7e830ec1e40aa917484386e0d2799d0f6116f410adda2d97fdce7 |
| R2_checkpoint | outputs/stage6a_future_interaction_reliability/07_checkpoints/stage6a_r2_best.pt | e3de457d635cd30c629ce316d73a87e419a3ded8abbe0e1f2601f1071527e314 |
| Stage6A_config | outputs/stage6a_future_interaction_reliability/00_manifest/stage6a_config.json | fe097f01f021e7cbf6503c877baa2ab01b2cf1b59f4a7a7325063cb250c0f31c |
| Stage6A_normalization | outputs/stage6a_future_interaction_reliability/02_features/stage6a_normalization.json | be045dc0ad0d1198bcfa8f8c270ab77297984597ee6891516e5f84ccc4e9b2d2 |
| Stage6A_report | outputs/stage6a_future_interaction_reliability/09_reports/stage6a_final_report.md | 84ead326a14e9571e6a2ae06e4051e1aeea2dc1d6f1064dcc0d05d822e207c07 |
| Stage6A_actor_results | outputs/stage6a_future_interaction_reliability/04_evaluation/stage6a_actor_ranking.csv | 2184908eff7b0788d388f2dfe85d41df13a32efd601d2d18c1c78524bed8cbe2 |

Frozen results: Overall minFDE6=1.337995889 m; Overall R2 Top1FDE6=2.645562578 m. Vehicle=3.010194075 m; Pedestrian=1.418252746 m; vehicle.moving=10.450046384 m. ReliabilityHead=SUPPORTED; FutureInteractionContribution=SUPPORTED; PaperUsableReliability=YES; RecommendedFinalVariant=R2. Stage5A prior NOT SUPPORTED is unchanged.

【Current Map Input】

Current input is lane and lane_connector centerline geometry at unchanged 2 m resolution and per-actor 50 m radius. is_intersections / turn_directions / traffic_controls are all zero on 15,049,131 segment occurrences. Actual source lines, hashes and stored tensor checks are in [existing map feature audit](../01_data_audit/stage7a_existing_map_feature_audit.md).

Official TRAIN700 has 16,930 candidate windows (32 empty-supervision); VAL150 has 3,619 (16 empty-supervision). Full target actor-windows: TRAIN=290,085, VAL=54,990. All candidate windows, including empty ones, contribute map-segment counts.

No-current-actor null graphs: TRAIN=22, VAL=12; each original index confirms zero actors/targets and therefore no stored lane segments. All-null scenes: []. Their absent stored map location is explicitly NOT AVAILABLE, rather than inferred from incomplete logs. All nonempty scene graphs are checked against their actual recorded location.

Raw trainval mount absent; audit reuses byte-frozen official trainval shards and metadata SQLite, verifies each map segment against actual regional Map Expansion JSON. The shared map install contains four complete regional maps; it is not restricted to the mini scenes. No raw scene/log/annotation read from the absent trainval mount is claimed.

【nuScenes Map Schema】

| location | layer | record_count |
| --- | --- | --- |
| boston-seaport | lane | 1215 |
| boston-seaport | lane_connector | 1631 |
| boston-seaport | stop_line | 775 |
| boston-seaport | traffic_light | 307 |
| boston-seaport | ped_crossing | 340 |
| boston-seaport | walkway | 301 |
| boston-seaport | road_segment | 928 |
| boston-seaport | road_block | 969 |
| boston-seaport | lane_divider | 671 |
| boston-seaport | road_divider | 377 |
| boston-seaport | carpark_area | 275 |
| singapore-hollandvillage | lane | 601 |
| singapore-hollandvillage | lane_connector | 719 |
| singapore-hollandvillage | stop_line | 300 |
| singapore-hollandvillage | traffic_light | 119 |
| singapore-hollandvillage | ped_crossing | 28 |
| singapore-hollandvillage | walkway | 498 |
| singapore-hollandvillage | road_segment | 167 |
| singapore-hollandvillage | road_block | 387 |
| singapore-hollandvillage | lane_divider | 220 |
| singapore-hollandvillage | road_divider | 107 |
| singapore-hollandvillage | carpark_area | 0 |
| singapore-onenorth | lane | 936 |
| singapore-onenorth | lane_connector | 1066 |
| singapore-onenorth | stop_line | 451 |
| singapore-onenorth | traffic_light | 127 |
| singapore-onenorth | ped_crossing | 120 |
| singapore-onenorth | walkway | 838 |
| singapore-onenorth | road_segment | 783 |
| singapore-onenorth | road_block | 645 |
| singapore-onenorth | lane_divider | 357 |
| singapore-onenorth | road_divider | 152 |
| singapore-onenorth | carpark_area | 39 |
| singapore-queenstown | lane | 910 |
| singapore-queenstown | lane_connector | 1173 |
| singapore-queenstown | stop_line | 437 |
| singapore-queenstown | traffic_light | 81 |
| singapore-queenstown | ped_crossing | 75 |
| singapore-queenstown | walkway | 457 |
| singapore-queenstown | road_segment | 260 |
| singapore-queenstown | road_block | 676 |
| singapore-queenstown | lane_divider | 257 |
| singapore-queenstown | road_divider | 172 |
| singapore-queenstown | carpark_area | 40 |

The raw/API field-by-field inventories are [schema JSON](../01_data_audit/stage7a_map_schema_audit.json) and [schema Markdown](../01_data_audit/stage7a_map_schema_audit.md). Raw lane_connector has only token and polygon_token; source arcline_path_3 and connectivity separately provide directed geometry/topology. Raw lane_type is a road lane category, not left/straight/right. No raw turn_type, lane-token control annotation or per-frame current/future signal state exists. API cue shortcuts derive from stop-line references and do not provide a lane-control label.

【Turn Semantics】

All 4,589 connectors have finite original centerlines and analytic entrance/exit tangents from the first start_pose[2] / last end_pose[2] in arcline_path_3. delta_theta=wrap(theta_out−theta_in). We first computed the full real histogram, then examined coarse 2 m chords, independent 0.2 m chords, connectivity endpoint gaps and seeded random plots before adopting audit candidates. Original map inputs retain the original 2 m geometry.

| Interval_deg | Connectors |
| --- | --- |
| [-180, -135) | 16 |
| [-135, -90) | 454 |
| [-90, -60) | 420 |
| [-60, -30) | 95 |
| [-30, -15) | 84 |
| [-15, 15) | 2357 |
| [15, 30) | 132 |
| [30, 60) | 156 |
| [60, 90) | 534 |
| [90, 135) | 334 |
| [135, 180] | 7 |

The 2 m chord angle differs from the analytic tangent angle by up to 24.574510 degrees and crosses the simple ±20 degree class boundary in 32 records. Finer 0.2 m chords have maximum difference 8.925806 degrees; three >2 degree short-path cases were inspected. There are two simple fine/analytic class disagreements: one +20.019 degree boundary (fine +19.929), and one U-turn wrap branch cut. Analytic tangents provide the stable continuous angle; classification near ±20 remains an arbitrary audit taxonomy boundary.

Candidate labels: |delta|≤20 = straight; delta>20 = left; delta<−20 = right, except |delta|≥150 = unknown. The 12 near U-turn cases lie outside the intended V1 taxonomy, and assigning them a signed left/right label after wrap can invert physical turning direction. All 12 were visually checked and left unknown; no pseudo turn labels were made. Ordinary lanes also carry unknown (not a connector-turn label). This is a source-geometry decision, not a VAL-error threshold search.

| location | turn_type | connector_count | segment_count | angle_mean | angle_std | angle_min | angle_max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| boston-seaport | left | 395 | 5509 | 85.66131132542733 | 17.192673641634837 | 20.241363367332116 | 135.29166264523937 |
| boston-seaport | straight | 812 | 10508 | 0.07916785211453836 | 4.1023956645930895 | -19.864795495832894 | 17.989806232943543 |
| boston-seaport | right | 421 | 4245 | -84.64160850713355 | 18.658522380611707 | -139.30178100448575 | -20.42193680239209 |
| boston-seaport | unknown | 3 | 45 | -58.542712080894184 | 167.88541119486752 | -179.57392409538002 | 178.8679227621898 |
| singapore-hollandvillage | left | 173 | 1433 | 76.81357927412947 | 23.767356956731028 | 20.019235585021843 | 140.5009382765449 |
| singapore-hollandvillage | straight | 398 | 4210 | -0.3286869524618296 | 5.1546522495140525 | -19.755443758354705 | 19.73573017256433 |
| singapore-hollandvillage | right | 147 | 1799 | -85.0476037620408 | 21.720726949865597 | -142.4560222106679 | -21.127843663287177 |
| singapore-hollandvillage | unknown | 1 | 11 | 161.35152515241984 | 0.0 | 161.35152515241984 | 161.35152515241984 |
| singapore-onenorth | left | 228 | 2188 | 68.60025400049726 | 26.04669437135824 | 20.446043572850265 | 122.3166376697537 |
| singapore-onenorth | straight | 680 | 8389 | 0.3370221139286009 | 6.11065404872484 | -18.301833026999937 | 18.996173949054793 |
| singapore-onenorth | right | 156 | 2400 | -85.33121956431405 | 24.185055753665864 | -128.7463661016138 | -21.15929987543784 |
| singapore-onenorth | unknown | 2 | 20 | 0.04095752105781969 | 179.43471644803833 | -179.3937589269805 | 179.47567396909614 |
| singapore-queenstown | left | 308 | 2622 | 72.37275578559283 | 24.32407448892312 | 20.302328285109404 | 129.61656740778662 |
| singapore-queenstown | straight | 560 | 7024 | 0.48381993876463003 | 7.1120741955398845 | -19.36418300483721 | 19.185824649570183 |
| singapore-queenstown | right | 299 | 3623 | -82.3735002830264 | 24.92991355662298 | -148.34943317289984 | -20.761898195016148 |
| singapore-queenstown | unknown | 6 | 71 | -112.13310157838357 | 129.01167827072356 | -179.30438677849116 | 175.81347392936985 |

Codex directly inspected 20 seeded random left, 20 straight and 20 right connectors, plus 16 U-turn/boundary/short-path cases: no sign inversions in the 60 ordinary random examples. Review is agent visual inspection, not an independent blinded expert annotation. Global +x east/+y north is right-handed; positive CCW is left. Unchanged determinant+1 ego rotations preserve sign. Full token/sample identities and source geometry are saved in the manual-sample JSON files.

[Boston map](../05_figures/stage7a_turn_semantics_boston.png); [Singapore map](../05_figures/stage7a_turn_semantics_singapore.png); [left20](../05_figures/stage7a_turn_manual_left.png); [straight20](../05_figures/stage7a_turn_manual_straight.png); [right20](../05_figures/stage7a_turn_manual_right.png); [boundary cases](../05_figures/stage7a_turn_boundary_cases.png). PNG300dpi, editable SVG/PDF and scripts are retained.

There are 16 raw dangling connectivity links on ordinary lanes in singapore-onenorth; none originate from connectors. RawConnectivitySemantic=PARTIAL. The complete raw defect list is retained without repair. The V1 candidate cache uses known token ownership, actual centerline geometry and valid control/crossing references; raw connectivity strings are not used as features. Readiness here is for that geometric candidate only, not an unrestricted topology-based extension.

【Traffic Control】

| StopLineType | Count |
| --- | --- |
| PED_CROSSING | 570 |
| STOP_SIGN | 387 |
| TRAFFIC_LIGHT | 220 |
| TURN_STOP | 783 |
| YIELD | 3 |

| location | control_type | record_count | associated_lane_count | segment_count | associated_stop_line_records |
| --- | --- | --- | --- | --- | --- |
| boston-seaport | PED_CROSSING | 248 | 463 | 8064 | 248 |
| boston-seaport | STOP_SIGN | 100 | 241 | 2944 | 99 |
| boston-seaport | TRAFFIC_LIGHT | 115 | 462 | 9175 | 115 |
| boston-seaport | TURN_STOP | 312 | 496 | 7814 | 312 |
| singapore-hollandvillage | PED_CROSSING | 79 | 55 | 1809 | 79 |
| singapore-hollandvillage | STOP_SIGN | 67 | 158 | 1458 | 67 |
| singapore-hollandvillage | TRAFFIC_LIGHT | 36 | 119 | 1952 | 35 |
| singapore-hollandvillage | TURN_STOP | 118 | 223 | 2669 | 118 |
| singapore-onenorth | PED_CROSSING | 82 | 148 | 1740 | 82 |
| singapore-onenorth | STOP_SIGN | 108 | 229 | 2559 | 108 |
| singapore-onenorth | TRAFFIC_LIGHT | 43 | 149 | 3119 | 43 |
| singapore-onenorth | TURN_STOP | 215 | 345 | 5164 | 215 |
| singapore-onenorth | YIELD | 3 | 5 | 39 | 3 |
| singapore-queenstown | PED_CROSSING | 161 | 192 | 3200 | 161 |
| singapore-queenstown | STOP_SIGN | 112 | 286 | 2548 | 112 |
| singapore-queenstown | TRAFFIC_LIGHT | 26 | 117 | 2638 | 26 |
| singapore-queenstown | TURN_STOP | 138 | 264 | 2959 | 138 |

Control association means the whole token centerline intersects the actual typed stop-line polygon. There is no direct authoritative lane-control field. We preserve intersected raw types and valid static light token references; no nearest-signal substitution or topology propagation is used. Associated lane counts include connectors and are token unions within each type; segment counts broadcast to all valid 2 m segments and may overlap across types. Some stop lines have no intersecting centerline, documented in the full association table.

Exclusive cache categories: {'none': 5271, 'other_control': 1214, 'stop_sign': 415, 'traffic_light': 590, 'mixed_control': 757, 'yield': 4}. Actual overlaps require mixed_control in addition to the five initial candidate labels; no invented precedence or forced 5D one-hot is used. There are 1766 unique tokens with at least one traffic_light / stop_sign / yield geometry association. YIELD has only three source records and five intersecting tokens (four exclusive yield, one mixed); its dedicated future feature is removed from the recommended set, while the raw audit remains.

The six-way exclusive control one-hot is audit bookkeeping of actual categories only. Recommended future candidates keep separate static light, stop-sign and other-source-stop-line existence; dedicated YIELD is removed, and future control encoding is not frozen. No sparse real YIELD label is relabeled or synthesized.

NearTrafficControl exposure means any typed stop-line-associated token, including TURN_STOP/PED_CROSSING. It does not establish that an actor must obey a particular sign or light. The three regulatory flags remain separately recorded in segment coverage. TrafficControlSemantic=PARTIAL refers to derived spatial associations, sparse YIELD and absence of direct lane authority; at least static TRAFFIC_LIGHT and STOP_SIGN association is available and visually checked.

Holland Village has119/119 and Queenstown81/81 all-zero traffic-light XY poses. These are not usable geolocations. Static lens colors/shapes are descriptors of hardware, not the active light. No red/yellow/green labels, current_signal_state or future_signal_state are generated. The source record’s static type/references and actual stop-line polygon are used. Ten traffic-light and ten stop/yield-associated observed tokens were visually inspected; static token references are all valid.

【Pedestrian Crossing】

Actual polygon count=563. All crossings, stop lines and lane/connector polygons checked are nonempty, valid and finite. No geometry repair was applied. We audit centerline intersects and distance≤2 m/≤5 m as alternatives; intersections are the primary conservative relation, selected before reading VAL errors.

| location | layer | definition | token_count | associated_token_count | token_coverage_rate | broadcast_segment_coverage_rate | direct_segment_coverage_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| boston-seaport | lane | intersects | 1215 | 169 | 0.1390946502057613 | 0.18238117355627806 | 0.0191980182690819 |
| boston-seaport | lane | within2m | 1215 | 329 | 0.2707818930041152 | 0.31945089539144345 | 0.04076998503380296 |
| boston-seaport | lane | within5m | 1215 | 568 | 0.4674897119341564 | 0.5618516798265986 | 0.09258399132992723 |
| boston-seaport | lane_connector | intersects | 1631 | 792 | 0.4855916615573268 | 0.5730043827251686 | 0.16211158713744028 |
| boston-seaport | lane_connector | within2m | 1631 | 820 | 0.5027590435315757 | 0.5895996454424582 | 0.26813414093662286 |
| boston-seaport | lane_connector | within5m | 1631 | 880 | 0.539546290619252 | 0.6237750529374108 | 0.4039493770620968 |
| singapore-hollandvillage | lane | intersects | 601 | 13 | 0.021630615640599003 | 0.012526096033402923 | 0.0020876826722338203 |
| singapore-hollandvillage | lane | within2m | 601 | 49 | 0.08153078202995008 | 0.06376130828114127 | 0.006784968684759917 |
| singapore-hollandvillage | lane | within5m | 601 | 84 | 0.13976705490848584 | 0.1372651356993737 | 0.016788448155880306 |
| singapore-hollandvillage | lane_connector | intersects | 719 | 99 | 0.1376912378303199 | 0.1858312089091641 | 0.050046960955320005 |
| singapore-hollandvillage | lane_connector | within2m | 719 | 102 | 0.14186369958275383 | 0.18851469207030727 | 0.08184623641486649 |
| singapore-hollandvillage | lane_connector | within5m | 719 | 112 | 0.15577190542420027 | 0.19763853481819402 | 0.12491614115121427 |
| singapore-onenorth | lane | intersects | 936 | 46 | 0.049145299145299144 | 0.04018516033811888 | 0.008251710720515228 |
| singapore-onenorth | lane | within2m | 936 | 115 | 0.12286324786324786 | 0.10069770562189723 | 0.01851603381188783 |
| singapore-onenorth | lane | within5m | 936 | 194 | 0.20726495726495728 | 0.20676237756608076 | 0.04320407889440494 |
| singapore-onenorth | lane_connector | intersects | 1066 | 161 | 0.15103189493433397 | 0.21289528352696777 | 0.06255289682234362 |
| singapore-onenorth | lane_connector | within2m | 1066 | 172 | 0.16135084427767354 | 0.21951219512195122 | 0.10571670385473571 |
| singapore-onenorth | lane_connector | within5m | 1066 | 209 | 0.19606003752345216 | 0.24913441563437716 | 0.1670385473570824 |
| singapore-queenstown | lane | intersects | 910 | 52 | 0.05714285714285714 | 0.09636496706513784 | 0.011466211271041717 |
| singapore-queenstown | lane | within2m | 910 | 72 | 0.07912087912087912 | 0.11628852565666423 | 0.02000487923883874 |
| singapore-queenstown | lane | within5m | 910 | 125 | 0.13736263736263737 | 0.22379442140359437 | 0.037651459705619256 |
| singapore-queenstown | lane_connector | intersects | 1173 | 146 | 0.12446717817561807 | 0.18493253373313343 | 0.03665667166416792 |
| singapore-queenstown | lane_connector | within2m | 1173 | 154 | 0.13128729752770674 | 0.19362818590704647 | 0.06626686656671664 |
| singapore-queenstown | lane_connector | within5m | 1173 | 168 | 0.1432225063938619 | 0.20209895052473764 | 0.10419790104947527 |

Token-level metadata is inherited by every segment owned by that token, as requested. A nearby segment can therefore carry a crossing/control association located farther along its token. Broadcast coverage is not the same as individual segment/object proximity; both are quantified. Actor exposure below means semantic tokens visible within the existing 50 m map edges, not direct actor-to-object distance or actor intent. Ten observed crossing-associated cases were visually inspected.

【Actor Semantic Exposure】

Denominator: official full-horizon target actor-windows. Rates are proportions [0,1], using existing per-actor edges to stored segment starts at <50 m. A turning connector is left/right; unknown U-turns are excluded. Each table gives scene/instance support; overlapping windows are not independent observations.

Official TRAIN700:

| Group | Count | NearConnectorRate | NearTurnConnectorRate | NearTrafficControlRate | NearCrosswalkRate | Scenes | Instances |
| --- | --- | --- | --- | --- | --- | --- | --- |
| overall | 290085 | 0.9783959873830085 | 0.9136253167175138 | 0.971591085371529 | 0.8333833186824552 | 700 | 21743 |
| vehicle | 213745 | 0.9782170343165922 | 0.9048679501274883 | 0.9666705653933425 | 0.8283234695548434 | 699 | 15856 |
| pedestrian | 73202 | 0.9789076801180295 | 0.9370645610775662 | 0.9855878254692495 | 0.8507964263271496 | 603 | 5596 |
| bicycle | 3138 | 0.978648820905035 | 0.9633524537922243 | 0.9802421924792861 | 0.7718291905672403 | 181 | 291 |
| vehicle.moving | 46795 | 0.9806175873490758 | 0.9283897852334652 | 0.9809808740250027 | 0.8279517042419062 | 664 | 3814 |
| Vehicle >5m | 42731 | 0.9799911071587372 | 0.9252299267510706 | 0.9785401699000725 | 0.8142566286770728 | 665 | 3692 |

Official VAL150:

| Group | Count | NearConnectorRate | NearTurnConnectorRate | NearTrafficControlRate | NearCrosswalkRate | Scenes | Instances |
| --- | --- | --- | --- | --- | --- | --- | --- |
| overall | 54990 | 0.9804873613384252 | 0.9129841789416258 | 0.9680669212584107 | 0.7297690489179851 | 150 | 4323 |
| vehicle | 42332 | 0.97550316545403 | 0.9011386185391665 | 0.9609279032410469 | 0.7436927147311726 | 150 | 3144 |
| pedestrian | 12002 | 0.9970004999166806 | 0.9523412764539243 | 0.9915014164305949 | 0.6828861856357273 | 121 | 1108 |
| bicycle | 656 | 1.0 | 0.9573170731707317 | 1.0 | 0.6890243902439024 | 47 | 71 |
| vehicle.moving | 10461 | 0.9905362776025236 | 0.95526240321193 | 0.9776312016059651 | 0.7664659210400535 | 137 | 850 |
| Vehicle >5m | 9744 | 0.9892241379310345 | 0.9563834154351396 | 0.9810139573070608 | 0.7603653530377669 | 137 | 853 |

All current Stage3 segment occurrences (ordinary lanes are unknown turn):

| Split | Total | ConnectorRate | LeftRate | StraightRate | RightRate | UnknownRate | UnknownConnectorRate | TrafficLightRate | StopSignRate | YieldRate | CrosswalkRate | DirectCrosswalkRate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 12334046 | 0.540151544756684 | 0.13095483833934138 | 0.2902049335635687 | 0.1184008880784132 | 0.46043934001867676 | 0.0005908847753608184 | 0.2434773633891101 | 0.08414975913013459 | 6.129375551218149e-05 | 0.3322559361299609 | 0.0860172728397478 |
| val | 2715085 | 0.5533009095479515 | 0.12226836360555932 | 0.3164788579363077 | 0.11401889811921173 | 0.44723388033892125 | 0.0005347898868727867 | 0.2964113462377789 | 0.06927481091752193 | 9.981271304581624e-05 | 0.3419863466521306 | 0.08651368189209546 |
| combined | 15049131 | 0.5425238839372187 | 0.12938767029139422 | 0.29494513669925526 | 0.11761031251571935 | 0.45805688049363114 | 0.000580764430849861 | 0.25302743394286353 | 0.08146609927177854 | 6.824314307583607e-05 | 0.3340114455778211 | 0.08610683234799404 |

【Moving Vehicle Exposure】

| Split | Group | Count | NearConnectorRate | NearTurnConnectorRate | NearTrafficControlRate | NearCrosswalkRate |
| --- | --- | --- | --- | --- | --- | --- |
| train | vehicle.moving | 46795 | 0.9806175873490758 | 0.9283897852334652 | 0.9809808740250027 | 0.8279517042419062 |
| train | Vehicle >5m | 42731 | 0.9799911071587372 | 0.9252299267510706 | 0.9785401699000725 | 0.8142566286770728 |
| val | vehicle.moving | 10461 | 0.9905362776025236 | 0.95526240321193 | 0.9776312016059651 | 0.7664659210400535 |
| val | Vehicle >5m | 9744 | 0.9892241379310345 | 0.9563834154351396 | 0.9810139573070608 | 0.7603653530377669 |

vehicle.moving uses the unchanged t0 annotation attribute. Vehicle >5m uses frozen full-future endpoint displacement only as an offline group. Before exposure counts, we registered an operational sufficient-sample screen: each semantic should expose ≥100 moving actor-windows, ≥10 scenes and ≥50 distinct instances in each split. This checks audit feasibility, not statistical power or a frozen future subgroup definition.

| Split | Semantic | Count | Scenes | Instances | sufficient |
| --- | --- | --- | --- | --- | --- |
| train | NearConnector | 45888 | 662 | 3780 | True |
| train | NearTurnConnector | 43444 | 646 | 3669 | True |
| train | NearTrafficControl | 45905 | 657 | 3781 | True |
| train | NearCrosswalk | 38744 | 557 | 3353 | True |
| val | NearConnector | 10362 | 137 | 845 | True |
| val | NearTurnConnector | 9993 | 137 | 825 | True |
| val | NearTrafficControl | 10227 | 137 | 842 | True |
| val | NearCrosswalk | 8018 | 111 | 708 | True |
| combined | NearConnector | 56250 | 799 | 4625 | True |
| combined | NearTurnConnector | 53437 | 783 | 4494 | True |
| combined | NearTrafficControl | 56132 | 794 | 4623 | True |
| combined | NearCrosswalk | 46762 | 668 | 4061 | True |

Intersection candidates use t0 global actor-to-complete-connector-centerline minimum distance <20 m / <30 m; they are distinct from the 50 m segment-start semantic-token exposure above. TurningVehicle_GT diagnostic requires vehicle endpoint displacement>5 m, first/last 1 s secant displacement>0.5 m, and absolute secant-heading change>20 degrees. These future positions are used only for offline grouping; the map-semantic cache cannot read them. No future group definitions are frozen or training inputs created.

| Split | Candidate | Count | VehicleDenominator | VehicleRate | Scenes | Instances | Stage5A_minFDE | Stage5A_Top1FDE | R2_Top1FDE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | IntersectionVehicle_A | 166480 | 213745 | 0.778872020398138 | 693 | 12086 | — | — | — |
| train | IntersectionVehicle_B | 194064 | 213745 | 0.9079229923506983 | 695 | 14250 | — | — | — |
| train | GT_heading_eligible | 37943 | 213745 | 0.17751526351493602 | 660 | 3409 | — | — | — |
| train | TurningVehicle_GT | 8337 | 213745 | 0.039004421156050434 | 474 | 1195 | — | — | — |
| val | IntersectionVehicle_A | 32085 | 42332 | 0.7579372578663895 | 150 | 2431 | 1.7496305406733366 | 3.6969094330616934 | 3.6507402069804886 |
| val | IntersectionVehicle_B | 37842 | 42332 | 0.8939336672021166 | 150 | 2859 | 1.5417435407183977 | 3.2848896127072953 | 3.24246075372347 |
| val | GT_heading_eligible | 8566 | 42332 | 0.20235283001039403 | 137 | 772 | 5.523143236335091 | 12.065999645583233 | 11.924482426536319 |
| val | TurningVehicle_GT | 1663 | 42332 | 0.03928470188037419 | 106 | 265 | 14.436744033842293 | 18.645765899034846 | 18.53079139949755 |

【Current Error vs Semantic Context】

Only the frozen Stage6A actor CSV is read: minFDE6 is the shared frozen Stage5A geometry; R0 Top1FDE is Stage5A original ranking; R2 Top1FDE is final frozen reranking. Every one of the85,027 full+partial VAL identities, node/type/motion/future mask and exact GT SHA pairs to the frozen shard; primary context tables use only full-horizon moving vehicles. No model evaluation or cache inference is repeated.

| Context | Count | Stage5A_minFDE | Stage5A_Top1FDE | R2_Top1FDE | Scenes | Instances |
| --- | --- | --- | --- | --- | --- | --- |
| NearConnector | 10362 | 4.746694529762865 | 10.610266517937117 | 10.436095339329142 | 137 | 845 |
| NoConnector | 99 | 6.747866350141439 | 12.089210740695096 | 11.910255719766472 | 13 | 16 |
| NearTurnConnector | 9993 | 4.811589002460673 | 10.639757246264596 | 10.474838844369167 | 137 | 825 |
| NoTurnConnector | 468 | 3.784355949737832 | 10.293417436476702 | 9.920663783342665 | 38 | 75 |
| NearTrafficControl | 10227 | 4.817714672910276 | 10.692723811642603 | 10.530524067164533 | 137 | 842 |
| NoTrafficControl | 234 | 2.489399684672682 | 7.632167096257719 | 6.932758920913578 | 14 | 27 |
| NearCrosswalk | 8018 | 4.854644829013775 | 10.653534694614033 | 10.501049468332349 | 111 | 708 |
| NoCrosswalk | 2443 | 4.473493756461054 | 10.52819170723614 | 10.282652716044485 | 73 | 215 |

connector: R2 mean Top1FDE is 10.436095 vs 11.910256 m (near−no=-1.474160); near n=10362, no n=99 / 16 instances / 13 scenes.

turning connector: R2 mean Top1FDE is 10.474839 vs 9.920664 m (near−no=+0.554175); near n=9993, no n=468 / 75 instances / 38 scenes.

typed traffic-control token: R2 mean Top1FDE is 10.530524 vs 6.932759 m (near−no=+3.597765); near n=10227, no n=234 / 27 instances / 14 scenes.

crosswalk token: R2 mean Top1FDE is 10.501049 vs 10.282653 m (near−no=+0.218397); near n=8018, no n=2443 / 215 instances / 73 scenes.

The simple assertion that all connector-associated cases are harder is not supported: their mean minFDE and Top1FDE are lower than the very small NoConnector subset. Turning, control and crossing-associated subsets have higher R2 point-estimate Top1FDE, with different effect sizes. Broad 50 m token exposure is highly saturated (moving connector99.054%, turning95.526%, control97.763%); NoConnector has only99 windows /16 instances /13 scenes, and NoTrafficControl234 /27 /14. These results do not establish concentration, statistically resolved differences, or a semantic model benefit. More selective t0 distance groups are candidates for later review, not definitions frozen here.

All values are descriptive actor-window means in metres. Near/no comparisons describe associated difficulty, not causal effects of intersections, crossings or traffic control. Groups overlap, windows repeat actors, and location/speed/trajectory differences can confound comparisons. No semantic model benefit, confidence interval, subgroup checkpoint selection or VAL threshold tuning is inferred. R2 only changes ranking; minFDE remains identical.

【Feasibility】

Core candidate audit: token parse failures=0; NaN=0; Inf=0; turn/control one-hot sums=1; connector angles finite; all light/crossing/road-block references valid; TRAIN/VAL overlap=0;850 shard SHA unchanged. Maximum map start error=1.52496633e-05 m, vector error=8.42310444e-08 m, roundtrip=3.26849658e-13 m. Every stored 50 m edge exactly recomputes. Prior project/checkpoint/result SHA checks pass. Raw topology is separately PARTIAL and excluded from candidate feature dependencies.

Semantic candidate cache is keyed by (location,lane_token), computed once per token; expensive API calls are batched per location and are never repeated per observed segment. Local derived caches and actor-level audit rows are ignored by Git; paths, schema, counts and SHA are saved. No checkpoint copies exist. 03_training / 04_evaluation / 07_checkpoints contain README only.

TurnSemantic = AVAILABLE

TrafficControlSemantic = PARTIAL

CrosswalkSemantic = AVAILABLE

SemanticMapStage7A = READY

Readiness scope is the validated static geometric candidate only, subject to the recorded partial traffic/topology limitations. Dedicated sparse YIELD, raw dangling topology links and dynamic signals are excluded from the recommended future feature set. Future backbone guidance is Stage3B HiVT + TypeEmbedding + Semantic Map; no Stage4A bias or Stage4F gate is inherited, and Stage5A residual-decoder inheritance remains undecided.

STOP. Stage7A is not trained. Await 大脑AI review. Git tag push remains an external authentication delivery item until local Git credentials are configured.
