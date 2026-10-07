【Scope】

Stage8A-0 only. All new files and this execution plan live under outputs/stage8a_future_scene_compatibility_graph/. No head fitting, tiny overfit, optimizer step, new checkpoint or official ranking comparison was run. G1/G2/G3 here are feasibility prototypes only. Frozen Stage7A and Stage7D NOT_SUPPORTED decisions remain unchanged. Stage8A-1 requires separate review and authorization even if readiness passes.

【Frozen Candidate Predictor】

Stage5A checkpoint SHA256=88fe3feb59b7e830ec1e40aa917484386e0d2799d0f6116f410adda2d97fdce7. Stage6A R2 checkpoint SHA256=e3de457d635cd30c629ce316d73a87e419a3ded8abbe0e1f2601f1071527e314; its historical Overall Top1FDE=2.645563 is retained as the future strong baseline, not a new Stage8A measurement. Official TRAIN700/VAL150, K6/Tf12. Candidate cache is referenced, never regenerated/overwritten. HeadTrain630/HeadDev70 split SHA256=55d8a856ce71c75b7af359e8dc66232907ae54b9a14e43c9a5f7f6148ce7d208; future head checkpoint selection must use HeadDev ranking loss as the new requirements specify, not the old R2 Top1FDE selection and never official VAL.

【Candidate Identity】

PASS. Seed2022 random100 TRAIN +100 VAL windows replayed in original batch16 membership. All actors in these windows checked: 5316. Maximum raw/candidate/logit/probability/rotation differences are exactly0. Identities, actor types, history and masks are bitwise identical. Predictor eval()/requires_grad=False, gradient count0, state unchanged. Entire1283 source cache batches were SHA-verified during coverage construction. R0/R2/G1/G2/G3 use references to the same cached geometry; graph head outputs only logits/probabilities. Trained G1/G2/G3 oracle/ranking metrics are not yet evaluated.

【Map Coordinates】

```json
{
  "status": "PASS",
  "windows": 200,
  "coordinate_frame": "t0 ego x-forward/y-left meters",
  "origin_yaw_source": "same original scene shard, current LIDAR_TOP ego pose",
  "maxima": {
    "roundtrip_max_m": 2.8421709430404007e-13,
    "map_start_max_m": 8.007587454257712e-06,
    "map_vector_max_m": 5.960446003427933e-08,
    "distance_frame_max_m": 1.3642420526593924e-12
  },
  "source_map_segments_checked": 144914,
  "global_local_full_region_distance_agrees": true,
  "brute_force_retrieval_candidates": 400,
  "centerline_resolution_source": "unaltered Stage7A 2m discretized complete centerline archive",
  "duplicate_vertices": "source unchanged; segment match uses both endpoints",
  "future_GT_used_for_frame": false
}
```

Both endpoints identify original segments even where the unchanged source centerline contains repeated vertices. Whole-region brute-force retrieval and global/local distance agreement verify the coordinate/index mapping. No centerline repair, semantic change or radius tuning occurred.

【Graph Definition】

```json
{
  "node": [
    "type_vehicle",
    "type_pedestrian",
    "type_bicycle",
    "original_logit",
    "original_probability",
    "history_recent_displacement",
    "history_net_displacement",
    "history_path_length",
    "endpoint_x",
    "endpoint_y",
    "candidate_net_displacement",
    "candidate_path_length",
    "candidate_mean_speed",
    "endpoint_heading_sin",
    "endpoint_heading_cos"
  ],
  "interaction_edge": [
    "future_minimum_distance",
    "closest_timestep_over_Tf",
    "future_mean_distance",
    "endpoint_distance",
    "initial_future_distance",
    "relative_endpoint_heading_sin",
    "relative_endpoint_heading_cos",
    "minimum_relative_step_displacement",
    "closing_distance_over_6s",
    "neighbor_original_probability",
    "target_original_probability",
    "target_vehicle",
    "target_pedestrian",
    "target_bicycle",
    "neighbor_vehicle",
    "neighbor_pedestrian",
    "neighbor_bicycle"
  ],
  "map_node": [
    "is_connector",
    "turn_left",
    "turn_straight",
    "turn_right",
    "turn_unknown",
    "traffic_light_controlled",
    "stop_sign_controlled",
    "other_control",
    "crosswalk_intersects",
    "local_tangent_sin",
    "local_tangent_cos",
    "is_lane",
    "is_connector"
  ],
  "map_edge": [
    "minimum_polyline_distance",
    "mean_future_point_distance",
    "endpoint_distance",
    "heading_difference_sin",
    "heading_difference_cos",
    "fraction_points_within2m",
    "fraction_points_within4m",
    "closest_timestep_over_Tf",
    "is_connector",
    "turn_left",
    "turn_straight",
    "turn_right",
    "turn_unknown",
    "traffic_light_controlled",
    "stop_sign_controlled",
    "other_control",
    "crosswalk_intersects"
  ],
  "feature_dimensions": {
    "node": 15,
    "interaction_edge": 17,
    "map_node": 13,
    "map_edge": 17
  },
  "history_motion": "raw meters, valid chronological observed history; bridge missing-history gaps like frozen Stage5A",
  "candidate_path": "t0 current position then12 predicted future positions; mean speed=path/6s",
  "candidate_heading": "last1s secant prediction[-1]-prediction[-3]; atan2(0,0)=0 for stationary secants",
  "minimum_relative_displacement": "minimum norm of target candidate step displacement minus neighbor candidate step displacement over12 steps, including t0->first step",
  "closest_interaction_time": "(first argmin index+1)/12",
  "closing_tendency": "(distance at first future point-distance at endpoint)/6s; descriptive signed tendency",
  "map_geometry": "12 future points only form a continuous predicted polyline; complete frozen centerline, distinct tokens",
  "map_direction": "directed centerline tangent at nearest polyline-centerline point; transformed into t0 ego coordinates",
  "map_closest_time": "fractional nearest polyline segment index+1, divided by12; stationary polylines use first future point1/12",
  "semantics": "frozen9D vector retained in map node and mode-map edge; duplicate connector flag is intentional",
  "GT_is_not_a_feature": true,
  "node_features_not_standardized_or_fitted_in_Stage0": true
}
```

Each target has6 mode nodes and at most8 current-valid nearest neighbor actors with6 modes each: at most54 local mode nodes and288 directed mode-mode relations. Stable current-distance neighbor selection matches Stage6A. Each6x6 neighbor-mode pair remains an independent17D relation before a learned64D message and softmax attention; no pre-aggregation into min/mean/max across neighbors. Map messages use separate retrieved tokens with geometry and frozen semantics. One LayerNorm update of node+interaction+map messages and a zero-final-layer64-32-1 head changes logits only.

【Graph Coverage】

| Split | Population | Group | Actors | Candidates | AtLeast1Within10m | AtLeast1Within10mRate | AtLeast4Within10mRate | FallbackCandidates | FallbackRate | RetrievedAtLeast1IncludingFallbackRate | MeanMapTokensPerMode |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | current_valid | Overall | 459612 | 2757672 | 2141079 | 0.776408 | 0.458336 | 616593 | 0.223592 | 1.000000 | 3.895867 |
| train | current_valid | Vehicle | 336635 | 2019810 | 1511409 | 0.748293 | 0.439801 | 508401 | 0.251707 | 1.000000 | 3.820996 |
| train | current_valid | Pedestrian | 117082 | 702492 | 598471 | 0.851926 | 0.508850 | 104021 | 0.148074 | 1.000000 | 4.094600 |
| train | current_valid | Bicycle | 5895 | 35370 | 31199 | 0.882075 | 0.513486 | 4171 | 0.117925 | 1.000000 | 4.224314 |
| train | current_valid | MovingVehicle | 74485 | 446910 | 437702 | 0.979396 | 0.892347 | 9208 | 0.020604 | 1.000000 | 6.911568 |
| val | current_valid | Overall | 91092 | 546552 | 419524 | 0.767583 | 0.492934 | 127028 | 0.232417 | 1.000000 | 4.063330 |
| val | current_valid | Vehicle | 67308 | 403848 | 296670 | 0.734608 | 0.471083 | 107178 | 0.265392 | 1.000000 | 3.979777 |
| val | current_valid | Pedestrian | 22343 | 134058 | 115665 | 0.862798 | 0.556893 | 18393 | 0.137202 | 1.000000 | 4.327724 |
| val | current_valid | Bicycle | 1441 | 8646 | 7189 | 0.831483 | 0.521860 | 1457 | 0.168517 | 1.000000 | 3.866528 |
| val | current_valid | MovingVehicle | 17003 | 102018 | 100096 | 0.981160 | 0.920298 | 1922 | 0.018840 | 1.000000 | 7.068243 |

Coverage is measured before fallback: at least1 token within10m, at least4 distinct tokens within10m. Nearest-one fallback yields100% nonempty retrieved sets, but is never counted as within-radius coverage. Primary counts include all current-valid context/partial/full actor modes. This matches graph-neighbor inference availability and avoids selecting graph nodes using future labels. Full-horizon ranking-target-only coverage is separately shown below; its future mask is an audit population label, never a graph input.

| Split | Population | Group | Actors | Candidates | AtLeast1Within10m | AtLeast1Within10mRate | AtLeast4Within10mRate | FallbackCandidates | FallbackRate | RetrievedAtLeast1IncludingFallbackRate | MeanMapTokensPerMode |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | full_horizon_ranking_targets | Overall | 290085 | 1740510 | 1423089 | 0.817628 | 0.500384 | 317421 | 0.182372 | 1.000000 | 4.138674 |
| train | full_horizon_ranking_targets | Vehicle | 213745 | 1282470 | 1014605 | 0.791134 | 0.477383 | 267865 | 0.208866 | 1.000000 | 4.032843 |
| train | full_horizon_ranking_targets | Pedestrian | 73202 | 439212 | 391124 | 0.890513 | 0.563514 | 48088 | 0.109487 | 1.000000 | 4.423160 |
| train | full_horizon_ranking_targets | Bicycle | 3138 | 18828 | 17360 | 0.922031 | 0.594487 | 1468 | 0.077969 | 1.000000 | 4.710962 |
| train | full_horizon_ranking_targets | MovingVehicle | 46795 | 280770 | 277118 | 0.986993 | 0.897318 | 3652 | 0.013007 | 1.000000 | 6.946130 |
| val | full_horizon_ranking_targets | Overall | 54990 | 329940 | 260660 | 0.790022 | 0.512648 | 69280 | 0.209978 | 1.000000 | 4.195469 |
| val | full_horizon_ranking_targets | Vehicle | 42332 | 253992 | 192104 | 0.756339 | 0.481417 | 61888 | 0.243661 | 1.000000 | 4.053183 |
| val | full_horizon_ranking_targets | Pedestrian | 12002 | 72012 | 65089 | 0.903863 | 0.614925 | 6923 | 0.096137 | 1.000000 | 4.683928 |
| val | full_horizon_ranking_targets | Bicycle | 656 | 3936 | 3467 | 0.880843 | 0.656758 | 469 | 0.119157 | 1.000000 | 4.440549 |
| val | full_horizon_ranking_targets | MovingVehicle | 10461 | 62766 | 62143 | 0.990074 | 0.932575 | 623 | 0.009926 | 1.000000 | 7.118153 |

Combined TRAIN/VAL:

| Group | Candidates | CoverageWithin10m | AtLeast4TokensWithin10m | FallbackRate | MeanMapTokensPerMode |
| --- | --- | --- | --- | --- | --- |
| Overall | 3304224 | 0.774948 | 0.464058 | 0.225052 | 3.923567 |
| Vehicle | 2423658 | 0.746012 | 0.445013 | 0.253988 | 3.847453 |
| Pedestrian | 836550 | 0.853668 | 0.516549 | 0.146332 | 4.131959 |
| Bicycle | 44016 | 0.872137 | 0.515131 | 0.127863 | 4.154035 |
| MovingVehicle | 548928 | 0.979724 | 0.897542 | 0.020276 | 6.940686 |

【Graph Statistics】

| Split | Population | Group | Targets | MeanNeighborActors | MeanLocalModeNodes | MeanModeModeEdgesPerTarget | MeanMapNodesPerMode | MeanModeMapEdgesPerTarget | ZeroNeighborTargetRate | FallbackRate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | current_valid | Overall | 459612 | 7.501693 | 51.010156 | 270.060938 | 3.895867 | 23.375203 | 0.005624 | 0.223592 |
| train | current_valid | Vehicle | 336635 | 7.433315 | 50.599890 | 267.599341 | 3.820996 | 22.925976 | 0.007005 | 0.251707 |
| train | current_valid | Pedestrian | 117082 | 7.699997 | 52.199980 | 277.199877 | 4.094600 | 24.567602 | 0.001828 | 0.148074 |
| train | current_valid | Bicycle | 5895 | 7.467854 | 50.807125 | 268.842748 | 4.224314 | 25.345886 | 0.002205 | 0.117925 |
| train | current_valid | MovingVehicle | 74485 | 6.911714 | 47.470283 | 248.821696 | 6.911568 | 41.469410 | 0.017990 | 0.020604 |
| train | full_horizon_ranking_targets | Overall | 290085 | 7.581795 | 51.490770 | 272.944620 | 4.138674 | 24.832042 | 0.004326 | 0.182372 |
| train | full_horizon_ranking_targets | Vehicle | 213745 | 7.520218 | 51.121308 | 270.727849 | 4.032843 | 24.197057 | 0.005483 | 0.208866 |
| train | full_horizon_ranking_targets | Pedestrian | 73202 | 7.762124 | 52.572744 | 279.436463 | 4.423160 | 26.538961 | 0.001052 | 0.109487 |
| train | full_horizon_ranking_targets | Bicycle | 3138 | 7.569471 | 51.416826 | 272.500956 | 4.710962 | 28.265774 | 0.001912 | 0.077969 |
| train | full_horizon_ranking_targets | MovingVehicle | 46795 | 7.081483 | 48.488898 | 254.933390 | 6.946130 | 41.676782 | 0.014788 | 0.013007 |
| val | current_valid | Overall | 91092 | 7.397741 | 50.386444 | 266.318667 | 4.063330 | 24.379978 | 0.006433 | 0.232417 |
| val | current_valid | Vehicle | 67308 | 7.303233 | 49.819397 | 262.916384 | 3.979777 | 23.878662 | 0.008112 | 0.265392 |
| val | current_valid | Pedestrian | 22343 | 7.670277 | 52.021662 | 276.129974 | 4.327724 | 25.966343 | 0.001790 | 0.137202 |
| val | current_valid | Bicycle | 1441 | 7.586398 | 51.518390 | 273.110340 | 3.866528 | 23.199167 | 0.000000 | 0.168517 |
| val | current_valid | MovingVehicle | 17003 | 6.793154 | 46.758925 | 244.553549 | 7.068243 | 42.409457 | 0.018173 | 0.018840 |
| val | full_horizon_ranking_targets | Overall | 54990 | 7.474013 | 50.844081 | 269.064484 | 4.195469 | 25.172813 | 0.005601 | 0.209978 |
| val | full_horizon_ranking_targets | Vehicle | 42332 | 7.400524 | 50.403147 | 266.418879 | 4.053183 | 24.319097 | 0.006969 | 0.243661 |
| val | full_horizon_ranking_targets | Pedestrian | 12002 | 7.722546 | 52.335277 | 278.011665 | 4.683928 | 28.103566 | 0.001083 | 0.096137 |
| val | full_horizon_ranking_targets | Bicycle | 656 | 7.669207 | 52.015244 | 276.091463 | 4.440549 | 26.643293 | 0.000000 | 0.119157 |
| val | full_horizon_ranking_targets | MovingVehicle | 10461 | 6.892744 | 47.356467 | 248.138801 | 7.118153 | 42.708919 | 0.016155 | 0.009926 |

Neighbor-count distributions0..8 are in stage8a_neighbor_distribution.csv. Source cache counts and dataset membership agree exactly. Retrieval cache stores token indices, distances, neighbor selectors and current frame metadata; candidate tensors/history/GT are not copied into it. Node/edge features are constructed on demand from the frozen source predictions. Maximum current-valid actors per scene-window=108.

【Semantic Attachment】

PASS. All8251 frozen lane/connector tokens have exact Stage7A semantic_vector labels. On16 real graph windows, 7949 mode-map relations checked bitwise. Turns20°/150°, static control associations and centerline-crosswalk intersection definitions are unchanged. No dynamic signal state, priority label or invented right-of-way is used.

【No-Future-Leakage Audit】

```json
{
  "status": "PASS",
  "observable_fields": [
    "history",
    "history_padding",
    "actor_type",
    "predicted",
    "original_logits",
    "original_probability",
    "scene_token",
    "sample_token",
    "instance_tokens",
    "map_location",
    "origin",
    "yaw"
  ],
  "GT_mask_target_mask_access_raises": true,
  "poisoned_GT_and_inverted_future_masks_leave_all_graph_arrays_bitwise_identical": true,
  "windows": 16,
  "GT_used_for_neighbor_selection": false,
  "GT_used_for_map_retrieval": false,
  "ranking_labels_not_built_in_Stage0": true,
  "full_target_mask_used_only_for_coverage_population_counts": true,
  "graph_predictor_backward_executed": false,
  "predictor_gradient_count": 0
}
```

The ObservableWindow dataclass is an explicit observed/predicted allowlist. Access-protected GT/future-mask/target-mask fields raise if read; changing their contents leaves every node, selector, interaction relation and map relation identical. GT labels are not constructed in this stage. Later ranking supervision/evaluation belongs outside graph construction.

【Neutral Initialization】

PASS. Shared node encoder, LayerNorm and scoring head initialize bitwise identically withseed2022. Interaction/map branches use fixed independent2123/2124 initialization. Final head linear weight/bias exact0. Neutral logit maxdiff0, same-device probability maxdiff0. CPU versus original GPU cached softmax maxdiff=4.47034836e-08, below1e-6; exact predictor replay on the original device has probability maxdiff0. No claim that these untrained graph heads already improve ranking.

【Memory and Parameter Audit】

| Model | Parameters | TargetsPerForward | InputTensorBytes | InputKiBPerTarget | PeakCUDAAllocatedMiB | ForwardIncrementalPeakMiB | TrainingExecuted | Weights |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R2 | 673 | 128 | 61440 | 0.468750 | 33.156250 | 1.093750 | False | frozen trained R2 |
| G1 | 24066 | 128 | 3668992 | 27.992188 | 85.360840 | 49.765625 | False | zero-final-layer untrained prototype |
| G2 | 20674 | 128 | 3668992 | 27.992188 | 44.347656 | 8.765625 | False | zero-final-layer untrained prototype |
| G3 | 37315 | 128 | 3668992 | 27.992188 | 85.413086 | 49.765625 | False | zero-final-layer untrained prototype |

G3 parameters=37315, below preferred50k and mandatory150k pause threshold. Dense padded raw graph inputs=28664 bytes/target. Batch128 prototype forward CUDA peak=85.413086MiB. Device=NVIDIA GeForce RTX 3080. Actual compressed retrieval cache=96.384MiB; candidates remain in the old1.19GB cache. Inference only; autograd training activations and optimizer memory have not been measured. Cache generation includes file loading/spatial lookup and is not presented as real-time predictor or reranker latency.

【Limitations】

Map entities here are road lane/connector centerlines with frozen static semantics. Within10m coverage may be poor for some candidate locations; nearest-one fallback guarantees a nonempty set but does not make a distant relation reliable. Coverage and map-only geometry do not establish semantic usefulness or ranking gains. One fixed graph architecture, no radius/M/hidden/neighbor search. No G1/G2/G3 training, formal ablation, bootstrap, ranked case studies or scientific support decision has been performed. Current readiness is separate from AgentGraph/SemanticGraph/FSCG effectiveness.

【Stage8A-0 Decision】

```json
{
  "Stage8A_0": "NOT_READY",
  "failures": [
    "train Vehicle fallback=25.170734% exceeds fixed5%",
    "val Vehicle fallback=26.539193% exceeds fixed5%"
  ],
  "candidate_identity": "PASS",
  "coordinate_audit": "PASS",
  "semantic_attachment": "PASS",
  "GT_leakage_audit": "PASS",
  "memory_audit": "PASS",
  "G1_G2_G3_training_executed": false,
  "tiny_training_executed": false,
  "Stage8A_1_started": false,
  "test_used": false,
  "map_radius_m": 10.0,
  "map_topM": 8,
  "neighbors_max": 8,
  "neighbor_radius_m": 50.0,
  "thresholds_changed": false,
  "counts": {
    "train": {
      "current_valid_actors": 459612,
      "current_valid_candidates": 2757672,
      "full_target_actors": 290085,
      "full_target_candidates": 1740510,
      "windows": 16898,
      "scenes": 700
    },
    "val": {
      "current_valid_actors": 91092,
      "current_valid_candidates": 546552,
      "full_target_actors": 54990,
      "full_target_candidates": 329940,
      "windows": 3603,
      "scenes": 150
    }
  },
  "combined_map_coverage": [
    {
      "Group": "Overall",
      "Candidates": 3304224,
      "CoverageWithin10m": 0.7749483691178322,
      "AtLeast4TokensWithin10m": 0.46405842945272474,
      "FallbackRate": 0.22505163088216779,
      "MeanMapTokensPerMode": 3.923567227887698
    },
    {
      "Group": "Vehicle",
      "Candidates": 2423658,
      "CoverageWithin10m": 0.7460124324471522,
      "AtLeast4TokensWithin10m": 0.44501328157685616,
      "FallbackRate": 0.2539875675528478,
      "MeanMapTokensPerMode": 3.8474533123072643
    },
    {
      "Group": "Pedestrian",
      "Candidates": 836550,
      "CoverageWithin10m": 0.853668041360349,
      "AtLeast4TokensWithin10m": 0.5165489211643057,
      "FallbackRate": 0.14633195863965096,
      "MeanMapTokensPerMode": 4.131958639650947
    },
    {
      "Group": "Bicycle",
      "Candidates": 44016,
      "CoverageWithin10m": 0.8721374045801527,
      "AtLeast4TokensWithin10m": 0.5151308615049073,
      "FallbackRate": 0.12786259541984732,
      "MeanMapTokensPerMode": 4.154034896401309
    },
    {
      "Group": "MovingVehicle",
      "Candidates": 548928,
      "CoverageWithin10m": 0.9797241168240644,
      "AtLeast4TokensWithin10m": 0.8975421184563367,
      "FallbackRate": 0.020275883175935643,
      "MeanMapTokensPerMode": 6.940686210213362
    }
  ],
  "mean_neighbor_actors_per_target": 7.484498024347018,
  "mean_mode_mode_edges_per_target": 269.4419288764927,
  "mean_mode_map_edges_per_target": 23.541403367326186,
  "estimated_G3_parameters": 37315,
  "graph_cache_bytes": 101065981,
  "estimated_padded_raw_bytes_per_target": 28664,
  "G3_prototype_batch128_peak_CUDA_MiB": 85.4130859375,
  "STOP": true
}
```

The fixed Vehicle fallback gate fails. Stage8A-0=NOT_READY; do not train G1/G2/G3 or start Stage8A-1. No retrieval parameter was adjusted to obtain a pass.

STOP. Push only this stage branch, do not merge main. Await review.
