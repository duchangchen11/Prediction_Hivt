# Actual nuScenes Map Expansion schema audit

Both original regional JSON and the installed NuScenesMap API were read. API shortcuts are reported separately from raw fields. Static map version is 1.3 in all four locations.

## boston-seaport

Source: `/home/lrj/datasets/nuscenes-mini/expansion/boston-seaport.json`. SHA256: `fc33712481efd763be39ce7edca0ec924d2d1fe20e1c89d52e21bf43d86e6b59`.

### lane (1215 records)

| field | present | non_null | types |
| --- | --- | --- | --- |
| from_edge_line_token | 1215 | 1215 | string |
| lane_type | 1215 | 1215 | string |
| left_lane_divider_segments | 1215 | 1215 | list |
| polygon_token | 1215 | 1215 | string |
| right_lane_divider_segments | 1215 | 1215 | list |
| to_edge_line_token | 1215 | 1215 | string |
| token | 1215 | 1215 | string |

API-only shortcut fields: exterior_node_tokens, holes, left_lane_divider_segment_nodes, right_lane_divider_segment_nodes.

### lane_connector (1631 records)

| field | present | non_null | types |
| --- | --- | --- | --- |
| polygon_token | 1631 | 1631 | string |
| token | 1631 | 1631 | string |

API-only shortcut fields: none.

### stop_line (775 records)

| field | present | non_null | types |
| --- | --- | --- | --- |
| ped_crossing_tokens | 775 | 775 | list |
| polygon_token | 775 | 775 | string |
| road_block_token | 775 | 775 | string |
| stop_line_type | 775 | 775 | string |
| token | 775 | 775 | string |
| traffic_light_tokens | 775 | 775 | list |

API-only shortcut fields: cue, exterior_node_tokens, holes.

### traffic_light (307 records)

| field | present | non_null | types |
| --- | --- | --- | --- |
| from_road_block_token | 307 | 307 | string |
| items | 307 | 307 | list |
| line_token | 307 | 307 | string |
| pose | 307 | 307 | object |
| token | 307 | 307 | string |
| traffic_light_type | 307 | 307 | string |

API-only shortcut fields: node_tokens.

### ped_crossing (340 records)

| field | present | non_null | types |
| --- | --- | --- | --- |
| polygon_token | 340 | 340 | string |
| road_segment_token | 340 | 340 | string |
| token | 340 | 340 | string |

API-only shortcut fields: exterior_node_tokens, holes.

NOT AVAILABLE: lane.turn_type, lane_connector.turn_type, stop_line.lane_token, traffic_light.lane_token, traffic_light.current_signal_state, traffic_light.future_signal_state.

Traffic-light all-zero XY poses: 0. items[].color and shape describe static light lenses, not observations of signal state.

### arcline_path_3 (2846 dictionary keys)

| field | present | non_null | types |
| --- | --- | --- | --- |
| end_pose | 3819 | 3819 | list |
| radius | 3819 | 3819 | number |
| segment_length | 3819 | 3819 | list |
| shape | 3819 | 3819 | string |
| start_pose | 3819 | 3819 | list |

### connectivity (2846 dictionary keys)

| field | present | non_null | types |
| --- | --- | --- | --- |
| incoming | 2846 | 2846 | list |
| outgoing | 2846 | 2846 | list |

## singapore-hollandvillage

Source: `/home/lrj/datasets/nuscenes-mini/expansion/singapore-hollandvillage.json`. SHA256: `0c9209c277686e367b06ab5bbe2547ba357f7f790365cb4927f44ae0f78266b2`.

### lane (601 records)

| field | present | non_null | types |
| --- | --- | --- | --- |
| from_edge_line_token | 601 | 601 | string |
| lane_type | 601 | 601 | string |
| left_lane_divider_segments | 601 | 601 | list |
| polygon_token | 601 | 601 | string |
| right_lane_divider_segments | 601 | 601 | list |
| to_edge_line_token | 601 | 601 | string |
| token | 601 | 601 | string |

API-only shortcut fields: exterior_node_tokens, holes, left_lane_divider_segment_nodes, right_lane_divider_segment_nodes.

### lane_connector (719 records)

| field | present | non_null | types |
| --- | --- | --- | --- |
| polygon_token | 719 | 719 | string |
| token | 719 | 719 | string |

API-only shortcut fields: none.

### stop_line (300 records)

| field | present | non_null | types |
| --- | --- | --- | --- |
| ped_crossing_tokens | 300 | 300 | list |
| polygon_token | 300 | 300 | string |
| road_block_token | 300 | 182 | null, string |
| stop_line_type | 300 | 300 | string |
| token | 300 | 300 | string |
| traffic_light_tokens | 300 | 300 | list |

API-only shortcut fields: cue, exterior_node_tokens, holes.

### traffic_light (119 records)

| field | present | non_null | types |
| --- | --- | --- | --- |
| from_road_block_token | 119 | 119 | string |
| items | 119 | 119 | list |
| line_token | 119 | 119 | string |
| pose | 119 | 119 | object |
| token | 119 | 119 | string |
| traffic_light_type | 119 | 119 | string |

API-only shortcut fields: node_tokens.

### ped_crossing (28 records)

| field | present | non_null | types |
| --- | --- | --- | --- |
| polygon_token | 28 | 28 | string |
| road_segment_token | 28 | 0 | null |
| token | 28 | 28 | string |

API-only shortcut fields: exterior_node_tokens, holes.

NOT AVAILABLE: lane.turn_type, lane_connector.turn_type, stop_line.lane_token, traffic_light.lane_token, traffic_light.current_signal_state, traffic_light.future_signal_state.

Traffic-light all-zero XY poses: 119. items[].color and shape describe static light lenses, not observations of signal state.

### arcline_path_3 (1320 dictionary keys)

| field | present | non_null | types |
| --- | --- | --- | --- |
| end_pose | 1689 | 1689 | list |
| radius | 1689 | 1689 | number |
| segment_length | 1689 | 1689 | list |
| shape | 1689 | 1689 | string |
| start_pose | 1689 | 1689 | list |

### connectivity (1320 dictionary keys)

| field | present | non_null | types |
| --- | --- | --- | --- |
| incoming | 1320 | 1320 | list |
| outgoing | 1320 | 1320 | list |

## singapore-onenorth

Source: `/home/lrj/datasets/nuscenes-mini/expansion/singapore-onenorth.json`. SHA256: `17939122b940d7faff50fa7ff623ded7f739549a781c32dff1ecd1ac7bebca02`.

### lane (936 records)

| field | present | non_null | types |
| --- | --- | --- | --- |
| from_edge_line_token | 936 | 936 | string |
| lane_type | 936 | 936 | string |
| left_lane_divider_segments | 936 | 936 | list |
| polygon_token | 936 | 936 | string |
| right_lane_divider_segments | 936 | 936 | list |
| to_edge_line_token | 936 | 936 | string |
| token | 936 | 936 | string |

API-only shortcut fields: exterior_node_tokens, holes, left_lane_divider_segment_nodes, right_lane_divider_segment_nodes.

### lane_connector (1066 records)

| field | present | non_null | types |
| --- | --- | --- | --- |
| polygon_token | 1066 | 1066 | string |
| token | 1066 | 1066 | string |

API-only shortcut fields: none.

### stop_line (451 records)

| field | present | non_null | types |
| --- | --- | --- | --- |
| ped_crossing_tokens | 451 | 451 | list |
| polygon_token | 451 | 451 | string |
| road_block_token | 451 | 451 | string |
| stop_line_type | 451 | 451 | string |
| token | 451 | 451 | string |
| traffic_light_tokens | 451 | 451 | list |

API-only shortcut fields: cue, exterior_node_tokens, holes.

### traffic_light (127 records)

| field | present | non_null | types |
| --- | --- | --- | --- |
| from_road_block_token | 127 | 127 | string |
| items | 127 | 127 | list |
| line_token | 127 | 127 | string |
| pose | 127 | 127 | object |
| token | 127 | 127 | string |
| traffic_light_type | 127 | 127 | string |

API-only shortcut fields: node_tokens.

### ped_crossing (120 records)

| field | present | non_null | types |
| --- | --- | --- | --- |
| polygon_token | 120 | 120 | string |
| road_segment_token | 120 | 120 | string |
| token | 120 | 120 | string |

API-only shortcut fields: exterior_node_tokens, holes.

NOT AVAILABLE: lane.turn_type, lane_connector.turn_type, stop_line.lane_token, traffic_light.lane_token, traffic_light.current_signal_state, traffic_light.future_signal_state.

Traffic-light all-zero XY poses: 0. items[].color and shape describe static light lenses, not observations of signal state.

### arcline_path_3 (2002 dictionary keys)

| field | present | non_null | types |
| --- | --- | --- | --- |
| end_pose | 2595 | 2595 | list |
| radius | 2595 | 2595 | number |
| segment_length | 2595 | 2595 | list |
| shape | 2595 | 2595 | string |
| start_pose | 2595 | 2595 | list |

### connectivity (2002 dictionary keys)

| field | present | non_null | types |
| --- | --- | --- | --- |
| incoming | 2002 | 2002 | list |
| outgoing | 2002 | 2002 | list |

## singapore-queenstown

Source: `/home/lrj/datasets/nuscenes-mini/expansion/singapore-queenstown.json`. SHA256: `c7e75f5a5643c4831aae1c99a0fdcc61e9ba3c01f98d922f0fafa34ea1062927`.

### lane (910 records)

| field | present | non_null | types |
| --- | --- | --- | --- |
| from_edge_line_token | 910 | 910 | string |
| lane_type | 910 | 910 | string |
| left_lane_divider_segments | 910 | 910 | list |
| polygon_token | 910 | 910 | string |
| right_lane_divider_segments | 910 | 910 | list |
| to_edge_line_token | 910 | 910 | string |
| token | 910 | 910 | string |

API-only shortcut fields: exterior_node_tokens, holes, left_lane_divider_segment_nodes, right_lane_divider_segment_nodes.

### lane_connector (1173 records)

| field | present | non_null | types |
| --- | --- | --- | --- |
| polygon_token | 1173 | 1173 | string |
| token | 1173 | 1173 | string |

API-only shortcut fields: none.

### stop_line (437 records)

| field | present | non_null | types |
| --- | --- | --- | --- |
| ped_crossing_tokens | 437 | 437 | list |
| polygon_token | 437 | 437 | string |
| road_block_token | 437 | 279 | null, string |
| stop_line_type | 437 | 437 | string |
| token | 437 | 437 | string |
| traffic_light_tokens | 437 | 437 | list |

API-only shortcut fields: cue, exterior_node_tokens, holes.

### traffic_light (81 records)

| field | present | non_null | types |
| --- | --- | --- | --- |
| from_road_block_token | 81 | 81 | string |
| items | 81 | 81 | list |
| line_token | 81 | 81 | string |
| pose | 81 | 81 | object |
| token | 81 | 81 | string |
| traffic_light_type | 81 | 81 | string |

API-only shortcut fields: node_tokens.

### ped_crossing (75 records)

| field | present | non_null | types |
| --- | --- | --- | --- |
| polygon_token | 75 | 75 | string |
| road_segment_token | 75 | 0 | null |
| token | 75 | 75 | string |

API-only shortcut fields: exterior_node_tokens, holes.

NOT AVAILABLE: lane.turn_type, lane_connector.turn_type, stop_line.lane_token, traffic_light.lane_token, traffic_light.current_signal_state, traffic_light.future_signal_state.

Traffic-light all-zero XY poses: 81. items[].color and shape describe static light lenses, not observations of signal state.

### arcline_path_3 (2083 dictionary keys)

| field | present | non_null | types |
| --- | --- | --- | --- |
| end_pose | 2600 | 2600 | list |
| radius | 2600 | 2600 | number |
| segment_length | 2600 | 2600 | list |
| shape | 2600 | 2600 | string |
| start_pose | 2600 | 2600 | list |

### connectivity (2083 dictionary keys)

| field | present | non_null | types |
| --- | --- | --- | --- |
| incoming | 2083 | 2083 | list |
| outgoing | 2083 | 2083 | list |

## Limits

No direct raw lane control label or turn_type exists. Geometric static associations are derived and labeled as such. 16 dangling connectivity references occur on 16 ordinary lanes (8 absent targets) in singapore-onenorth; connector references are complete. No repairs or substitutes are made. Candidate features do not consume raw connectivity links.

The external original trainval mount is absent. Complete original regional map JSON is in the shared expansion install under the existing local mini dataroot view. Compatibility with trainval is established by all 850 frozen official scene shards and every stored segment start/vector, not by claiming a new read of absent raw annotations/logs.
