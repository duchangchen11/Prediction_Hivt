# Existing Stage3 map input audit

All 15,049,131 segment occurrences in 20,549 candidate windows / 850 official scenes were checked, including 48 empty-supervision windows. All three stored placeholders are zero.

- `preprocessing/extract_lane_polylines.py:10`: 50 m radius, 2 m resolution.
- `preprocessing/extract_lane_polylines.py:30`: queries lane and lane_connector only.
- `preprocessing/extract_lane_polylines.py:35`: batches missing token centerlines, caches per (location, token).
- `preprocessing/extract_lane_polylines.py:48`: repeated lane_tokens identify segment ownership.
- `preprocessing/extract_lane_polylines.py:52`: actor edges use strict <50 m distance to segment starts.
- `outputs/stage3_multitype_hivt/02_preprocessed/stage3_dataset.py:84`: is_intersections = zero uint8 placeholder.
- `outputs/stage3_multitype_hivt/02_preprocessed/stage3_dataset.py:85`: turn_directions and traffic_controls = zero uint8 placeholders.
- `outputs/stage3_multitype_hivt/02_preprocessed/stage3_dataset.py:92`: source location and ownership tokens retained.
- `preprocessing/coordinates.py:16` and `:21`: unchanged right-handed row-vector transforms; +x ego forward, +y ego left.

| path | sha256 |
| --- | --- |
| preprocessing/extract_lane_polylines.py | 303c3e173d1f0cfc649b9fbaa7d0aee76b349be773ab910ac654f28d02f92b90 |
| preprocessing/coordinates.py | dc110f79c4449f878c23f4316ba8c734938c93f8d7c1c529774f5199cc9a2b82 |
| outputs/stage3_multitype_hivt/02_preprocessed/stage3_dataset.py | d00360d0abd6ff4edfb61c6f332ccf482808b87e210e1655e9cb1d5e3e4cae35 |

Joint global start XY/vector XY matching to original 2 m centerlines resolves repeated arc-junction starts. Maximum position difference 1.52496633e-05 m; vector difference 8.42310444e-08 m; tolerances 1e-4 m. Full edge recomputation exactly matches lane_actor_index in every window.

No old extractor, polyline cache, dataset, coordinate transform, model, loss, decoder or R2 code was modified.
