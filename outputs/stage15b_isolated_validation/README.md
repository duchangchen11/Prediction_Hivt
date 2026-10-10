# Stage15B Scene-Isolated End-to-End Validation

Authorized formal Scheme A. Each fold uses the original 378 InnerTrain, 42 InnerDev, 210 OuterTest and 70 quarantined HeadDev scenes. Three fresh Stage5A predictors, then six fresh heads per fold. No historical weights.

Protocol and source registration precede fitting. Outer candidate generation is gated by all three predictor freezes; unified Outer evaluation is gated by all 18 head freezes. Raw data, local checkpoints and streamed shared caches remain local.

This is internal scene-isolated CV with historical development exposure, not pristine independent confirmation or an official nuScenes leaderboard evaluation.

See `00_manifest/stage15b_protocol.json`, `stage15b_registration.json`, and `01_data_isolation/stage15b_scene_integrity.json`.
