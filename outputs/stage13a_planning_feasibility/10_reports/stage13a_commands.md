# Stage13A execution record

Project: `/home/lrj/Prediction_Hivt`.
Base: `020bf774e33b3269a463bbc2d03e0c6f9185a1d1`.
Branch: `stage13a/prediction-to-ego-planning-audit`.
Existing interpreter: `/home/lrj/anaconda3/envs/ped_intent/bin/python`.
Every Python invocation uses `PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1`.

Executed in order (paths relative to this stage):

1. `00_manifest/stage13a_register.py`: before any forward, freeze all historical tracked files, preserved untracked files, checkpoints, original splits, normalization, temperatures and local sources; save protocol.
2. `02_ego_trajectory/stage13a_trajectory.py`: register the fixed12 scene pool and48 anchors before any predictor replay.
3. `06_planning_interface/stage13a_demo.py`: fixed48 historical/observation-only forwarding, original identity/frame/history joins, observation/evaluation separation and future-label/placeholder/raw-metadata poison checks.
4. `01_dataset_audit/stage13a_dataset_audit.py`: only after the small preflight passed, run full630 metadata coverage audit; do not forward all630.
5. `08_cases/stage13a_cases.py`:19 cases across fixed12 scenes, each with5 figure families, PNG/PDF/SVG;4 missing-future cases included.
6. `07_evaluation_preflight/stage13a_evaluation_replay.py`: recompute only offline proxies from separately cached observations and labels; recorded-ego pose heading used consistently for predicted-agent/GT-agent/map footprints. CV outputs unchanged.
7. `10_reports/stage13a_finalize.py`: verify local cached interface replay, required gates, numerical units, source/historical hashes, masks, folds, geometry and probabilities; generate report and decision.

Initial engineering corrections, before final acceptance:

- The first numerical preflight rejected list indexing in footprint sizes before any model forward. Convert sizes to NumPy arrays; the corrected numerical audit passes.
- A long partial demonstration was interrupted at map raster generation after16 completed windows. `shapely.prepare` now accelerates predicates on the unchanged map union. Against the prior sample001 cache, polygon WKB and [301,301] raster were byte-for-byte equal (`05_map_alignment/stage13a_map_predicate_acceleration.json`). Then rerun all48 windows to completion; none of the partial-run counts substitute for final evidence.
- Constrain the scene-density SQL query to original HeadTrain630. Re-running selection reproduces the already registered scene/anchor files byte-for-byte; the fixed pool did not change.
- Plot bounds include EvaluationOnly ego GT after inference/case selection so turning trajectories remain visible. This affects display only.

Full stdout logs and per-sample prediction/map arrays remain in ignored local files. Small tables and audit JSONs contain the completed evidence. No training, backward, optimizer, checkpoint creation, VAL/test model selection, new environment or downloads were performed. Historical Stage12B and earlier conclusions remain unchanged.

Publish only this stage's code, configurations, audit tables, report and case figures to its own branch; do not merge. The final publication verifies remote tree equality, single-parent commit ancestry and local/remote synchronization. Then STOP for 大脑AI review; Stage13B remains unauthorized.
