# Stage15B reproducibility commands

Use the existing ped_intent environment and the byte-verified original TRAIN scene shards. No dependency installation or old predictor/head weights. All commands run from `/home/lrj/Prediction_Hivt`.

```bash
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
/home/lrj/anaconda3/envs/ped_intent/bin/python -u outputs/stage15b_isolated_validation/00_manifest/stage15b_infrastructure.py outputs/stage15b_isolated_validation/02_training/stage15b_train.py --fold 1 --seed 2022 --training-scenes outputs/stage15b_isolated_validation/01_data_isolation/stage15b_fold1_InnerTrain.json --development-scenes outputs/stage15b_isolated_validation/01_data_isolation/stage15b_fold1_InnerDev.json --output-dir outputs/stage15b_isolated_validation/04_predictor_checkpoints/fold1 --mode formal
```

Folds2/3 use their original lists and seeds2122/2222. The formal entry recovers only its own exact-source model/optimizer/all RNG/cursor checkpoint. An existing COMPLETE summary verifies its checkpoint and executes no further optimizer update. Never overwrite a frozen experiment; fresh predictor replication can use an explicitly separate Stage15B-local output directory.

The sequential supervisor records each core entry and arguments. Its actual child launch includes the JSON wrapper described in `stage15b_process_command_clarification.json`.

After all three predictors qualify: run `03_checks/stage15b_checkpoint_check.py --fold N`, then `05_candidate_interface/stage15b_cache.py --freeze`. Cache each fold with the same file `--fold N`. Before each fold of head fitting, run `03_checks/stage15b_head_check.py --fold N`, then `06_rank_training/stage15b_heads.py --fold N`. Freeze all18 heads with `06_rank_training/stage15b_heads.py --freeze` before `08_evaluation/stage15b_evaluate.py`, `09_statistics/stage15b_statistics.py`, `09_statistics/stage15b_extra.py`, and `00_manifest/stage15b_finalize.py`. Use the same Python environment and registered wrapper for all entries. The original source, partition, per-role identity/cache SHA and final model SHA manifests are authoritative.

Completed candidate/evaluation caches reject overwrites. Detailed local case trajectory CSVs and all binary checkpoints/caches are intentionally absent from GitHub. Final checkpoint tests discard their diagnostic updates; no diagnostic model replaces a formal predictor or head.
