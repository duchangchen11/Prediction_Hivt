# Stage3A+ execution and audit notes

Use the existing environment from `/home/lrj/Prediction_Hivt`. No new environment, dependency changes or data download.
The original Stage3A best, all 850 scene shards and old experiment files are immutable. This stage uses the frozen local shards, without requiring the original source-volume remount.

```bash
/home/lrj/anaconda3/envs/ped_intent/bin/python outputs/stage3_multitype_hivt/03_no_type_baseline/stage3a_plus_extend_nll.py
/home/lrj/anaconda3/envs/ped_intent/bin/python outputs/stage3_multitype_hivt/03_no_type_baseline/stage3a_plus_finalize_checkpoint.py
/home/lrj/anaconda3/envs/ped_intent/bin/python outputs/stage3_multitype_hivt/03_no_type_baseline/stage3a_plus_finalize_curve.py
/home/lrj/anaconda3/envs/ped_intent/bin/python outputs/stage3_multitype_hivt/01_data_audit/stage3a_plus_interaction_audit.py
/home/lrj/anaconda3/envs/ped_intent/bin/python outputs/stage3_multitype_hivt/04_evaluation/stage3a_plus_motion_evaluate.py
/home/lrj/anaconda3/envs/ped_intent/bin/python outputs/stage3_multitype_hivt/05_figures/stage3a_plus_plot_motion.py
/home/lrj/anaconda3/envs/ped_intent/bin/python outputs/stage3_multitype_hivt/09_reports/stage3a_plus_report.py
```

The CPU interaction audit may run concurrently with GPU continuation; it never calls the model. Motion evaluation, case prediction reproduction and the final report wait for the bounded continuation to finish.
Continuation restores the original best optimizer and CPU/CUDA RNG, plus the exact sampler epoch/next-batch cursor. Recreating the first mid-epoch DataLoader preserves the saved CPU RNG to avoid an extra loader seed draw; the sequence of train batches follows the existing scene/window shuffle. Natural subsequent epoch loader behavior is unchanged.
LR=0.0001, batch=16, original NLL, no scheduler, original architecture and data. No reset or warm-up. Validation at every 500 extra updates saves actor errors; motion subgroup metrics filter the final-best existing saved errors without another inference pass. Six trajectory cases reproduce the same full-VAL batch partition for the exact chosen actors; this is inference only.
Converged means five consecutive validations without strict full-horizon overall minFDE improvement. Stopping at the 3000-update budget is not by itself convergence. The original best remains referenced if no extension checkpoint improves it; any improved checkpoint is saved under a new Stage3A+ name.

The first interaction scan reached all 850 shards, then failed a sanity check that incorrectly assumed every non-null graph had supervised targets. No metrics or figure were published by that failed scan. The corrected check accounts for candidate anchors with no graph and context graphs with zero supervised targets, while verifying all supervised-window counts and keeping the original target mask. The full corrected scan is saved in the interaction log.

Interaction statistics use all frozen candidate graphs at t0, self excluded, Euclidean distance <=10/20/30/50m. Target-neighbor counts are directed; unordered pair and window counts are reported separately, both target-incident and all-context. Spatial proximity does not establish causal interaction. Counts refer to actor-windows/pair-windows, not unique lifelong instances.
Figures use Python/matplotlib with saved JSON/CSV; no fake trajectories, smoothing, interpolation or coordinate edits. Each paired motion figure shows one oracle best-FDE success and one failure; Top1 is displayed independently, and 12 future markers are retained.
The final extension curve renders the unchanged CSV in two simple panels: an expanded-range overall FDE view for the small measured differences and a per-type diagnostic view. The CSV SHA and original checkpoint reference are audited; there is no fitting or smoothing.
If a new best is selected, its inherited parent `moving_ADE/FDE` diagnostic metadata is aligned with that checkpoint's measured full-VAL JSON. A metadata-only finalization verifies every model tensor, optimizer entry, RNG state, cursor and config bit-for-bit unchanged. The new-file SHA is refreshed before case export; the original Stage3A checkpoint is never modified. The audit records any corrected fields and the model-state content SHA.
