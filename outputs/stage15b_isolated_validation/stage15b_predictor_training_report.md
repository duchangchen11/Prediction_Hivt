# Stage15B predictor training

Three fresh Stage5A HiVTMotionAwareDecoder predictors, 650403 parameters each. Original type embedding, motion-conditioned residual decoder and HiVT backbone/K6 are retained. Scheme A exclusively uses InnerTrain378 and InnerDev42; Outer210 and HeadDev70 never contribute optimizer updates or selection. FP32, batch16, original AdamW decay groups/weight_decay0.0001; warmup5000/LR0.001, original NLL max16000/LR0.0001, validation every500, strict full-horizon Overall minFDE6, NLL patience5, final NLL best only. ADE monitoring uses the FDE-best mode as in the original metric function.

| fold | seed | phase | phase_steps | best_global_step | best_overall_FDE | stop_reason | elapsed_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2022 | fixed_scale | 5000 | 4500 | 1.210015 | warmup_fixed5000 | 2501.868718 |
| 1 | 2022 | original_nll | 6500 | 9000 | 1.180323 | patience_5 | 3248.244076 |
| 2 | 2122 | fixed_scale | 5000 | 4000 | 1.301567 | warmup_fixed5000 | 2483.946288 |
| 2 | 2122 | original_nll | 7000 | 9500 | 1.277962 | patience_5 | 3482.655932 |
| 3 | 2222 | fixed_scale | 5000 | 5000 | 1.146686 | warmup_fixed5000 | 2432.778464 |
| 3 | 2222 | original_nll | 8000 | 10500 | 1.125768 | patience_5 | 3872.273726 |

Full monitor curves: `02_training/stage15b_predictor_curves.csv`. Fresh source/config and every-batch source witness are retained locally. No old or tiny weights enter formal fitting.

Fold1 logging interruption after durable step1000: NumPy bool JSON serialization failed. The original fitting sources and protocol remain unchanged; a registered JSON-scalar wrapper repairs serialization. Exact same-checkpoint next-step loss, model, optimizer, batch, cursor and all RNG replay PASS before resuming at1001. The first replay diagnostic omitted frozen deterministic runtime flags and failed; its source/failure is retained, followed by a corrected deterministic check PASS. Four discarded diagnostic updates are separate from formal steps. No tuning or budget extension. See `00_manifest/stage15b_infrastructure_correction_receipt.json` and `03_checks/stage15b_resume_replay_detail.json`.

Every formal optimization batch has an independently verified source witness; see [actual usage audit](01_data_isolation/stage15b_actual_data_usage.md).
