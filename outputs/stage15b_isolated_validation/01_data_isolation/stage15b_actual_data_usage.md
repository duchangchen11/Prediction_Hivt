# Stage15B actual optimizer-source audit

Status=PASS. This audit ran after all three predictor training runs completed, before formal head fitting or aggregate OuterTest evaluation.

| Fold | Optimizer steps | Train scenes | Dev scenes | Dev full targets | Selected global step |
| --- | --- | --- | --- | --- | --- |
| 1 | 11500 | 378 | 42 | 18327 | 9000 |
| 2 | 12000 | 378 | 42 | 19084 | 9500 |
| 3 | 13000 | 378 | 42 | 17484 | 10500 |

Every formal optimizer-step scene/sample pair was reconstructed against the original TRAIN index. Global step IDs are contiguous; warm-up binds the first 5,000 append-log lines, while NLL binds the complete witness. Actual training uses only InnerTrain; each 500-step validation uses all corresponding InnerDev full-horizon targets. OuterTest, HeadDev70 and official VAL contribute zero optimization batches. Diagnostic replay updates are excluded from these formal counts.

Evidence: [full audit](stage15b_actual_data_usage.json). Run `python3 outputs/stage15b_isolated_validation/00_manifest/stage15b_data_usage_audit.py` from the project root to reproduce. Large source witnesses remain local under each predictor checkpoint folder.
