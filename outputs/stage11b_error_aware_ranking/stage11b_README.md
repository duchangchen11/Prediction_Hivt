# Stage11B Controlled Ranking Objectives

All new code and artifacts are under this directory. The exact user request is
in `00_manifest/stage11b_requirements.txt`; the immutable pre-training protocol
and source/history hashes are in `00_manifest/`.

The experiment uses the existing `ped_intent` environment and local frozen
Stage5A/Stage8/Stage11A caches. Data, model weights, NumPy arrays, full actor
records and runtime logs remain local and are excluded from Git.

Execution order from `/home/lrj/Prediction_Hivt`:

1. `00_manifest/stage11b_register.py`: register protocol and historical hashes.
2. `01_preflight/stage11b_prepare.py`: verify fixed candidates, reconstruct raw
   observable R2 features, split scenes, fit fold-only normalization and poison GT.
3. `01_preflight/stage11b_tiny.py`: gate all three objectives before formal fitting.
4. `03_training/stage11b_train.py`: Fold1 R2/A/B/C, Fold2, Fold3; freeze all 12 heads.
5. `09_reports/stage11b_final_audit.py --training-only`: independently verify
   initialization, controls, coverage, saved selection and isolation before OOF.
6. `05_oof_evaluation/stage11b_evaluate.py`: score only each model's OuterTest.
7. `06_bootstrap/stage11b_analyze.py`: registered paired scene bootstrap and costs.
8. `09_reports/stage11b_final_audit.py`: recompute OOF metrics and verify identity.
9. `09_reports/stage11b_report.py`: final report and PNG/SVG/PDF figures.

Run scripts with `PYTHONDONTWRITEBYTECODE=1 /home/lrj/anaconda3/envs/ped_intent/bin/python`.
Registration, preparation and
training preserve existing artifacts rather than silently overwriting frozen
experiments. Partial optimizer state and carry occurrences are saved locally;
resuming a partial run requires preserving those states explicitly.

The final review artifact is `09_reports/stage11b_final_report.md`.
This is ranking OOF conditional on a historically trained frozen predictor.
After Stage11B, stop and wait for review; no official VAL/test or further stage
is authorized by this experiment.
