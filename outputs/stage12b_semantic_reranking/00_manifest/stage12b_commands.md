# Stage12B execution record

Work directory: `/home/lrj/Prediction_Hivt`. Base commit: `c2ef16631f6eb67da7d98097967558e7a65925db`.

Branch created with `git switch -c stage12b/motion-conditioned-semantic-reranking c2ef16631f6eb67da7d98097967558e7a65925db`.

Every Python command uses existing `/home/lrj/anaconda3/envs/ped_intent/bin/python`, prefixed with `PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1`. No environment installation or upgrade.

The scripts under `outputs/stage12b_semantic_reranking/` ran in this order:

1. `00_manifest/stage12b_register.py`: registered protocol and historical/source hashes before preflight or training.
2. `01_preflight/stage12b_preflight.py`: semantic/candidate identity, corresponding-fold C/R2 cache replay, normalizers, shuffled semantics and neutral gates. All three OuterTest C outputs match bitwise.
3. `03_tiny/stage12b_tiny.py`: fixed128 pedestrians,300 updates G/S/P, all gates PASS; weights discarded.
4. `04_training/stage12b_train.py`: nine independent runs in fixed Fold1/2/3 G/S/P order; only InnerDev selector; all nine checkpoint identities frozen before residual OOF evaluation.
5. `06_oof_evaluation/stage12b_evaluate.py`: unified four-model OOF predictions and complete actor routing.
6. `07_bootstrap/stage12b_analyze.py`: paired whole-scene2000 bootstrap, controls, switch costs and frozen NOT_SUPPORTED/STOP decision.
7. `09_figures/stage12b_figures.py`: five summary figures and ten BEV cases in PNG/SVG/PDF. A field-dictionary lookup error in reporting code was corrected to use the original semantic manifest; inference, cached inputs, weights and metrics unchanged.
8. `10_reports/stage12b_verify.py`: independent historical/source hashing, normalization reconstruction, exact training-order/selector checks, all nine bitwise OOF replays and all scene intervals. The same reporting-only dictionary lookup was corrected before a complete PASS run.
9. `10_reports/stage12b_report.py`: final report, raw-probability diagnostics plus separately identified fixed-temperature neutral teacher and efficiency scope.

Checkpoints, full actor predictions, source-sized arrays and resumable optimizer states stay local in ignored paths. Small SHA manifests, scripts, aggregate tables, figures and reports are versioned. No historical paths or five unrelated Stage2C drawing files are staged.

Final publication uses GitHub Git objects when native HTTPS credentials are unavailable; every blob and tree SHA is checked. The pushed branch has the exact validated tree and base commit. No merge is performed. STOP after publication; wait for 大脑AI review.
