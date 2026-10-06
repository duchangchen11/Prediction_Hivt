#!/usr/bin/env bash
set -euo pipefail
cd /home/lrj/Prediction_Hivt
export PYTHONDONTWRITEBYTECODE=1
STAGE5A_PYTHON=/home/lrj/anaconda3/envs/ped_intent/bin/python
STAGE5A_ROOT=outputs/stage5a_motion_aware_decoder
# Commands are run sequentially with audit gates in the Python entrypoints.
"$STAGE5A_PYTHON" "$STAGE5A_ROOT/00_manifest/stage5a_register.py"
"$STAGE5A_PYTHON" -u "$STAGE5A_ROOT/03_training/stage5a_audits_tiny.py" > "$STAGE5A_ROOT/08_logs/stage5a_audits_tiny.log" 2>&1
"$STAGE5A_PYTHON" -u "$STAGE5A_ROOT/03_training/stage5a_train.py" > "$STAGE5A_ROOT/08_logs/stage5a_formal_training.log" 2>&1
"$STAGE5A_PYTHON" -u "$STAGE5A_ROOT/04_evaluation/stage5a_evaluate.py" > "$STAGE5A_ROOT/08_logs/stage5a_evaluation.log" 2>&1
"$STAGE5A_PYTHON" -u "$STAGE5A_ROOT/04_evaluation/stage5a_efficiency.py" > "$STAGE5A_ROOT/08_logs/stage5a_efficiency.log" 2>&1
"$STAGE5A_PYTHON" "$STAGE5A_ROOT/05_figures/stage5a_figures.py"
"$STAGE5A_PYTHON" "$STAGE5A_ROOT/05_figures/stage5a_cases.py"
"$STAGE5A_PYTHON" "$STAGE5A_ROOT/09_reports/stage5a_finalize.py"
