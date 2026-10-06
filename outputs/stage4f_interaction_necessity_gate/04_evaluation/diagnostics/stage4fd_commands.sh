#!/usr/bin/env bash
# Original audit commands. Capture protects existing raw diagnostic files.
set -euo pipefail
cd /home/lrj/Prediction_Hivt
export PYTHONDONTWRITEBYTECODE=1
STAGE4FD_PYTHON=/home/lrj/anaconda3/envs/ped_intent/bin/python
STAGE4FD_DIAG=outputs/stage4f_interaction_necessity_gate/04_evaluation/diagnostics
"$STAGE4FD_PYTHON" "$STAGE4FD_DIAG/stage4fd_capture.py" > "$STAGE4FD_DIAG/stage4fd_capture.log" 2>&1
"$STAGE4FD_PYTHON" "$STAGE4FD_DIAG/stage4fd_summarize.py" > "$STAGE4FD_DIAG/stage4fd_summarize.log" 2>&1
"$STAGE4FD_PYTHON" outputs/stage4f_interaction_necessity_gate/05_figures/stage4fd_figures.py > "$STAGE4FD_DIAG/stage4fd_figures.log" 2>&1
# Manual visual QA is recorded before running the final report command.
"$STAGE4FD_PYTHON" outputs/stage4f_interaction_necessity_gate/09_reports/stage4fd_report.py > "$STAGE4FD_DIAG/stage4fd_report.log" 2>&1
