#!/usr/bin/env bash
# Existing environment and frozen data; execute from the fixed project root.
set -euo pipefail
cd /home/lrj/Prediction_Hivt
export PYTHONDONTWRITEBYTECODE=1
stage4f_python=/home/lrj/anaconda3/envs/ped_intent/bin/python
"$stage4f_python" outputs/stage4f_interaction_necessity_gate/03_training/stage4f_audits_tiny.py
# Registration and code SHA are frozen before formal training.
"$stage4f_python" outputs/stage4f_interaction_necessity_gate/03_training/stage4f_train.py
"$stage4f_python" outputs/stage4f_interaction_necessity_gate/04_evaluation/stage4f_evaluate.py
"$stage4f_python" outputs/stage4f_interaction_necessity_gate/04_evaluation/stage4f_efficiency.py
"$stage4f_python" outputs/stage4f_interaction_necessity_gate/05_figures/stage4f_quantitative.py
"$stage4f_python" outputs/stage4f_interaction_necessity_gate/05_figures/stage4f_cases.py
"$stage4f_python" outputs/stage4f_interaction_necessity_gate/09_reports/stage4f_finalize.py
