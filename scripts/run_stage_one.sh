#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
STAGE_ONE_PYTHON="${STAGE_ONE_PYTHON:-/home/lrj/anaconda3/envs/ped_intent/bin/python}"
mkdir -p outputs/reports outputs/debug outputs/figures
exec > >(tee outputs/reports/stage_one_run.txt) 2>&1
set -x
"$STAGE_ONE_PYTHON" -m preprocessing.inspect_nuscenes
"$STAGE_ONE_PYTHON" -m unittest discover -s tests -v
"$STAGE_ONE_PYTHON" -m preprocessing.build_one_window
"$STAGE_ONE_PYTHON" -m scripts.validate_window
"$STAGE_ONE_PYTHON" -m preprocessing.visualize_window
if [[ "${STAGE_ONE_VISUAL_APPROVED:-0}" != 1 ]]; then
    printf '%s\n' 'Review outputs/figures/one_window.png, then run python -m scripts.evaluate_cv.'
    exit 0
fi
"$STAGE_ONE_PYTHON" -m scripts.evaluate_cv
