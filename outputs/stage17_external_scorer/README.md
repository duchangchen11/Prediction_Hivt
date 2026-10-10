# Stage17: external trajectory scoring baseline

Single-component TNT adaptation on **unchanged Stage15B scene-isolated candidates**. This is supplementary internal CV, not a full TNT reproduction or official nuScenes benchmark. All added artifacts belong here; third-party source, candidate/context arrays and model checkpoints remain local and ignored.

## Frozen experiment

[Registration](00_manifest/stage17_registration.json) and [public freeze receipt](00_manifest/stage17_public_registration.json) precede fitting and the first new TNT Outer performance. [Source audit](stage17_baseline_source_audit.md) documents the nonofficial implementation, exact commit, CE/BCE distinction and unresolved redistribution license. [Adapter](stage17_adapter_design.md) records context/coordinates/temperature/routing differences. No HiVT fitting, G-C modification, VAL/HeadDev selection or Outer tuning is authorized.

## Reproduction on the existing local environment

Use the existing `ped_intent` interpreter and existing local source/data paths. Required prerequisites: immutable Stage15B three predictors, original candidate/identity/context caches, original six heads per fold, and Stage16’s hash-verified TNT source cache. The source-fetch command in Stage16 is available for restoring small pinned source files in a clean installation; do not upgrade the source commit, replace the scorer, download new datasets or overwrite scientific results. NuScenes data and candidate caches are not distributed in Git.

The preparation script seals the experiment once on a fresh Stage17 directory. **Do not rerun preparation against this completed freeze or overwrite selected checkpoints.** The runner retains partial runs for diagnosis and does not retry or tune automatically.

```bash
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
PY=/home/lrj/anaconda3/envs/ped_intent/bin/python
ROOT=outputs/stage17_external_scorer
# Executed once before export/fitting:
$PY $ROOT/00_manifest/stage17_prepare.py
# Train/Dev export → actual-data preflight → exactly three scorer fits → freeze → Outer export:
$PY $ROOT/00_manifest/stage17_run_registered.py
# Only after all three selected scorer checkpoints are globally frozen:
$PY $ROOT/06_evaluation/stage17_evaluate.py
$PY $ROOT/07_efficiency/stage17_benchmark.py
$PY $ROOT/08_figures/stage17_figures.py
# Inspect both generated PNGs, then record font/source/export checks and actual visual review:
$PY $ROOT/08_figures/stage17_export_qa.py --visual-review-confirmed
$PY $ROOT/00_manifest/stage17_finalize.py
```

The original primary TNT result scores **all actor types**. A separate table exposes same-fold R2 Bicycle-routing sensitivity; it never changes the primary checkpoint or result. Existing seven-model results are loaded unchanged and checked bitwise. Paired scene bootstrap reuses the original scene identities and verifies original bootstrap draws, with descriptive 95% intervals outside confirmatory family3.

Detailed commands, hashes, model choices, source identities, forward invariance, gradient/state checks, resource measurements and figure source data are retained here. Compute benchmarks include frozen-context capture for TNT and actual graph/R2 preparation for the existing methods. Timings are scoped engineering measurements on fixed InnerDev windows, not deployment throughput.
