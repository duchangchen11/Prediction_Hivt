# Stage7A formal experiment

All new formal code and outputs live in this Stage7A directory. Earlier Stage7A-0 audit files, including their reservation READMEs and pipeline state, describe the completed audit phase and remain unchanged.

The formal authorization and frozen scope are in [stage7a_formal_requirements.txt](00_manifest/stage7a_formal_requirements.txt) and [stage7a_formal_preregistration.json](00_manifest/stage7a_formal_preregistration.json). The audit-only `stage7a_common.py` is preserved; formal training uses `stage7a_formal_common.py`.

- `00_manifest`: isolated model/dataset, preregistration, neutral and gradient audits, preservation manifest.
- `01_data_audit`: bitwise graph identity audit; local frozen full-VAL evaluation membership sidecar.
- `02_semantic_cache`: existing read-only static metadata and centerlines; local large caches excluded from Git.
- `03_training`: one from-scratch Stage7A training run, phase summaries and curves; separate fixed tiny health check.
- `04_evaluation`: fresh baseline/ON/ZERO inference, paired metrics, scientific decision and efficiency analysis.
- `05_figures`: quantitative comparisons and three matched prediction examples with numerical source data.
- `06_tables`: compact aggregate metrics, bootstrap, residual and efficiency tables.
- `07_checkpoints`: local large weights, optimizer and RNG state excluded from Git; compact SHA manifest tracked.
- `08_logs`: tracked commands and local untracked process logs.
- `09_reports`: final formal report and requested summary fields.

Only Stage7A is authorized. No Stage7B execution or merge into main is part of this experiment.
