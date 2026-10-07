# Stage7A formal commands

Use the existing `ped_intent` environment, `PYTHONDONTWRITEBYTECODE=1`, project directory.

```bash
PYTHONDONTWRITEBYTECODE=1 /home/lrj/anaconda3/envs/ped_intent/bin/python -u outputs/stage7a_semantic_map_hivt/00_manifest/stage7a_formal_register.py
PYTHONDONTWRITEBYTECODE=1 /home/lrj/anaconda3/envs/ped_intent/bin/python -u outputs/stage7a_semantic_map_hivt/00_manifest/stage7a_preflight.py
PYTHONDONTWRITEBYTECODE=1 /home/lrj/anaconda3/envs/ped_intent/bin/python -u outputs/stage7a_semantic_map_hivt/03_training/stage7a_train.py
PYTHONDONTWRITEBYTECODE=1 /home/lrj/anaconda3/envs/ped_intent/bin/python -u outputs/stage7a_semantic_map_hivt/04_evaluation/stage7a_evaluate.py
PYTHONDONTWRITEBYTECODE=1 /home/lrj/anaconda3/envs/ped_intent/bin/python -u outputs/stage7a_semantic_map_hivt/04_evaluation/stage7a_residual_efficiency.py
PYTHONDONTWRITEBYTECODE=1 /home/lrj/anaconda3/envs/ped_intent/bin/python -u outputs/stage7a_semantic_map_hivt/05_figures/stage7a_plot_formal.py
PYTHONDONTWRITEBYTECODE=1 /home/lrj/anaconda3/envs/ped_intent/bin/python -u outputs/stage7a_semantic_map_hivt/09_reports/stage7a_formal_finalize.py
```

Training restores its own phase checkpoint after interruption. A completed run is not repeated.
Neutral CUDA comparison uses deterministic algorithms with `CUBLAS_WORKSPACE_CONFIG=:4096:8` in the audit process. Training keeps Stage3B's existing kernel policy. An initial unrestricted CUDA comparison showed differences of about 2e-6, also observed in Stage3B's own repeated forward; deterministic equality is audited separately.

The former audit-stage files remain read-only. Formal requirements, registration, training and evaluation use separate files. Only the Stage7A ignore file is extended to exclude large formal artifacts. Stage2C redraw files remain unsubmitted. Git publication uses the connected GitHub Git-object API when native push authentication is unavailable; local and remote trees and parents are verified before alignment.
