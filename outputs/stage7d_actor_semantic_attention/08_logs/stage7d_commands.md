All commands run from the existing project workspace and Python environment with `PYTHONDONTWRITEBYTECODE=1`; no environment or runtime upgrades.

```bash
git switch -c stage7d/actor-conditioned-semantic-attention 7df27d2091425c6836a66d6987714b1d9569efb5
PYTHONDONTWRITEBYTECODE=1 /home/lrj/anaconda3/envs/ped_intent/bin/python -u outputs/stage7d_actor_semantic_attention/00_manifest/stage7d_register.py
PYTHONDONTWRITEBYTECODE=1 /home/lrj/anaconda3/envs/ped_intent/bin/python -u outputs/stage7d_actor_semantic_attention/00_manifest/stage7d_preflight.py
PYTHONDONTWRITEBYTECODE=1 /home/lrj/anaconda3/envs/ped_intent/bin/python -u outputs/stage7d_actor_semantic_attention/03_training/stage7d_train.py
```

The registration and preflight were rerun before formal training after fixing two audit/compatibility issues: the audit reads rotation from the model output because the canonical forward clones its input, and the new module name avoids the unchanged optimizer's substring-based decay-partition collision. No forecasting architecture or hyperparameter was changed in response to tiny/VAL results.

Training used published source commit `ad43f821606de97d1878e95f8849da31c42c3c5a`. The immutable preregistration records the training-source hashes. Final evaluation and reporting commands:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/lrj/anaconda3/envs/ped_intent/bin/python -u outputs/stage7d_actor_semantic_attention/02_model_audit/stage7d_scope_audit.py
PYTHONDONTWRITEBYTECODE=1 /home/lrj/anaconda3/envs/ped_intent/bin/python -u outputs/stage7d_actor_semantic_attention/04_evaluation/stage7d_evaluate.py
PYTHONDONTWRITEBYTECODE=1 /home/lrj/anaconda3/envs/ped_intent/bin/python -u outputs/stage7d_actor_semantic_attention/04_evaluation/stage7d_attention_integrity.py
PYTHONDONTWRITEBYTECODE=1 /home/lrj/anaconda3/envs/ped_intent/bin/python -u outputs/stage7d_actor_semantic_attention/04_evaluation/stage7d_capture.py
PYTHONDONTWRITEBYTECODE=1 /home/lrj/anaconda3/envs/ped_intent/bin/python -u outputs/stage7d_actor_semantic_attention/04_evaluation/stage7d_analyze.py
PYTHONDONTWRITEBYTECODE=1 /home/lrj/anaconda3/envs/ped_intent/bin/python -u outputs/stage7d_actor_semantic_attention/04_evaluation/stage7d_efficiency.py
PYTHONDONTWRITEBYTECODE=1 /home/lrj/anaconda3/envs/ped_intent/bin/python -u outputs/stage7d_actor_semantic_attention/05_figures/stage7d_plot.py
PYTHONDONTWRITEBYTECODE=1 /home/lrj/anaconda3/envs/ped_intent/bin/python -u outputs/stage7d_actor_semantic_attention/05_figures/stage7d_cases.py
PYTHONDONTWRITEBYTECODE=1 /home/lrj/anaconda3/envs/ped_intent/bin/python -u outputs/stage7d_actor_semantic_attention/09_reports/stage7d_finalize.py
```

All seven PNG figure bundles were visually inspected; their PDF fonts/selectable text and editable SVG text were verified. The interim eight-row forecasting bootstrap was checked against the final paired scene-bootstrap table and retained as a completed audit. Raw edge captures, actor-level ledgers, checkpoints and verbose logs remain local and are excluded from Git.

Git publication uses the authorized GitHub connector to create identical Git blobs and trees, then advances only `stage7d/actor-conditioned-semantic-attention` without force. After fetching the published commit, its tree and parent must match the native local commit before aligning local HEAD. No merge is performed. Only this stage directory is staged; the five unrelated Stage2C redraw files remain untouched and untracked.
