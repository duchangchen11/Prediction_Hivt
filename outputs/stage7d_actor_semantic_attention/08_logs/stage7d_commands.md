All commands run from the existing project workspace and Python environment with `PYTHONDONTWRITEBYTECODE=1`; no environment or runtime upgrades.

```bash
git switch -c stage7d/actor-conditioned-semantic-attention 7df27d2091425c6836a66d6987714b1d9569efb5
PYTHONDONTWRITEBYTECODE=1 /home/lrj/anaconda3/envs/ped_intent/bin/python -u outputs/stage7d_actor_semantic_attention/00_manifest/stage7d_register.py
PYTHONDONTWRITEBYTECODE=1 /home/lrj/anaconda3/envs/ped_intent/bin/python -u outputs/stage7d_actor_semantic_attention/00_manifest/stage7d_preflight.py
PYTHONDONTWRITEBYTECODE=1 /home/lrj/anaconda3/envs/ped_intent/bin/python -u outputs/stage7d_actor_semantic_attention/03_training/stage7d_train.py
```

The registration and preflight were rerun before formal training after fixing two audit/compatibility issues: the audit reads rotation from the model output because the canonical forward clones its input, and the new module name avoids the unchanged optimizer's substring-based decay-partition collision. No forecasting architecture or hyperparameter was changed in response to tiny/VAL results.
