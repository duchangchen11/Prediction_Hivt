# Stage7C executed commands

Working directory: `/home/lrj/Prediction_Hivt`. Existing `ped_intent` environment.
No dependency install, environment creation, data download or training command.

```bash
git switch -c stage7c/actor-conditioned-semantic-relevance-audit 77e77b3e1f48a5e68e48d5665308232b97d058f4
```

Commands below each used `PYTHONDONTWRITEBYTECODE=1`, existing Python
`/home/lrj/anaconda3/envs/ped_intent/bin/python`, with stdout/stderr recorded in the
corresponding `08_logs/stage7c_*.log` file.

```bash
PYTHONDONTWRITEBYTECODE=1 /home/lrj/anaconda3/envs/ped_intent/bin/python outputs/stage7c_actor_semantic_relevance_audit/01_attention_capture/stage7c_integrity.py
PYTHONDONTWRITEBYTECODE=1 /home/lrj/anaconda3/envs/ped_intent/bin/python outputs/stage7c_actor_semantic_relevance_audit/01_attention_capture/stage7c_capture.py
PYTHONDONTWRITEBYTECODE=1 /home/lrj/anaconda3/envs/ped_intent/bin/python outputs/stage7c_actor_semantic_relevance_audit/02_relevance_analysis/stage7c_analyze.py
PYTHONDONTWRITEBYTECODE=1 /home/lrj/anaconda3/envs/ped_intent/bin/python outputs/stage7c_actor_semantic_relevance_audit/04_perturbation/stage7c_perturb.py
PYTHONDONTWRITEBYTECODE=1 /home/lrj/anaconda3/envs/ped_intent/bin/python outputs/stage7c_actor_semantic_relevance_audit/02_relevance_analysis/stage7c_association_uncertainty.py
PYTHONDONTWRITEBYTECODE=1 /home/lrj/anaconda3/envs/ped_intent/bin/python outputs/stage7c_actor_semantic_relevance_audit/04_perturbation/stage7c_perturb_tables.py
PYTHONDONTWRITEBYTECODE=1 /home/lrj/anaconda3/envs/ped_intent/bin/python outputs/stage7c_actor_semantic_relevance_audit/05_figures/stage7c_cases.py
PYTHONDONTWRITEBYTECODE=1 /home/lrj/anaconda3/envs/ped_intent/bin/python outputs/stage7c_actor_semantic_relevance_audit/09_reports/stage7c_finalize.py
```

Integrity PASS preceded all official full capture/perturbation runs. The integrity
process uses deterministic CUDA and `CUBLAS_WORKSPACE_CONFIG=:4096:8` for a
controlled output comparison. Complete diagnosis uses the frozen original
evaluation policy and checks reproduction of old actor-window mean errors.
All models run eval/no_grad with unchanged checkpoint state.

The case plotting command was repeated once to separate control/crosswalk marker
symbols from turn-colored geometry and improve trajectory endpoint visibility.
This repeated only four case forwards and plotting; it did not repeat either
official SHUFFLE/CENTERED VAL pass or change case choices/aggregate metrics.

Visual QA: all four PNGs inspected after revision, common axes/color scales and
GT/prediction source checked. PDF fonts embedded as CID TrueType and extracted
text checked; SVG editable text and export dimensions verified. Frozen SHA audit
checked1835 previous files and850 immutable Stage3 scene shards; each of226
local edge archives also hashed. Final protocol PASS.

Git publishes only this Stage7C root. Edge NPZ, raw actor CSV and logs stay local;
small tables, source, case exports, SHA manifests and report are committed.
An existing GitHub connector publishes matching Git objects and a nonforced
branch reference. Public git fetch verifies the remote commit tree/parent against
the scoped local commit; publication proof is local and ignored to avoid a
self-referential commit hash. No merge or Stage7B command.
