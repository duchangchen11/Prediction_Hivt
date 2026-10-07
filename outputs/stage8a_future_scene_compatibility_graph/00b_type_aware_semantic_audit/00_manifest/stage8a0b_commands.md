# Reproducible commands

Working directory: `/home/lrj/Prediction_Hivt`. Existing Python: `/home/lrj/anaconda3/envs/ped_intent/bin/python`.
Each Python command uses `PYTHONDONTWRITEBYTECODE=1`. No environment was installed or changed.

```bash
git switch -c stage8a0b/type-aware-semantic-entity-audit 232a0ad7cf73674a8c651eb34301356f62cd6f1a
PYTHONDONTWRITEBYTECODE=1 /home/lrj/anaconda3/envs/ped_intent/bin/python outputs/stage8a_future_scene_compatibility_graph/00b_type_aware_semantic_audit/00_manifest/stage8a0b_register.py
PYTHONDONTWRITEBYTECODE=1 /home/lrj/anaconda3/envs/ped_intent/bin/python outputs/stage8a_future_scene_compatibility_graph/00b_type_aware_semantic_audit/01_schema_audit/stage8a0b_source_geometry_audit.py
PYTHONDONTWRITEBYTECODE=1 /home/lrj/anaconda3/envs/ped_intent/bin/python outputs/stage8a_future_scene_compatibility_graph/00b_type_aware_semantic_audit/04_coverage/stage8a0b_run_coverage.py
PYTHONDONTWRITEBYTECODE=1 /home/lrj/anaconda3/envs/ped_intent/bin/python outputs/stage8a_future_scene_compatibility_graph/00b_type_aware_semantic_audit/03_entity_retrieval/stage8a0b_geometry_checks.py
PYTHONDONTWRITEBYTECODE=1 /home/lrj/anaconda3/envs/ped_intent/bin/python outputs/stage8a_future_scene_compatibility_graph/00b_type_aware_semantic_audit/05_graph_estimate/stage8a0b_summarize.py
PYTHONDONTWRITEBYTECODE=1 /home/lrj/anaconda3/envs/ped_intent/bin/python outputs/stage8a_future_scene_compatibility_graph/00b_type_aware_semantic_audit/07_figures/stage8a0b_plot_coverage.py
PYTHONDONTWRITEBYTECODE=1 /home/lrj/anaconda3/envs/ped_intent/bin/python outputs/stage8a_future_scene_compatibility_graph/00b_type_aware_semantic_audit/09_reports/stage8a0b_finalize.py
git add outputs/stage8a_future_scene_compatibility_graph/00b_type_aware_semantic_audit
git diff --cached --check
git commit -m 'audit: finalize Stage8A-0B actor-type-aware semantic entity readiness'
PYTHONDONTWRITEBYTECODE=1 /home/lrj/anaconda3/envs/ped_intent/bin/python outputs/stage8a_future_scene_compatibility_graph/00b_type_aware_semantic_audit/00_manifest/stage8a0b_git_publish.py
```

Console output is saved in `08_logs/` locally. Large retrieval/accounting caches, raw map dictionaries and Git publication payloads are ignored. Only this stage's code, small audit reports, tables and diagnostic figures are versioned. Remote Git objects may be published using the authenticated GitHub connector; native and remote commit trees/parents must match before local SHA alignment. No merge is performed.
