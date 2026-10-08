# Reproducible Stage8A-0C commands

Project: `/home/lrj/Prediction_Hivt`. Use the existing ped_intent Python; no install or environment upgrade.

```bash
git switch -c stage8a0c/sparse-type-aware-graph-spec dbb743ab79d3d0309419cd051624ff99c7a6a0ee
PYTHONDONTWRITEBYTECODE=1 /home/lrj/anaconda3/envs/ped_intent/bin/python outputs/stage8a_future_scene_compatibility_graph/00c_sparse_type_aware_graph_spec/00_manifest/stage8a0c_register.py
PYTHONDONTWRITEBYTECODE=1 /home/lrj/anaconda3/envs/ped_intent/bin/python outputs/stage8a_future_scene_compatibility_graph/00c_sparse_type_aware_graph_spec/04_coverage/stage8a0c_run_selector.py
PYTHONDONTWRITEBYTECODE=1 /home/lrj/anaconda3/envs/ped_intent/bin/python outputs/stage8a_future_scene_compatibility_graph/00c_sparse_type_aware_graph_spec/03_model_audit/stage8a0c_integrity_audit.py
PYTHONDONTWRITEBYTECODE=1 /home/lrj/anaconda3/envs/ped_intent/bin/python outputs/stage8a_future_scene_compatibility_graph/00c_sparse_type_aware_graph_spec/01_selector/stage8a0c_determinism.py
PYTHONDONTWRITEBYTECODE=1 /home/lrj/anaconda3/envs/ped_intent/bin/python outputs/stage8a_future_scene_compatibility_graph/00c_sparse_type_aware_graph_spec/03_model_audit/stage8a0c_model_audit.py
PYTHONDONTWRITEBYTECODE=1 /home/lrj/anaconda3/envs/ped_intent/bin/python outputs/stage8a_future_scene_compatibility_graph/00c_sparse_type_aware_graph_spec/04_coverage/stage8a0c_summarize.py
PYTHONDONTWRITEBYTECODE=1 /home/lrj/anaconda3/envs/ped_intent/bin/python outputs/stage8a_future_scene_compatibility_graph/00c_sparse_type_aware_graph_spec/07_figures/stage8a0c_plot_comparison.py
PYTHONDONTWRITEBYTECODE=1 /home/lrj/anaconda3/envs/ped_intent/bin/python outputs/stage8a_future_scene_compatibility_graph/00c_sparse_type_aware_graph_spec/09_reports/stage8a0c_finalize.py
git add outputs/stage8a_future_scene_compatibility_graph/00c_sparse_type_aware_graph_spec
git diff --cached --check
git commit -m 'audit: freeze Stage8A-0C sparse type-aware graph specification'
PYTHONDONTWRITEBYTECODE=1 /home/lrj/anaconda3/envs/ped_intent/bin/python outputs/stage8a_future_scene_compatibility_graph/00c_sparse_type_aware_graph_spec/00_manifest/stage8a0c_git_publish.py
```

Selector cache and accounting NPZ files, entity dictionaries, console logs and publication payloads stay local and ignored. Only source, fixed manifests, small reports/tables and figures are committed. GitHub publication may use the authenticated Git-object connector; every blob/tree must match native Git and the remote commit must retain the exact source parent. No merge. STOP after publication; Stage8A-1 training requires another instruction.
