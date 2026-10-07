All commands use the existing project and existing `ped_intent` Python environment. No data download, environment creation, PyTorch/CUDA upgrade or historical runtime modification.

```bash
git switch -c stage8a/future-scene-compatibility-graph 35b7811ce2ec0c7dd3b252995b49724e5971b4be
PYTHONDONTWRITEBYTECODE=1 /home/lrj/anaconda3/envs/ped_intent/bin/python -u outputs/stage8a_future_scene_compatibility_graph/00_manifest/stage8a_register.py
PYTHONDONTWRITEBYTECODE=1 /home/lrj/anaconda3/envs/ped_intent/bin/python -u outputs/stage8a_future_scene_compatibility_graph/01_cache_audit/stage8a_identity_audit.py
PYTHONDONTWRITEBYTECODE=1 /home/lrj/anaconda3/envs/ped_intent/bin/python -u outputs/stage8a_future_scene_compatibility_graph/03_model_audit/stage8a_graph_audit.py
PYTHONDONTWRITEBYTECODE=1 /home/lrj/anaconda3/envs/ped_intent/bin/python -u outputs/stage8a_future_scene_compatibility_graph/02_graph_cache/stage8a_build_graph_cache.py
PYTHONDONTWRITEBYTECODE=1 /home/lrj/anaconda3/envs/ped_intent/bin/python -u outputs/stage8a_future_scene_compatibility_graph/03_model_audit/stage8a_memory_audit.py
PYTHONDONTWRITEBYTECODE=1 /home/lrj/anaconda3/envs/ped_intent/bin/python -u outputs/stage8a_future_scene_compatibility_graph/09_reports/stage8a_finalize_stage0.py
```

The coordinate audit initially matched segments by their start alone. A repeated source centerline vertex exposed an ambiguous zero-length segment. The audit was corrected to match both start and end; source geometry was preserved and the full200-window replay reran successfully. Stationary predicted polylines retain their original geometry and use the first future point for closest-time bookkeeping. GEOS can emit floating-point warnings on degenerate geometries; all retained distances/features are explicitly required to be finite and geometry fixtures verify their values.

Map coverage gates and radius10m/M8 remain fixed. Full cache coverage includes all current-valid actor modes, with full-horizon ranking-target-only results separately reported. No normalization fit, ranking label generation, optimizer or backward call occurred. Only graph prototypes were instantiated for parameter counting and no-gradient neutral/memory forwards. No new checkpoint was saved.

Raw graph retrieval `.npz` files and verbose `.log` files stay local. Version control contains scoped source, manifests and small audit tables/reports. Publication updates only the authorized Stage8A branch, without force or merge. Equivalent Git-object publication is verified by matching all blob SHAs, tree and parent; the five unrelated Stage2C files remain untracked and unchanged.
