# Stage7A-0 commands and boundaries

Working directory: `/home/lrj/Prediction_Hivt`.

Use only the existing `/home/lrj/anaconda3/envs/ped_intent/bin/python` environment. Set `PYTHONDONTWRITEBYTECODE=1` on every Python invocation. No packages, environments or data downloads were added. Local data references are in `stage7a_common.py` and the ignored existing `configs/local_paths.yaml`.

Initial freeze was run from exactly `e43347c2c2cd356775f4e0bb59fcf7dbda76f2b5`. The five unrelated untracked Stage2C redraw files were stashed by their exact paths, making `git status --porcelain` empty before tag creation. Their original byte hashes and stash reference are recorded in `stage7a_workspace_freeze_audit.json`; the files were restored byte-identically after branch creation.

```bash
git tag -a paper-v1-stage6a-r2 e43347c2c2cd356775f4e0bb59fcf7dbda76f2b5 -m 'Freeze paper V1: Stage5A + future interaction reliability R2'
GIT_TERMINAL_PROMPT=0 git push origin paper-v1-stage6a-r2
git switch -c stage7a/semantic-map-enhancement e43347c2c2cd356775f4e0bb59fcf7dbda76f2b5
```

The tag is a real annotated tag, not a lightweight ref. HTTPS push failed because no local GitHub credential is configured. Existing SSH transport was also checked with verified known hosts: port22 timed out; GitHub SSH-over-443 connected but rejected authentication, and the existing agent has no identities. No unverified host keys or secret extraction was used. The GitHub connector supports Git objects and branch publication, but no annotated tag write operation. Re-run the exact tag push above after configuring local Git credentials; do not send tokens/private keys through the conversation.

`stage7a_frozen_annotated_tag.txt` archives the canonical annotated tag object body. `git hash-object -t tag` on this file equals `bd70aeb7d61ec624f2559438b9ef5670d8c1e564`; it preserves the exact local tag identity for later recovery but does not claim that the remote tag ref exists.

Freeze registration is a once-only initialization step at the base commit; it verifies old files and records references without copying checkpoints:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/lrj/anaconda3/envs/ped_intent/bin/python outputs/stage7a_semantic_map_hivt/00_manifest/stage7a_register.py
```

Audit execution order (logs in `08_logs/`, all writes restricted to the new stage root):

```bash
PYTHONDONTWRITEBYTECODE=1 /home/lrj/anaconda3/envs/ped_intent/bin/python outputs/stage7a_semantic_map_hivt/01_data_audit/stage7a_map_inventory.py
PYTHONDONTWRITEBYTECODE=1 /home/lrj/anaconda3/envs/ped_intent/bin/python outputs/stage7a_semantic_map_hivt/01_data_audit/stage7a_turn_geometry_check.py
PYTHONDONTWRITEBYTECODE=1 /home/lrj/anaconda3/envs/ped_intent/bin/python outputs/stage7a_semantic_map_hivt/05_figures/stage7a_plot_audit.py
```

Direct visual review of all three random20 turn sheets, the two regional examples and all U-turn/boundary/short-path cases is saved in `stage7a_turn_manual_review.json`. That recorded review is a prerequisite for candidate adoption. `stage7a_exposure_registration.json` records definitions before the exposure/error audit.

```bash
PYTHONDONTWRITEBYTECODE=1 /home/lrj/anaconda3/envs/ped_intent/bin/python outputs/stage7a_semantic_map_hivt/02_semantic_cache/stage7a_semantic_cache.py
PYTHONDONTWRITEBYTECODE=1 /home/lrj/anaconda3/envs/ped_intent/bin/python outputs/stage7a_semantic_map_hivt/01_data_audit/stage7a_exposure_audit.py
PYTHONDONTWRITEBYTECODE=1 /home/lrj/anaconda3/envs/ped_intent/bin/python outputs/stage7a_semantic_map_hivt/05_figures/stage7a_plot_audit.py --semantic
```

Inspect all five10-case semantic sheets and record the findings in `stage7a_semantic_manual_review.json`, then run:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/lrj/anaconda3/envs/ped_intent/bin/python outputs/stage7a_semantic_map_hivt/09_reports/stage7a_finalize.py
```

Early audit development uncovered arc-junction repeated starts (segment matching now uses joint start+vector), empty point arrays (explicit Shapely object dtype) and34 legitimate no-current-actor null graphs (checked against original index status and zero actor/target counts). These implementation corrections did not alter any source data, map features or old results; the final complete pass is the reported result. Source dangling topology references are preserved and documented rather than repaired.

Only the Stage7A root is staged. Derived caches, full actor audit rows, logs and transient Git upload payloads are ignored. GitHub publication uses the existing authorized connector if native Git authentication remains unavailable; remote blob/tree/parent identities must equal the local committed tree. The exact remote commit and delivery proof are retained locally in the ignored Git-upload verification record.

`03_training/`, `04_evaluation/`, `07_checkpoints/` must contain README only. Stop after this audit and its delivery. Do not run Stage7A training or new Stage6A evaluation.
