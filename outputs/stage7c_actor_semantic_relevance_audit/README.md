# Stage7C: Actor-conditioned semantic relevance audit

Source: Stage7A final commit `77e77b3e1f48a5e68e48d5665308232b97d058f4`.
Branch: `stage7c/actor-conditioned-semantic-relevance-audit`.
No training, fine-tuning, new checkpoint, new attention model, test split or Stage7B.

Read [the complete audit report](09_reports/stage7c_actor_conditioned_semantic_audit_report.md)
and [machine-readable final fields](09_reports/stage7c_final_fields.json).
The descriptive decision is `RELEVANCE_MISALLOCATION`, with localized and weak
Moving/GT-left evidence. The overall Vehicle attention/error association is near
zero. `CONTINUE` / `YES` is only a score-only future diagnostic recommendation;
no implementation or training is authorized by this audit.

All 54990 full-horizon VAL targets are paired by frozen identity/GT; Turning count
1663. Observation integrity checks 100 different random VAL batches per predictor,
with exact-zero prediction/logit/probability changes. The captured 26807585 actual
lane-actor edges and large actor CSVs are local-only; SHA/metadata are versioned.
Existing Stage7A definitions and NOT_SUPPORTED/NO/NO conclusions are unchanged.

Directories:

- `00_manifest`: requirements, fixed interpretation/geometry rules, frozen SHA audit.
- `01_attention_capture`: no-op hooks, integrity runner, normalized edge archives.
- `02_relevance_analysis`: true segment/GT geometry, paired statistics, bootstrap.
- `03_representation_analysis`: local residual pattern vectors and frequencies.
- `04_perturbation`: exactly one SHUFFLE and one CENTERED full VAL inference pass.
- `05_figures`: four matching case panels in PNG/PDF/SVG, and export audits.
- `06_tables`: required and supplementary small statistical tables.
- `07_cases`: complete measured case source data and fixed selection rules.
- `08_logs`: commands, local run logs.
- `09_reports`: complete interpretation, decision and final fields.

Run commands and dependency order are in
[stage7c_commands.md](08_logs/stage7c_commands.md). For inspection, for example:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/lrj/anaconda3/envs/ped_intent/bin/python \
  outputs/stage7c_actor_semantic_relevance_audit/01_attention_capture/stage7c_read_edges.py \
  outputs/stage7c_actor_semantic_relevance_audit/01_attention_capture/stage7c_edges_batch_0000.npz \
  --model Stage3B --limit 5
```

GT-nearest rank uses all current graph segments and average attention ties;
unobserved edges have zero weight. Incoming-only and co-nearest sensitivity
metrics, empty-edge counts, entropy denominators and NO_CONNECTOR membership
are explicitly reported. Rank is segment-level, not token-level; turn ambiguity
counts distinct map tokens, not their multiple segments. SHUFFLE and CENTERED
are OOD diagnostics, not trained models or official baselines.

STOP after delivery. Await new authorization before any future experiment.
