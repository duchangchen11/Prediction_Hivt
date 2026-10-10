# Stage17 — integrity checks

**PASS.** 108 paired Train/Dev batches passed before formal fitting; 162 paired batches including post-freeze Outer export. Exact raw-prediction, logits and probability instrumentation differences are zero. All original ego candidate coordinates/logits/probabilities match their Stage15B scene caches **bitwise**, across every exported window. Predictor SHA256 and tensor-state digests remain unchanged.

| Fold | Role | Scenes | Targets | PairedBatches | RawDiff | LogitDiff |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | InnerTrain | 378 | 159151 | 18 | 0.000000 | 0.000000 |
| 1 | InnerDev | 42 | 18327 | 18 | 0.000000 | 0.000000 |
| 1 | OuterTest | 210 | 82673 | 18 | 0.000000 | 0.000000 |
| 2 | InnerTrain | 378 | 157356 | 18 | 0.000000 | 0.000000 |
| 2 | InnerDev | 42 | 19084 | 18 | 0.000000 | 0.000000 |
| 2 | OuterTest | 210 | 83711 | 18 | 0.000000 | 0.000000 |
| 3 | InnerTrain | 378 | 148900 | 18 | 0.000000 | 0.000000 |
| 3 | InnerDev | 42 | 17484 | 18 | 0.000000 | 0.000000 |
| 3 | OuterTest | 210 | 93767 | 18 | 0.000000 | 0.000000 |

Actor-centered scoring coordinates use a rigid transform of those unchanged candidates. Largest round-trip difference from native HiVT local coordinates: 2.28881835938e-05 m; original FDE agreement tolerance was 5e-5 m plus relative2e-6. This numerical transformation is not a modification of the immutable candidate geometry. Evaluation reads original candidate error arrays, preserving the oracle bitwise.

Future poisoning replaces future positions/padding/labels, then confirms the predictor’s strict observed input and outputs are identical. The context hook captures only the observed decoder input and never returns a replacement. `TNTStore.inputs` physically excludes the separate GT array. GT is used only for fitting loss and offline masks/evaluation. No future-conditioned context encoder is fitted.

All three actual-InnerTrain preflights pass [receipts](04_checks/stage17_preflight.json): shapes [128,1,64]/[128,6,24]→[128,6], normalized finite probabilities, all 16,001 parameters connected to finite gradients, exact component loss equivalence,20 tiny updates, strict checkpoint reload and next-optimizer-step replay. Tiny weights are discarded. Train/Dev scenes are disjoint and checked against original protocol tokens. Each optimization stream uses only its InnerTrain targets; selection only InnerDev. Outer folders, original Outer performance arrays, official VAL/test and HeadDev reads are blocked during fitting.

All three selected scorers were globally frozen before any new TNT Outer performance: [global freeze](05_training/stage17_all_frozen.json), [public registration](00_manifest/stage17_public_registration.json). All 260,151 original Outer actor-windows and630 scenes are retained; no difficult samples removed. Existing seven-model metrics and all eight models’ minFDE6 are bitwise unchanged: [evaluation receipt](06_evaluation/stage17_evaluation_integrity.json) (large arrays local only).

Historical preservation: 3882 files and 1971 frozen candidate/context files verified against original hashes; [preservation receipt](00_manifest/stage17_historical_preservation.json). Historical Stage15B/16 scientific outputs and checkpoints were never overwritten. GPU/disk checks use the existing environment; no HiVT training or new trainable encoder.
