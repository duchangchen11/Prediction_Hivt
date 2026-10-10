# Stage17 — baseline source audit

**Verified source; scoring-component adaptation, not a full TNT reproduction.**

[TNT: Target-driven Trajectory Prediction](https://proceedings.mlr.press/v155/zhao21b.html), Hang Zhao et al., CoRL 2020 proceedings, PMLR 155:895–904, **2021**. The paper’s Eq. 5–6 scores complete candidate trajectories conditioned on scene context using softmax and distance-derived soft-label cross entropy. Its full pipeline also generates targets and trajectories and suppresses redundant predictions. This experiment retains only scoring, because generation or suppression would change the frozen six candidates. [Primary paper, §4.4](https://proceedings.mlr.press/v155/zhao21b/zhao21b.pdf).

The [public implementation](https://github.com/Henry1iu/TNT-Trajectory-Prediction) is **unofficial**. The exact Stage16 source commit is `bcbccdc1d35a717793e3caa1d599c1f700612227`, reused without updating to a newer source. File hashes are in [source manifest](01_source/stage17_original_scoring_source.json).

## Actual computation

[`TrajScoreSelection`](https://github.com/Henry1iu/TNT-Trajectory-Prediction/blob/bcbccdc1d35a717793e3caa1d599c1f700612227/core/model/layers/scoring_and_selection.py), lines 34–122, consumes **context [B,1,C]**, trajectories **[B,M,2T]**. Its docstring incorrectly describes context as two-dimensional; the actual assertion requires three dimensions. The context is repeated across candidates, concatenated with flattened absolute future coordinates, passed through the actual residual MLP and a scalar linear layer, then softmax across M. No project residual logit or Loss C is added.

[`MLP`](https://github.com/Henry1iu/TNT-Trajectory-Prediction/blob/bcbccdc1d35a717793e3caa1d599c1f700612227/core/model/layers/basic_module.py), lines 11–69: two linear layers, LayerNorm, ReLU, plus a projected residual shortcut when input/output dimensions differ. At C=64,T=12,H=64 the scorer has **16,001 trainable parameters**. The exact classes are compiled from the hash-verified source AST by [Stage17 loader](00_manifest/stage17_common.py); method bodies and initialization remain unchanged. No third-party code is copied into the public project.

The **component’s `loss`** calls `distance_metric`: maximum over 12 points of squared 2D error, then `q=softmax(-distance/0.01)`, `L=-sum(q log p)`. Stage17 divides this sum by the effective batch size for a mean actor loss. The distance is numerically squared meters; despite the paper’s prose description of distance units, no square root is inserted. Temperature is fixed, not tuned.

## Important source-path discrepancy

The enlarged audit inspected the active full-system trainer, beyond Stage16’s component contract audit. `core/trainer/tnt_trainer.py:107–113,217–231` calls `TNTLoss`; `core/loss.py:126–130` uses **BCE on softmax probabilities** rather than the component’s soft CE. The paper and component agree on soft CE. **Stage17 deliberately uses the real scoring component’s own `.loss`, consistent with the paper scoring equation, and does not claim to reproduce the unofficial full trainer.** This newly identified difference does not rewrite Stage16’s historical component audit.

Original full training: AdamW, CLI default lr=0.001, weight decay=0.01, 50 epochs, scheduled learning rate with warm-up, and best full development loss (`train_tnt.py:61–79,97–121`). Stage17 registers the same optimizer family but project-comparable head weight decay, fixed learning rate, early stopping and balanced Vehicle/Pedestrian development FDE selection. Target/regression/auxiliary losses and joint encoder fitting are absent because only the scorer is fitted. Exact extra file hashes: [expanded source manifest](01_source/stage17_extra_source_manifest.json).

## License and attribution

The complete pinned tree contains 139 entries and **no LICENSE, COPYING or NOTICE file**; [tree receipt](01_source/stage17_license_tree_audit.json). The inspected README has no explicit code license grant or citation instructions. Public visibility is not represented as a redistribution license. Third-party source stays in ignored local caches; the repository distributes its own loader, source URLs and hashes, not the upstream class text. **Permission for redistribution remains unverified.** Before distributing a bundled scorer implementation, seek author permission or a clear license; this stage does not contact authors or invent permission.

Cite the original TNT paper and separately acknowledge Jianbang Liu / Henry1iu’s unofficial implementation, exact commit and adaptation. Publication name: **“Adapted TNT Scoring (frozen HiVT context and candidates)”**. Report local research results and the licensing limitation honestly; no external paper benchmark number enters the comparison.
