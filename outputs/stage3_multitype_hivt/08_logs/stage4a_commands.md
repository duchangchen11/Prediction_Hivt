# Stage4A commands and protocol

Working directory: `/home/lrj/Prediction_Hivt`.
Interpreter: `/home/lrj/anaconda3/envs/ped_intent/bin/python`.
Branch: `stage4a/type-conditioned-interaction`, from frozen Stage3B commit
`824c87d7c1bdb89ef1b79572f463bf864a4b840a`.

## Registration and integration

```bash
git switch -c stage4a/type-conditioned-interaction 824c87d7c1bdb89ef1b79572f463bf864a4b840a
/home/lrj/anaconda3/envs/ped_intent/bin/python outputs/stage3_multitype_hivt/00_manifest/stage4a_register.py --requirements /home/lrj/.codex/attachments/8b3b11c1-dc34-443c-90a3-593236c9cab2/已粘贴的文本.txt
/home/lrj/anaconda3/envs/ped_intent/bin/python -u outputs/stage3_multitype_hivt/03_type_interaction/stage4a_tiny.py
```

Registration independently verified all 850 prior frozen scene-shard hashes and
1006 protected file hashes, including every frozen Stage3B artifact/checkpoint.
The common constructor reproduces the exact canonical Stage3B step0 state,
copies shared name/shape matches bitwise, and restores the original CPU RNG state.
New relation MLP final layer is zero initialized. The original runtime and optimizer
are unchanged. Its module is named `relation_mlp` to preserve the existing optimizer's
parameter-name grouping (which treats any name containing `bias` as no-decay).

Only the five requested configuration keys are added. The copied YAML's historical
budget entries stay unchanged; execution follows the same actual registered Stage3B
5000 warm-up + at most16000 NLL / global21000 budget.

Official train700 / val150, test unused; batch16 / K6 / Th5 / Tf12 / seed2022.
Each real500-update VAL uses all150 official VAL scenes. Selection is strict
full-horizon overall minFDE6, final best selected only after an original-NLL update.
Five consecutive nonimprovements stops NLL. Own warm-up best restores model,
AdamW and RNG; only LR changes0.001->0.0001 and NLL sampler cursor resets as Stage3B.

## Formal execution

Formal training is forbidden until all12 unit tests, neutral step0, first10-update
relation gradients, tiny overfit and the preregistered t0-only20m subgroup ledger pass.
The tiny model/optimizer are discarded. No trained Stage3B weights enter formal init.

```bash
/home/lrj/anaconda3/envs/ped_intent/bin/python -u outputs/stage3_multitype_hivt/03_type_interaction/stage4a_train.py
```

The training source SHA256 and exact preregistration Git commit are recorded before
formal execution. Real CSV rows and matching full-VAL JSON record every500 updates.
No Stage4B Reliability, Intent, future compatibility, calibration, topology changes,
class-balanced loss, oversampling, extra seeds or hyperparameter search are executed.

## 2026-10-05 authorized continuation

User explicitly authorized continuation of current Stage4A. Resume from own saved NLL global5500 checkpoint; unsaved5600–5700 log updates replayed. Model, AdamW, RNG and scene sampler cursor restored; registered training sources unchanged. Training PID43145. Final evaluator waits for training exit and COMPLETE summary; figures wait for all four evaluation stages. Stage4B remains outside scope.

```bash
/home/lrj/anaconda3/envs/ped_intent/bin/python -u outputs/stage3_multitype_hivt/03_type_interaction/stage4a_train.py
/home/lrj/anaconda3/envs/ped_intent/bin/python -u outputs/stage3_multitype_hivt/04_evaluation/stage4a_wait_evaluate.py --training-pid 43145
/home/lrj/anaconda3/envs/ped_intent/bin/python -u outputs/stage3_multitype_hivt/05_figures/stage4a_wait_figures.py
```
