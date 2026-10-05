# Stage3B commands

Working directory: `/home/lrj/Prediction_Hivt`.
Python: `/home/lrj/anaconda3/envs/ped_intent/bin/python`.
Branch: `stage3b/type-embedding`, based on frozen `baf8b59b44047657dca0a64460dd4005c323af76`.
Registered training source commit: `0358a4434c50d6690b5ea4bae61012ccf3b24654`.

## Registration and pre-training gates

Run once from the frozen commit; never repeat registration after training:

```bash
/home/lrj/anaconda3/envs/ped_intent/bin/python outputs/stage3_multitype_hivt/00_manifest/stage3b_register.py --requirements /home/lrj/.codex/attachments/669f1f31-dac0-413a-a0c5-1fcfcd77ce14/已粘贴的文本.txt
/home/lrj/anaconda3/envs/ped_intent/bin/python -u outputs/stage3_multitype_hivt/03_type_embedding/stage3b_tiny.py
```

Initialization/unit/tiny PASS evidence is saved in the manifest and evaluation directories;
tiny output is in `stage3b_tiny_overfit.log`. Tiny checkpoint is discarded for formal initialization.

## Formal training

```bash
/home/lrj/anaconda3/envs/ped_intent/bin/python -u outputs/stage3_multitype_hivt/03_type_embedding/stage3b_train.py > outputs/stage3_multitype_hivt/08_logs/stage3b_formal_training.log 2>&1
```

The command was launched once from scratch after registration and pre-training gates.
It can resume only its own saved phase checkpoint on an actual interruption. It does not
restart completed phases or continue after the completed training summary. No training
source changes or hyperparameter exploration are allowed during the registered experiment.

## Frozen GT ledger

```bash
/home/lrj/anaconda3/envs/ped_intent/bin/python -u outputs/stage3_multitype_hivt/04_evaluation/stage3b_pairing_ledger.py > outputs/stage3_multitype_hivt/08_logs/stage3b_GT_ledger.log 2>&1
```

CPU-only read of the original unchanged VAL graph shards, required because the old
No-Type actor CSV lacks full GT arrays/mask bits. No prediction or old output changes.

## Final evaluation and figure audit sequence

Execute only after formal training exits successfully and the completed summary exists:

```bash
/home/lrj/anaconda3/envs/ped_intent/bin/python -u outputs/stage3_multitype_hivt/04_evaluation/stage3b_evaluate.py
/home/lrj/anaconda3/envs/ped_intent/bin/python -u outputs/stage3_multitype_hivt/05_figures/stage3b_plot_cases.py
/home/lrj/anaconda3/envs/ped_intent/bin/python -u outputs/stage3_multitype_hivt/05_figures/stage3b_plot_quantitative.py
/home/lrj/anaconda3/envs/ped_intent/bin/python -u outputs/stage3_multitype_hivt/05_figures/stage3b_check_exports.py
/home/lrj/anaconda3/envs/ped_intent/bin/python -u outputs/stage3_multitype_hivt/00_manifest/stage3b_final_audit.py
```

Review the actual exported PNG/PDF/SVG figures before marking `stage3b_visual_QA.json` PASS.
Then generate the evidence-based report and Stage3B artifact manifest:

```bash
/home/lrj/anaconda3/envs/ped_intent/bin/python -u outputs/stage3_multitype_hivt/09_reports/stage3b_finalize.py
```

The final report records actual executed/best steps and stop flags; per-step JSONs and
the unsmoothed CSV contain all real full-VAL points. Large checkpoints, per-actor CSV,
frozen GT ledger and original shards stay local. Only Stage3B files are staged, and
GitHub delivery targets `stage3b/type-embedding`; the frozen No-Type branch is never updated.
Stage4 is not executed.

## Completion and visual review

Formal training completed at global15500:5000 warm-up+10500 original NLL,
patience5 stop, best global13000. The evidence pipeline retained completed
fresh-VAL and qualitative gates when retrying the quantitative plotting import-path
fix; the original failure and retry logs are preserved. No training was rerun.

Visual review adjusted case layouts and bar/embedding text only. Case redraw
used the saved true source JSON with the same identities/coordinates and repeated
the <1e-4 numerical gate:

```bash
/home/lrj/anaconda3/envs/ped_intent/bin/python -u outputs/stage3_multitype_hivt/05_figures/stage3b_plot_cases.py --redraw-only
```

The final export QA, visual QA and protocol audit refer to the final files.
