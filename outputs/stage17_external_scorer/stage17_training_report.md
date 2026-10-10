# Stage17 — training report

**COMPLETE: exactly three scoring components, zero HiVT training steps.** Optimizer/choice rules were frozen publicly before fits and new TNT Outer metrics. Source/model/loss/temperature choices are unchanged after evaluation.

| Fold | Seed | SelectedEpoch | ExecutedEpochs | DevSelectionScore | TrainTargets | Seconds | PeakGPU_MB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2022 | 15 | 20 | 0.993896 | 159151 | 209.218581 | 75.426816 |
| 2 | 2122 | 8 | 13 | 1.016541 | 157356 | 135.346507 | 75.426816 |
| 3 | 2222 | 12 | 17 | 1.052388 | 148900 | 165.948858 | 75.426816 |

Total bounded scorer fit time: 0.141809 GPU hours (includes development assessment and checkpoint I/O). All distinct InnerTrain target identities were used. Selection is the strict minimum registered balanced relative Vehicle/Pedestrian InnerDev FDE; no official VAL, HeadDev70 or Outer selection. Original fold seeds2022/2122/2222 and private permutation/carry rules are preserved. No tiny or previous scoring weights initialize a formal scorer.

Training: AdamW lr0.001/wd0.0001, micro128×8, effective1024, FP32, at most50 epochs, patience5. Real pinned `TrajScoreSelection.loss` sum divided by effective batch; full-path maximum squared distance soft CE with fixed temperature0.01. Score softmax temperature1. No Loss C replacement, calibration, clipping or new encoder. This retains the **paper/component CE** mechanism; the unofficial full trainer’s active BCE, scheduler and joint-system fitting are explicitly different. Selection/weight decay follow the frozen project head comparison, not full TNT reproduction.

Each fold retains step-zero receipt, epoch curve, selected checkpoint, last training state with optimizer, pending/coverage and RNGs. [Global selected hashes](05_training/stage17_all_frozen.json) bind exactly three checkpoints; [config/curves](05_training/). The registered cap and stopping condition were not extended after Outer evaluation.
