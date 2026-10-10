# Stage17 — frozen adapter design

The [immutable registration](00_manifest/stage17_registration.json) is written before any new TNT Outer metric and before formal scoring fits. All choices below are fixed without Outer tuning.

## What is fitted

The unmodified public `TrajScoreSelection(C=64,T=12,H=64,temper=0.01)` is the sole trainable component. Input: [B,1,64] observed context plus [B,6,24] candidate coordinates. Output: six scalar scores, their softmax probabilities and argmax original mode identity. No separate trainable history encoder, candidate generator, semantic module, neighbor-message branch, residual original-logit addition or calibration parameter is created. The active loss is the original **component** soft CE; the full unofficial TNT trainer’s BCE difference is documented in the source audit.

## Context and coordinates

A decoder forward **pre-hook** captures the frozen Stage5A decoder’s local input, after actor-type fusion. This is a real 64-dimensional latent from five observed positions, history masks, lanes and local actor interactions. It is detached. It replaces TNT’s VectorNet scene context; the Stage5A long-range global interactor is mode-specific and is not pooled or repurposed. Consequently this is an adaptation with a different context encoder and horizon, not identical TNT information or full-system training.

The scorer uses current-actor-centered coordinates, oriented by the actor’s **observed** t0 angle, in meters. Both cached predictions and supervised future labels receive the same rigid transform. Coordinates are absolute future positions relative to t0, not incremental displacements. No candidate geometry or order in the immutable cache is changed. Round-off of the coordinate conversion is audited separately from exact candidate preservation. Existing HiVT historical padding remains active; all six candidates are valid. Only the original full-horizon evaluation mask defines targets. No new missing-GT fill or difficult-actor filtering occurs.

No additional feature standardization is fitted. The public scorer’s LayerNorm remains unchanged. Labels are stored separately; `TNTStore.inputs` returns only context and candidates, never GT. Model inference receives only the Stage15B strict observed-input whitelist. Future labels, future masks and target membership cannot influence context computation.

## Protocol and routing

The original scene tokens and order are reused: 378 InnerTrain, 42 InnerDev, 210 OuterTest and 70 quarantined HeadDev per fold. No official VAL, HeadDev fitting, or HiVT training. Formal seeds 2022/2122/2222; AdamW lr=0.001, wd=0.0001, FP32, microbatch128 ×8, effective1024, continuous remainder carry, maximum50 epochs, patience5. Select strict minimum balanced relative Vehicle/Pedestrian InnerDev Top1FDE, with the same frozen fold-R2 development denominators. Epoch1 is first eligible; temperatures stay 0.01 for labels and 1 for inference.

**Primary external baseline scores all types using TNT**, including Bicycle. Existing graph and no-graph methods retain their frozen R2 Bicycle route. To expose this routing difference, report an additional **TNT + same-fold frozen R2 Bicycle route** sensitivity result; it never changes the selected TNT checkpoint or replaces the primary result. Vehicle, Pedestrian and MovingVehicle comparisons do not depend on this routing difference.

## Gates and limitations

Before fitting: shape, finite connected gradients, probability sum, actual loss equivalence, split isolation, tiny optimization, save/load/optimizer replay, future-input poisoning and exact cached candidate/logit/probability invariance. Tiny weights are discarded. No numerical failure may be repaired by changing the loss, candidate count, or Outer-based choices.

Only after all three selected scorer checkpoints are frozen may Outer context export and performance evaluation begin. Shared candidates imply unchanged minFDE6. Bootstrap is supplementary paired scene analysis with 95% descriptive CI, outside the Stage15B confirmatory family3. The context provides independent learned information beyond the existing cached node/edge features; the comparison controls candidate geometry and scene protocol, **not identical input information**. Source licensing remains unverified; third-party files are local-only.
