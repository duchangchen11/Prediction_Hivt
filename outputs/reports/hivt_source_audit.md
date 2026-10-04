# HiVT source audit

repo=https://github.com/ZikangZhou/HiVT

commit=6876656ce7671982ebdc29113aaaa028c2931518

The author repository identifies this as the official CVPR 2022 implementation. A bounded local search found no existing HiVT checkout. Sources are frozen under `baselines/hivt_official`; LICENSE is Apache-2.0, original copyright headers remain, and `SOURCE.json` records file hashes. No pretrained weights were downloaded.

original dataset=Argoverse Motion Forecasting v1.1

original history length=20 steps at 10 Hz (approximately 2 s)

original future length=30 steps at 10 Hz (3 s)

number of modes=6

map representation=lane centerline segment vectors, lane-to-actor relative offsets; original categorical intersection/turn/traffic-control embeddings

local radius=50 m per actor (upstream default)

loss=best-mode (minimum summed masked L2 displacement) Laplace NLL + detached soft-target mode cross entropy

metrics=upstream validation chooses the mode minimizing final displacement, then computes ADE/FDE/MR on that mode; MR is final displacement >2 m

required dependencies=original README specifies Python3.8 / torch1.8 / PyG1.7.2 / pytorch-lightning1.5.2 / Argoverse API

## Stage 2 adaptations

Model name: HiVT-NuScenes-Vehicle-Baseline, HiVT-64, 6 modes, 8 attention heads, 4 temporal layers, 3 global layers. Historical/future steps are configurable and set to 5/12. No new architecture or feature head is added.

`models/hivt_runtime` preserves the official encoder, global interactor, MLP decoder, initialization and loss calculations. Changes: package-relative imports; PyG `__inc__` accepts current extra arguments; the custom Transformer layer accepts PyTorch's `is_causal` keyword while retaining its explicit causal mask. Original source remains untouched.

`models/hivt_nuscenes.py` uses a native PyTorch wrapper and training loop, avoiding obsolete Lightning Trainer and TorchMetrics APIs. It rotates a cloned graph so repeated forwards never rotate targets cumulatively. Vehicle target masks exclude context-only nodes and actors with fewer than two historical observations, as upstream preprocessing does.

The nuScenes adapter retains the verified t0-ego frame, uses agent motion heading for the original HiVT per-agent rotation, and also stores annotation heading. Inputs remain relative displacement, not velocity. Actual timestamps are stored and used for CV/horizon checks; irregular keyframe timing is not silently resampled and no time embedding is added.

Only centerline geometry is used. Existing lane semantic embedding inputs are constant zero placeholders (unspecified), not inferred labels. This intentionally loses original Argoverse lane semantics while retaining the model structure.

Metric source: `baselines/hivt_official/models/hivt.py` and `metrics/mr.py`; corresponding benchmark: https://github.com/argoverse/argoverse-api/blob/master/argoverse/evaluation/eval_forecasting.py . Report both benchmark-selected-mode ADE and independent minimum ADE, explicitly separated.
