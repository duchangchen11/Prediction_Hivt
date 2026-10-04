# Stage3A execution contract

Existing Python: `/home/lrj/anaconda3/envs/ped_intent/bin/python`. Official nuScenes source remains read-only. The Stage2C derived metadata SQLite is reused through a read-only connection; it is not copied or modified. No new environment, download or dependency update.

Run, in order, from the project root:

```bash
/home/lrj/anaconda3/envs/ped_intent/bin/python outputs/stage3_multitype_hivt/02_preprocessed/stage3_prepare.py
/home/lrj/anaconda3/envs/ped_intent/bin/python outputs/stage3_multitype_hivt/05_figures/stage3_plot_data.py
/home/lrj/anaconda3/envs/ped_intent/bin/python outputs/stage3_multitype_hivt/03_no_type_baseline/stage3_tiny.py
/home/lrj/anaconda3/envs/ped_intent/bin/python outputs/stage3_multitype_hivt/03_no_type_baseline/stage3_train.py --stage benchmark
/home/lrj/anaconda3/envs/ped_intent/bin/python outputs/stage3_multitype_hivt/03_no_type_baseline/stage3_train.py --stage warmup
/home/lrj/anaconda3/envs/ped_intent/bin/python outputs/stage3_multitype_hivt/03_no_type_baseline/stage3_train.py --stage nll
```

Training is gated by the three-class tiny PASS. Tiny uses train-only, full context windows with at least10 vehicle/10 pedestrian/5 bicycle target actor-windows; initial/final ADE and FDE and fixed-scale diagnostic regression must decrease >=50% for each class. No new formal-training weighting, oversampling or architectural changes. Smoke models and tiny weights are discarded; formal initialization uses seed2022 and the exact Stage2C HiVT64 constructor. Model input explicitly removes agent_type; metadata is used only for loss diagnostics and evaluation.

Formal protocol: fixed-scale warm-up LR.001,min2500/max10000; original learnable Laplace NLL LR.0001,max10000; complete official VAL every500 updates, early stopping five consecutive non-improvements after warm-up minimum. No scheduler. The best warm-up overall full-horizon FDE weights, AdamW and RNG state initialize original NLL. Only post-update NLL VAL overall full-horizon multi-type FDE selects the primary checkpoint. Sampler seed=2022+epoch (warm-up),2022+100000+epoch (NLL), matching Stage2C. Primary results use best-FDE mode ADE, MR endpoint>2m; independent minimum ADE and argmax-probability Top1 are separately recorded. NLL is the original best-summed-L2-mode Laplace density mean; negative continuous density NLL is valid. Original regression reduction weights valid time-coordinate observations without class-specific weights, matching Stage2C.

Natural-distribution training retains parked/stopped actors and all scene-window context; no type embedding or Stage3B. Exact Stage2C vehicle identities, positions, masks, targets and headings are compared during all850 scene shards. Previous tracked results and five uncommitted Stage2C redraw files are SHA-frozen. User explicitly authorized completing the entire Stage3A, overriding the earlier half-plan boundary.
