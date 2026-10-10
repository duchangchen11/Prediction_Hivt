# Stage17 — efficiency

Measured on NVIDIA GeForce RTX 3080, torch2.5.1+cu124, original environment, no simultaneous scorer-fit job. All methods use the same first1,024 InnerDev targets per fold,128-target microbatches,10 warm-up cycles and50 timed cycles; CUDA synchronization bounds each timing. Head-only means across three folds:

| Model | Parameters | MeanMSPer1024 |
| --- | --- | --- |
| Adapted TNT Scoring | 16001.000000 | 5.142953 |
| G-A | 24066.000000 | 10.421487 |
| G-C | 24066.000000 | 10.430799 |
| Matched-NG-C | 24001.000000 | 6.589770 |
| NG-A | 7425.000000 | 4.791603 |
| NG-C | 7425.000000 | 4.795783 |
| R0 | 0.000000 | 0.081508 |
| R2 | 673.000000 | 2.045010 |

Whole engineering path on the **same first16 InnerDev windows**, past-only target eligibility,3 warm-ups and10 repeats per model/fold. Equal fold means in seconds (counts differ by fold; see source CSV):

| Model | MeanSeconds |
| --- | --- |
| Adapted TNT Scoring | 0.080884 |
| G-A | 0.238925 |
| G-C | 0.235769 |
| Matched-NG-C | 0.232246 |
| NG-A | 0.222515 |
| NG-C | 0.222686 |
| R0 | 0.066129 |
| R2 | 0.209924 |

[Head source](07_efficiency/stage17_head_compute.csv) and [whole-path source](07_efficiency/stage17_whole_compute.csv) record scopes, counts, window identity digest and GPU peaks. Head-only excludes feature construction,H2D and routing. Full path includes frozen predictor, ego conversion, required context/features, head, actual Bicycle routing and argmax; excludes raw file I/O and audit hashing. TNT context capture is included in full path and shares the predictor forward. G-C retains its actual CPU graph preparation. Different feature pipelines explain engineering overhead; these are not deployment FPS estimates.

TNT has16,001 new scoring parameters and uses independent frozen64D local context. G-C has24,066 head parameters, NG-C7,425, Matched24,001; the shared Stage5A predictor650,403 is not counted as newly trained. Existing graph/no-graph Bicycle routing also invokes frozenR2(673 params); pure TNT does not. Including the shared predictor and required route heads: R0=650,403; R2=651,076; NG-A/NG-C=658,501; G-A/G-C=675,142; Matched-NG-C=675,077; pure Adapted TNT=666,404 total loaded parameters. Only the16,001 TNT head parameters receive updates in this stage. Parameter/input differences remain explicit; the comparison is not a parameter-matched or identical-input experiment.

Added context arrays 724,263,840 bytes; selected scorer checkpoint sizes {'1': 69372, '2': 69372, '3': 69372}; disk free at benchmark 24,928,440,320 bytes. Original large candidate caches reused read-only. Local checkpoints/caches stay ignored; source-data tables and figure exports alone are published.
