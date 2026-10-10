# Stage15B ranking training

All18 heads are fresh and trained only on fold-specific InnerTrain candidates. R2 uses673 parameters, seed2022 in every fold, original19→32→1 softCE, batch1024 including final partial batch, AdamW0.001/wd0.0001,max50epoch/patience5, strict InnerDev Overall Top1FDE.

Other heads use original15D nodes/17D edges/K6/radius50m/max8 neighbors; seeds2022/2122/2222, micro128×8=1024, continuous carry, common epoch order hashes, AdamW0.001/wd0.0001,max50/patience5. Strict Sdev=.5*VehicleDevFDE/R2VehicleDevFDE+.5*PedestrianDevFDE/R2PedestrianDevFDE. A is detached softCE q=softmax(-FDE/1m). C is mean sum softmax(z)*(FDE-minFDE)/max(1m,mean(FDE-minFDE)). No loss/architecture search. NG-A/NG-C7425; G-A/G-C24066; Matched-NG-C24001 active target-only parameters. Approximate capacity difference65 (0.2701%) and structural/optimization differences remain. Common additive score bias is softmax-unidentifiable, as in historical architecture.

Bicycle participates in all training targets/neighborhoods; all five ablation outputs route Bicycle to newly trained samefold R2. All18 heads freeze before unified Outer evaluation. No old head/predictor weights. Bounded new-data head checks discard their weights; original structural/capacity checks remain unchanged.

| Fold | Model | Params | SelectedEpoch | ExecutedEpochs | CheckpointScore | Seconds |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | R2 | 673 | 3 | 8 | 2.109865 | 7.266643 |
| 1 | NG-A | 7425 | 1 | 6 | 1.009010 | 107.631717 |
| 1 | NG-C | 7425 | 12 | 17 | 0.978517 | 267.888395 |
| 1 | G-A | 24066 | 9 | 14 | 1.002361 | 308.035192 |
| 1 | G-C | 24066 | 18 | 23 | 0.972230 | 511.607688 |
| 1 | Matched-NG-C | 24001 | 4 | 9 | 0.987525 | 165.563806 |
| 2 | R2 | 673 | 3 | 8 | 2.302662 | 7.379621 |
| 2 | NG-A | 7425 | 6 | 11 | 1.008034 | 183.606641 |
| 2 | NG-C | 7425 | 6 | 11 | 0.970008 | 170.048330 |
| 2 | G-A | 24066 | 8 | 13 | 0.993066 | 286.434992 |
| 2 | G-C | 24066 | 17 | 22 | 0.953121 | 481.764449 |
| 2 | Matched-NG-C | 24001 | 13 | 18 | 0.968496 | 325.979728 |
| 3 | R2 | 673 | 1 | 6 | 1.978357 | 5.175588 |
| 3 | NG-A | 7425 | 3 | 8 | 1.023588 | 125.272196 |
| 3 | NG-C | 7425 | 1 | 6 | 0.995522 | 88.153876 |
| 3 | G-A | 24066 | 5 | 10 | 1.000721 | 209.916574 |
| 3 | G-C | 24066 | 5 | 10 | 0.987530 | 208.078972 |
| 3 | Matched-NG-C | 24001 | 3 | 8 | 0.993047 | 138.274366 |
