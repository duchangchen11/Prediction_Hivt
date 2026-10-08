# Stage8A-1 execution protocol

Only the user-authorized ranking heads are fitted. Historical stages, 0B/0C,
Stage5A candidates and Stage6A R2 are immutable. Training and local caches reside
under the existing Stage8A root in the new 01_training directory. Large tensors
and checkpoints are not published.

1. Hash frozen sources and preserve the Stage6A 630/70 scene split. Construct
   observable graph tensors with the unchanged 0C selector and feature functions.
   Join full-horizon GT labels after graph construction. Fit per-column valid
   continuous moments on HeadTrain packed contexts only; audit actual normalized
   mean/std, padding, invalid directions and binary fields.
2. Check exact parameter counts and common initial weights. Use 1024 real
   HeadTrain targets for neutral identity. Train each independent FP32 tiny head
   on the same fixed128 targets for300 updates. Require20% loss reduction and
   frozen-predictor gradient0. Discard all tiny weights.
3. Sequentially fit G1, G2, G3 from their frozen seeds using AdamW1e-3,
   weight decay1e-4, effective1024 targets, max50 epochs and patience5.
   Carry epoch remainder into the next shuffled epoch rather than perform a
   smaller optimizer step. Select only minimum HeadDev soft ranking loss.
4. Freeze all three selected checkpoint SHA256s before any official evaluation.
   One fresh VAL150 pass loads frozen R0/R2 and all three heads, sharing one
   immutable candidate tensor. Verify exact identity with historical prediction
   caches and identical oracle metrics. No official test.
5. Reuse historical offline semantic memberships; compute GT turn signs only in
   evaluation. Preregister actor-level map availability over all6 modes and
   interaction distance bins before looking at results. Bootstrap all comparisons
   with the same seed2022 whole-scene resampling matrix,1000 draws.
6. Apply the user's fixed support gates. Select cases only after aggregate
   results. Measure head forward, live CPU graph/retrieval and predictor forward
   as separate scopes. Publish new code and small reports on the authorized
   branch without merging. Stop after the final report.

The frozen0C interaction message input is raw15+raw15+17=47 dimensions;
its attention input is hidden64+hidden64+17=145. This exact frozen architecture
and the required parameter counts take precedence over an ambiguous hi/hj
notation in the Stage8A-1 prose; no architecture change is introduced.
