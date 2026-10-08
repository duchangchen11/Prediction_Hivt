# Stage8A-1 tiny gate protocol amendment

Historical commit `2a79b0d687192002537073147d93658bf4c6423e` retains `STOP_TINY_GATE_FAILURE` and the original three FAIL outcomes. The failure source is `INFEASIBLE_TINY_CRITERION`, classified as `PROTOCOL_BUG`.

Original criterion: `FinalLoss < 0.8 * InitialLoss`. On the unchanged128 targets, `H(q)=1.294783592224121` and the old upper bound is `1.250974082946777`. Because `SoftCE(q,p)=H(q)+KL(q||p)>=H(q)`, that bound is unattainable.

Amended engineering criterion: `ExcessLoss=L-H(q)`, `ExcessLossReduction=1-(FinalLoss-H)/(InitialLoss-H) >=0.90`. Finite/nonzero residuals and graph gradients, frozen predictor gradients0, candidate.requires_grad=False and no NaN/Inf remain required.

Only saved seed2022 fixed128/300-update FP32 results are rejudged. No resampling, new tiny update or hyperparameter change. This amendment is recorded before any formal training or officialVAL.

|Variant|Initial excess|Final excess|Reduction|Amended gate|
|---|---:|---:|---:|---|
|G1|0.268934011459|0.001233935356|99.541175%|PASS|
|G2|0.268934011459|0.001076936722|99.599554%|PASS|
|G3|0.268934011459|0.000429749489|99.840203%|PASS|

Formal outputs use `01_training/G{1,2,3}/formal/` to preserve first-attempt config/gradient/tiny records. The final resumed report has a new filename; the first STOP report is immutable.
