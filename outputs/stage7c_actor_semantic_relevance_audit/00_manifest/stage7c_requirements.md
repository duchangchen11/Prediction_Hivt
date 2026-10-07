# Stage7C-Audit: Actor-Conditioned Semantic Lane Relevance Audit

Authorized source: Stage7A final 77e77b3e1f48a5e68e48d5665308232b97d058f4.
Branch: stage7c/actor-conditioned-semantic-relevance-audit. Unique Stage7C root.

Only evaluation and diagnosis: no training, fine-tuning, new checkpoint, model
architecture implementation, Stage7B, test split, main merge, or historical edits.
Freeze Stage3B and Stage7A checkpoints, 9D semantic definitions, 20-degree turn
threshold, 50m local radius, crosswalk intersection definition, all scene shards
and all Stage7A conclusions (NOT_SUPPORTED / NO / NO).

Official VAL150: 3603 supervised windows, 54990 full-horizon actor-window targets.
Observe post-softmax 8-head actor-lane attention in eval mode for both predictors.
Audit at least 100 random VAL batches: prediction/logit/probability differences
strictly below 1e-6; failure means STOP. Stage3B never receives lane_semantic;
its lane tags are joined offline by unchanged tokens.

For every retained lane-actor edge save scene/sample/instance IDs, local actor
and lane indices, lane token, 9D semantic, actor-lane distance, alpha[8], mean
and max alpha. Use true segment distance from the 12-step GT future polyline
to [lane_position, lane_position + lane_vector]. GT is analysis-only.
Primary metrics: nearest-GT segment attention rank and weight; auxiliary
relevant mass at <=2m/<=4m, mean per-head entropy, top1/3/5 GT distance.

Count distinct left/straight/right connector tokens with complete centerline
strictly within20m of vehicle t0 (all regional map tokens, not segment duplicates).
TurnOptionCount20 is number of nonempty turn classes, 0/1/2/3. TurningVehicle_GT
membership must exactly equal frozen Stage7A Count1663; retain signed heading.
Measure correct/opposite/straight/unknown turn attention and highest-attention
connector match rate (NO_CONNECTOR is explicitly recorded).

Decompose r(s), r(0), delta_r(s); occurrence-weighted semantic group norms and
ratio ||delta_r||/(||r||+1e-8), plus actor attention-weighted semantic effect and
GTRelevant2m effect. Capture geometry/key/value representations diagnostically.
Join exact actor identities with frozen Stage3B/Stage7A errors. Descriptive
Spearman correlations only; p-values do not imply causality or independent actors.
1000 paired whole-scene bootstrap replicates, seed2022, all150 VAL scenes,
for relevant mass, nearest rank, entropy, correct/opposite turn and connector match.

Inference perturbations: reuse ON/ZERO frozen results; one full SHUFFLE pass
(within-graph semantic row permutation only, seed2022), one full CENTERED pass
(r(s)-r(0), same checkpoint). Neither is a baseline or trained new method.
Report Overall, Vehicle, MovingVehicle, TurningVehicle_GT minFDE/Top1FDE.

Six required tables: attention_relevance, turn_attention, turn_ambiguity,
residual_decomposition, attention_error_correlation, perturbation_results.
At least four illustrative cases: harmful relevant-attention shift, opposite-turn
increase, improvement with correct-turn increase, reasonable attention with
prediction degradation. Same actor/GT/map/axes; show semantic colors, both
attention fields and both best trajectories. Do not claim case frequencies.

Final report sections: Stage7A Failure Question; Attention Capture Integrity;
GT Future Lane Relevance; Stage3B vs Stage7A Attention; Turning Vehicle;
Turn Ambiguity; Semantic Residual Decomposition; ON/ZERO/SHUFFLE/CENTERED;
Attention-Error Association; Case Studies; Interpretation.

Decision must be one of RELEVANCE_MISALLOCATION, VALUE_CONTAMINATION,
GENERIC_ADAPTER_DOMINANCE, NO_ACTIONABLE_SIGNAL. RecommendSemanticAttentionModel
YES only for first two; SemanticRoute CONTINUE only when recommendation YES.
Record a score-only semantic attention proposal only when appropriate; do not
implement or train it. Commit/push isolated code, small tables, figures and report
to Stage7C branch, no merge. Final reply uses the fields specified by the user,
then STOP and wait for review. Large edge archives/actor CSVs stay local.
