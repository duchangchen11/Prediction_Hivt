# Diagnostic metric definitions

Candidate net displacement uses only the predicted twelve-point candidate: Euclidean distance between its first and last future points. Bins are [0,1], (1,5], (5,infinity) metres. No observed position or GT is used for this stratification.

Lane counts mean distinct complete-centerline map tokens within 10 m, before nearest-one fallback. MeanLaneTokensPerMode is uncapped; MeanTop8LaneTokensBeforeFallback reports the capped count separately. All six modes are equally weighted.

Type-aware coverage uses all matched relevant tokens before top8. The top8 selectors never include stop_line; its counts are audit-only. All-entity overflow includes all six primary entity types, including secondary context. Relevant overflow restricts to the actor-type relevance set. Future graph estimates use relevant selectors capped at eight, with genuinely empty selectors retained rather than invented fallback coverage.

Recovery rates divide by lane-uncovered candidate count. Drivable and carpark recoveries may overlap; their union and intersection are reported. StillUncoveredRate divides by all candidates, while StillUncoveredAmongLaneFallbackRate uses the conditional denominator.

GenericOnlyRate divides by all candidates and requires drivable context with no matched lane, connector, carpark, crossing or walkway. For pedestrians, DrivableOnlyPrimaryRate is separately computed over their three primary types; it can include secondary lane context, so it is distinct from GenericOnlyRate.

Missing vehicle motion groups have Actors=0 and Candidates=0, and undefined rates are reported as NA rather than zero or a passing gate.
