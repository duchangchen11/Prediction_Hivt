# Multipart polygon handling

The first feature-audit attempt encountered a GEOS `TopologyException` when applying `contains` to a drivable-area GeometryCollection whose individually valid source polygons touch or overlap. No completed audit from that attempt is used.

The final implementation evaluates strict containment and polyline intersection on each unmodified valid source polygon component and uses boolean OR per entity. Filled geometry distances remain the minimum distance over the original components. No union, buffer, geometry repair, threshold change, new semantic label or selector change is introduced. A boundary point is counted inside only if at least one original component strictly contains it. The full 200-window audit is rerun after this correction.

Additional implementation corrections before finalization: explicit NumPy index arrays preserve the batch dimension for single-target packs; the local process uses the historical CuBLAS deterministic workspace setting; neighbor counts are cast from int8 to int32 before multiplying by36 for mode-mode edge accounting. The latter changes reporting arithmetic only, while the actual interaction graph remains the frozen Stage8A-0 implementation. All final tables are regenerated after the arithmetic correction.
