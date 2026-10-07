# Execution notes

The initial serial implementation passed exact Stage8A-0 lane-cache checks but was slow. An attempted shared-geometry thread implementation terminated with native GEOS segmentation faults (exit 139); it produced no complete result or readiness decision. It was replaced by independent spawned processes, each constructing its own map index and GEOS objects. Prepared geometries are used only within their owning process.

The final complete run recomputes and overwrites every batch. It verifies all original source and historical graph-cache SHA256 values, compares lane token IDs and distances bitwise, and only finalizes the manifest after all 1283 batches and 200 integrity windows pass. No partial-run results are used in final population tables. Retrieval radius, polygon rule, semantic attributes and entity taxonomy are unchanged.
