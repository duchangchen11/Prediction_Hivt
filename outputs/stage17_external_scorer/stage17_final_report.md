# Stage17 — final report

**COMPLETE.** Three genuine TNT scoring components fitted on frozen Stage15B candidates; no HiVT training, no G-C alteration, no Outer tuning. G-C has lower pooled Top1FDE than the adapted TNT scorer.

| Item | Result |
|---|---|
| Branch | stage17/external-trajectory-scorer |
| Base commit | e6682e4855d99eb598a3a40d117a81a41448e940 |
| Source | unofficial Henry1iu TNT, bcbccdc1d35a717793e3caa1d599c1f700612227; actual component classes and CE loss |
| Adapter / initialization / GT isolation | PASS; frozen64D observed context; fresh16,001-param scorers |
| Training / three-fold status | COMPLETE / YES;0.141809 scorer GPU hours;0 HiVT training steps |
| Adapted TNT / G-C / NG-C Overall FDE | 2.421995 / 2.261161 / 2.306543 m |
| G-C−TNT /95% descriptive CI | -0.160834 m /[-0.191230,-0.133289] |
| Analysis classification | supplementary, outside unchanged Stage15B confirmatory family3 |
| Shared candidates / historical preservation | bitwise candidate/logit/oracle PASS;3882 historical files plus1971 candidate/context files verified |
| Paper table / figure | READY;8-model source table and comparison figure; SVG/PDF/PNG |
| TNT reproduction / licensing | scoring adaptation only; full system not reproduced; no upstream license grant found, no third-party source redistributed |
| Next action | STOP; await brain-AI review; no automatic network innovation experiment |

Deliverables: [source audit](stage17_baseline_source_audit.md), [adapter design](stage17_adapter_design.md), [integrity](stage17_integrity_checks.md), [training](stage17_training_report.md), [eight-model comparison](stage17_external_comparison.csv), [paired bootstrap](stage17_bootstrap_comparison.csv), [efficiency](stage17_efficiency_report.md), [scientific conclusion](stage17_scientific_conclusion.md), [main table source](stage17_main_performance_table.csv), [figure exports](08_figures/).

Use **“Adapted TNT Scoring”** in the paper and cite the original paper plus pinned unofficial implementation. All estimates are custom internal scene-isolated CV on630 scenes /260,151 full-horizon actor-windows, not official nuScenes test performance. Context and Bicycle-route information differences, CE/BCE source discrepancy and prior scene exposure are disclosed in the audits. Report both favorable and unfavorable class/fold differences from the CSV; do not retune G-C.

The immutable public preregistration commit is `31594289ee02ece081b013d954741da8c0e97247`. The final Git commit is the commit containing this report, available in Git history and the final user receipt; the report does not embed its own self-referential commit hash.
