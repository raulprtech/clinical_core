# KIRC integration review — 2026-09-04

Integrates research head `418d923` (21 commits after `1d6f530`) through a
consolidated integration branch. The original research branch remains for
protocol chronology and catalog revision anchors. This is software integration,
not new scientific validation.

Review covers shared runner changes, path resolution, deterministic model
initialization, synthetic sequence/pooling and nested-CV contracts, catalog
integrity, LUAD regression, and result artifact boundaries.

Two newly tracked STU-Net files contained individual cohorts and predictions:
`cohort_complete.csv` and `outer_predictions.csv` under
`results_vision/stunet_volumetric_pooling_nested_75/`. They are excluded from
this integration and ignored. Original local research copies remain. The
source branch still contains these files in its history; no remote history
purge or visibility change is part of this integration.

The result checker rejects patient identifier columns and nested fields, and
known TCGA/CPTAC identifiers in added or modified result/publication files.
Historical artifacts already in master are outside this differential check
and have not been certified as aggregate-only.

KIRC CI runs CPU synthetic contracts, both function-style catalog checks, and
the LUAD integrity verifier with the existing CPU dependency lock. It does not
retrain the reported models or establish external performance. The existing
configuration-ready variants retain their scientific status.

Internal repeated CV does not demonstrate a stable multimodal advantage over
tabular. STU-Net moments improve the tested visual representation, without
establishing multimodal superiority. These interpretations remain unchanged.
