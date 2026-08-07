# TCGA-LUAD postoperative OS adapter

This adapter implements human-approved Nigma plan `8fc10f9f9062185cd985d4ee529b4291f15a` with digest `907581ea5e85b1f0c7cddc638160c63d229e581b4d91e83536f2bffe4b21c853`.

The target is TCGA-LUAD only. TCGA-LUSC is explicitly excluded. The prediction time is post-surgery and the endpoint is overall survival.

The contract reuses demographic and pathological variables with plausible cross-disease semantics. Renal laboratory variables remain excluded after the LUAD-specific availability audit.

The adapter recognizes current GDC follow-up names `days_to_last_follow_up` and `days_to_follow_up`, plus the legacy Clinical-Core alias `days_to_last_followup`. Deaths use `days_to_death`; censored cases use an available follow-up value.

`pack_years_smoked` remains a numeric LUAD feature after 68.2% observed coverage. `cigarettes_per_day` was removed after 0% observed coverage. `tobacco_smoking_status` remains excluded because the XML uses legacy codes 1–5 while the current GDC dictionary uses descriptive values. `asbestos_exposure_type` also remains an audit candidate.

## Real source checkpoint

With explicit user authorization, 522 open-access TCGA-LUAD patient clinical XML files were downloaded from GDC. Every file was checked against its published MD5 before atomic installation under the ignored local `data/` tree. No patient-level file is tracked by Git or copied into Nigma.

The aggregate-only extraction produced:

- 522 parsed cases and zero XML parsing failures;
- 494 cases with usable overall-survival time (94.6%);
- 127 death events (24.3%);
- median observed follow-up of 255.5 days;
- nine retained features;
- no patient-level output artifacts.

The experiment configuration now seals the authorized local holdout: only phase_2_holdout is enabled, with seed 42, an 80/20 split, cox_baseline, no raw extraction persistence and no model artifacts. Renal data must never be used to claim a pulmonary execution.

## Authorized Cox holdout

The single authorized run used 395 training cases and 99 holdout cases, including 24 holdout events. The Cox baseline obtained C-index 0.6390, ECE 0.2576 and Brier score 0.25; the ingestion contract was satisfied and the run reported zero errors.

Exactly six allowlisted aggregate files were produced. Nigma found no nested directories, oversized files or TCGA patient identifiers, and no model artifacts were saved.

## Validation checkpoint

Local validation passes 5 LUAD-specific contract tests and 21 existing Clinical-Core regression tests. Nigma accepts the immutable plan, approval, base commit, change scope, schema, prediction-time and leakage evidence.

Nigma previously returned requires_correction because the source was absent. The authorized holdout has now completed; only a second execution from a fresh environment remains unclaimed.

Evidence:

- https://docs.gdc.cancer.gov/API/Users_Guide/Appendix_A_Available_Fields/
- https://docs.gdc.cancer.gov/Data_Dictionary/Release_Notes/Data_Dictionary_Release_Notes/
- https://docs.gdc.cancer.gov/API/Release_Notes/API_Release_Notes/
- https://docs.gdc.cancer.gov/API/Users_Guide/Downloading_Files/
