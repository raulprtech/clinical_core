# TCGA-LUAD postoperative OS adapter

This adapter implements human-approved Nigma plan `8fc10f9f9062185cd985d4ee529b4291f15a` with digest `907581ea5e85b1f0c7cddc638160c63d229e581b4d91e83536f2bffe4b21c853`.

The target is TCGA-LUAD only. TCGA-LUSC is explicitly excluded. The prediction time is post-surgery and the endpoint is overall survival.

The first contract reuses demographic and pathological variables with plausible cross-disease semantics. Renal laboratory variables remain excluded until a LUAD-specific availability and temporal audit passes.

The adapter recognizes current GDC follow-up names `days_to_last_follow_up` and `days_to_follow_up`, plus the legacy Clinical-Core alias `days_to_last_followup`. Deaths use `days_to_death`; censored cases use an available follow-up value.

`pack_years_smoked` and `cigarettes_per_day` are numeric LUAD candidates. `tobacco_smoking_status` and `asbestos_exposure_type` remain audit candidates because their current dictionary values and project coverage require explicit harmonization.

The experiment configuration is intentionally non-executable. It points to the expected local reference path but every training phase remains disabled until TCGA-LUAD clinical data is registered and inspected. Renal data must never be used to claim a pulmonary execution.

Evidence:

- https://docs.gdc.cancer.gov/API/Users_Guide/Appendix_A_Available_Fields/
- https://docs.gdc.cancer.gov/Data_Dictionary/Release_Notes/Data_Dictionary_Release_Notes/
- https://docs.gdc.cancer.gov/API/Release_Notes/API_Release_Notes/


## Validation checkpoint

Local validation passed 5 LUAD-specific contract tests and 21 existing Clinical-Core regression tests. Nigma accepted the immutable plan, approval, base commit, change scope, schema, prediction-time and leakage evidence.

Nigma returned `requires_correction` with validation digest `0986251230a2094537c54def8abad5fc54afa2a488f02bbf7c8276687a250a72` only because the TCGA-LUAD patient-level source is not present locally and clean-environment reproduction over the real cohort has therefore not been claimed.
