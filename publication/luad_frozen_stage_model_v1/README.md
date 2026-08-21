# Frozen TCGA-LUAD stage Cox model

This directory contains the portable, full-development model required before
an approved external transport study. It is downstream of the immutable
`tcga-luad-baseline-os-v2` evaluation tag and does not alter that baseline.

The JSON artifact contains aggregate model parameters only: feature and
endpoint contracts, imputation and scaling parameters, Cox coefficients,
centering, and baseline cumulative hazard. It contains no patient identifiers,
rows, or per-patient predictions. Its in-sample C-index is descriptive and is
not evidence of validation.

Rebuild twice and compare bytes:

```bash
python code/tools/export_luad_frozen_model.py --output /tmp/model-a.json
python code/tools/export_luad_frozen_model.py --output /tmp/model-b.json
cmp /tmp/model-a.json /tmp/model-b.json
```

External application remains prohibited until the institutional data-access
approval, aggregate preflight, frozen cohort definition, and explicit execution
approval documented by Clinical-Nigma are all satisfied.
