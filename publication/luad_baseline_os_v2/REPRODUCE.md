# Clean reproduction of the TCGA-LUAD baseline

## Scope

This procedure reproduces the active internal TCGA-LUAD clinicopathologic
baseline. It does not constitute external validation or demonstrate clinical
utility. The frozen source contains open-access GDC Clinical Supplement BCR XML
files and the tracked inventory contains file UUIDs, byte sizes and MD5 hashes,
but no filenames, patient barcodes, extracted rows or predictions.

## Requirements

- Linux or WSL with Python 3.12 and network access to api.gdc.cancer.gov.
- Approximately 6 GB free for the isolated Python environment. The verified
  Linux PyTorch wheel currently brings CUDA runtime packages even for this
  CPU-only baseline; reducing that footprint is tracked as packaging debt.
- Approximately 50 MB free for source XML and aggregate results.

## Procedure

```bash
git clone https://github.com/raulprtech/clinical_core.git
cd clinical_core
git switch codex/nigma-controlled-dry-run

python3 -m venv .venv-luad
.venv-luad/bin/python -m pip install --upgrade pip
.venv-luad/bin/python -m pip install -r requirements-luad.lock
.venv-luad/bin/python -m unittest discover -s tests -p 'test_*.py' -q
.venv-luad/bin/python -m unittest discover -s code/tests -p 'test_*.py' -q

.venv-luad/bin/python code/tools/restore_luad_source.py
.venv-luad/bin/python code/tools/verify_luad_publication_package.py \
  --require-data
.venv-luad/bin/python code/core/experiment_runner.py \
  code/experiments/experiment_config_nigma_luad_baseline_os_v2.yaml
```

The runner prints the new run directory. Compare its summary against the frozen
aggregate values:

```bash
.venv-luad/bin/python code/tools/compare_luad_reproduction.py \
  results_nigma_dry_run/luad_baseline_os_v2/RUN_ID/summary.json
```

A successful comparison must pass exact cohort counts, absence of runtime
errors, all three holdout C-indices, all three repeated-CV means and fold counts,
and all temporal Harrell/IPCW C-indices within the manifest tolerance.

## Small restoration smoke test

To test only network access, atomic installation and checksums:

```bash
.venv-luad/bin/python code/tools/restore_luad_source.py \
  --output-dir /tmp/clinical-core-luad-smoke --limit 1
```

## Expected aggregate outcomes

- 522 parsed XML files.
- 507 survival-eligible cases and 122 eligible events.
- Holdout C-index: Stage 0.6801, TNM 0.6898, Stage+TNM 0.6905.
- Repeated five-fold CV over three seeds: Stage 0.7142, TNM 0.7004,
  Stage+TNM 0.7121.
- Internal temporal transport Stage+TNM C-index 0.720186 and IPCW C-index
  0.723896.

If the source endpoint no longer serves one of the frozen UUIDs, report the
missing UUID and do not silently substitute a newer file. A changed GDC cohort
requires a new source inventory, experiment identifier and approval.
