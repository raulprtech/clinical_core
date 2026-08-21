# Data retention for the publication-focused LUAD line

The active publication baseline is tabular TCGA-LUAD. It does not consume the
TCGA-KIRC DICOM collection, STU-Net inputs, vision intermediates or downloaded
vision model weights. Unit and contract tests use synthetic temporary inputs.

## Retain locally

- data/raw/TCGA-LUAD and SOURCE.json (~23 MB): exact source for the active LUAD
  reproduction.
- data/raw/clinicalsupplement (~22 MB): inexpensive KIRC tabular baseline.
- data/manifests (~2 MB): selection and provenance needed to regenerate retired
  inputs.
- results_nigma_dry_run/luad_baseline_os_v2 (~700 KB): aggregate LUAD evidence.
- Compact CSV, JSON and NPZ summaries needed for a future vision comparison.

## Regenerable cold data

- data/raw/tcia_kirc_dicom: public TCIA input selected by
  code/tools/download_tcia_kirc.py and data/manifests/tcia_kirc.
- STU-Net and TurboConv input volumes, preprocessing caches and per-case logits
  beneath data/embeddings/vision.
- data/models: downloaded public weights and a source checkout.

Before removing cold data, retain the manifests, aggregate summaries and
provenance sidecars. Restoration requires network bandwidth and compute, but
not undocumented patient-level state. Data removal does not imply that the
corresponding historical experiment has been reproduced from a clean clone.

The Python .venv is not a dataset. Rebuilding it from requirements-luad.lock
provides the publication route; vision work should use a separate environment.

The canonical aggregate evidence and experiment index live under
publication/luad_baseline_os_v2. They deliberately contain no patient rows,
case identifiers, predictions or model artifacts.
