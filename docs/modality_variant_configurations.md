# Portable TCGA-KIRC modality configurations

Date: 2026-08-24

Clinical-Core exposes all seven non-empty combinations of tabular, text and
vision as exact, CPU-compatible Phase 5 configurations for post-surgery
overall-survival evaluation:

| Catalog variant | Exact modalities | Configuration | Minimum RAM |
| --- | --- | --- | ---: |
| `tcga-kirc-tabular-only-config@1.0.0` | tabular | `experiment_config_kirc_tabular_only_portable_v1.yaml` | 4 GB |
| `tcga-kirc-text-only-config@1.0.0` | text | `experiment_config_kirc_text_only_portable_v1.yaml` | 4 GB |
| `tcga-kirc-vision-only-config@1.0.0` | vision | `experiment_config_kirc_vision_only_portable_v1.yaml` | 6 GB |
| `tcga-kirc-tabular-text-config@1.0.0` | tabular + text | `experiment_config_kirc_tabular_text_portable_v1.yaml` | 6 GB |
| `tcga-kirc-tabular-vision-config@1.0.0` | tabular + vision | `experiment_config_kirc_tabular_vision_portable_v1.yaml` | 8 GB |
| `tcga-kirc-text-vision-config@1.0.0` | text + vision | `experiment_config_kirc_text_vision_portable_v1.yaml` | 8 GB |
| `tcga-kirc-trimodal-config@1.0.0` | tabular + text + vision | `experiment_config_kirc_trimodal_portable_v1.yaml` | 10 GB |

All paths are portable. Set:

```bash
export CLINICAL_CORE_DATA_ROOT=/absolute/path/to/clinical-core-data
export CLINICAL_CORE_OUTPUT_ROOT=/absolute/path/to/clinical-core-results
```

The data root must contain the clinical supplement and the exact precomputed
embedding files named in the selected YAML. Run only after separate dataset and
scientific-execution approval:

```bash
python code/core/experiment_runner.py \
  code/experiments/experiment_config_kirc_text_only_portable_v1.yaml
```

Replace the YAML with any of the other six exact profiles as required.

## Safety and evidence boundary

The runner fails before modelling when a modality declared in
`required_precomputed_modalities` is disabled, unsupported, unconfigured, or
its cache is missing. It does not silently substitute mock or zero embeddings.
Each profile fixes one exact ablation, disables raw extraction output and saves
the resolved configuration.

## Local reproducibility result

All seven profiles were executed twice after repairing fold-level model seeding.
Each pair produced a byte-identical aggregate CSV:

| Modalities | Cases | C-index mean ± SD |
| --- | ---: | ---: |
| tabular | 451 | 0.7660 ± 0.0063 |
| text | 441 | 0.6610 ± 0.0224 |
| vision | 214 | 0.6629 ± 0.0079 |
| tabular + text | 441 | 0.7354 ± 0.0072 |
| tabular + vision | 214 | 0.7197 ± 0.0230 |
| text + vision | 210 | 0.6657 ± 0.0111 |
| tabular + text + vision | 210 | 0.7093 ± 0.0160 |

The first pre-repair repetition failed determinism for 7/7 profiles because
the Cox head used random Xavier initialization before a fold-derived Torch seed
was set. The repair derives model initialization from `seed * 1000 + fold`;
a focused regression and the two full real passes verify the result. Exact
metrics, input/code hashes, aggregate hashes and claim bounds are sealed in
`publication/kirc_modality_profiles_v1/aggregate_validation.json`.

The seven catalog entries remain `configuration_ready`, not `prepared`,
pending an explicit promotion decision and reproduction from a clean
installation. They now carry local internal reproduction evidence for every
exact modality set. None supports external validation, clinical utility,
prospective use, or deployment claims.

Clinical-Nigma selects an exact modality set by default and carries the
configuration hash and required environment-variable names into its sealed
coding-agent handoff. It does not run these profiles or authorize data access.
