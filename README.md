# CLINICAL-CORE / RENAL-CORE (Modular Modality Architecture)

## Active publication baseline: TCGA-LUAD

The active publication-focused line is the aggregate-only TCGA-LUAD
clinicopathologic overall-survival baseline. It is distinct from the historical
TCGA-KIRC multimodal research described below.

Reproduce the active line from the repository root:

```bash
python code/core/experiment_runner.py \
  code/experiments/experiment_config_nigma_luad_baseline_os_v2.yaml
```

Verify the tracked publication package plus the retained local source and
canonical run:

```bash
python code/tools/verify_luad_publication_package.py \
  --require-data --require-run
```

The package index, aggregate summary and claim bounds are in
`publication/luad_baseline_os_v2/`. The exact lightweight environment is
`requirements-luad.lock`; the retention and restoration boundary is described
in `docs/data_retention_policy.md`. External validation of the current line,
clinical utility and prospective performance remain explicitly unclaimed.

## Prepared variant registry

Prepared and frozen Clinical-Core variants are declared in
[`variants/registry-v1.json`](variants/registry-v1.json). The catalog seals the
source revision, entrypoints, evidence, artifact hashes, resource needs,
scientific claim limits and downstream approvals. It contains no patient rows
or predictions.

The catalog also exposes all seven portable, configuration-ready TCGA-KIRC
combinations of tabular, text and vision. Their setup, required embedding
caches, aggregate validation and evidence limits are documented in
[`docs/modality_variant_configurations.md`](docs/modality_variant_configurations.md).

The registry is passive: Clinical-Core does not select, approve or execute a
variant from a request. Deterministic compatibility assessment and explanation
belong to Clinical-Nigma; data access and scientific execution remain separate
human approval gates.

### Use with Clinical-Nigma

Keep `Clinical-Nigma` and `clinical_core` as sibling checkouts. Start the
Clinical-Nigma API and follow its README section **Clinical-Core integration**.
That workflow builds a bounded request from this registry, selects an exact
modality profile, requires an API-only human approval and emits an immutable
coding-agent handoff.

The handoff returns a repository-relative `configuration_ref.path`, its
SHA-256, the sealed Clinical-Core source revision and required runtime inputs.
It does not grant access to datasets or permission to run an experiment. After
those permissions are separately approved, run the returned KIRC configuration
from this repository:

```bash
export CLINICAL_CORE_DATA_ROOT=/absolute/path/to/authorized/kirc-data
export CLINICAL_CORE_OUTPUT_ROOT=/absolute/path/to/new/output-directory
python code/core/experiment_runner.py \
  code/experiments/experiment_config_kirc_tabular_text_portable_v1.yaml
```

The path above is only the tabular+text example. Always use the exact
`configuration_ref.path` selected by Clinical-Nigma, verify its hash first and
write outputs outside Nigma. Clinical-Core remains the scientific executor;
Clinical-Nigma remains the deterministic selector, approval gate and provenance
handoff layer.

End-to-end multimodal ecosystem for CLINICAL-CORE, validated on TCGA-KIRC. This repository implements a modular, modality-centric structure based on **Hexagonal Architecture** principles, where logic and models are isolated from external interfaces via **Adapters**.

```mermaid
graph TD
    A[TABULAR Adapter] --> D[FUSION]
    B[TEXT Adapter] --> D
    C[VISION Adapter] --> D[FUSION]
    D --> E[PROGNOSIS]
    E --> F[C-index / Survival Risk]
```

## The Three Rules of the Ecosystem

1.  **Declarative configs**: Every decision lives in configuration YAMLs. No hardcoding in Python.
2.  **Structured provenance**: Every run is logged in a unique timestamped directory with a copy of its config.
3.  **Component Modularity**: Each modality (Tabular, Vision, Text) is self-contained with its own models and tools.

## File Structure

The project is organized into a modular hierarchy:

```
code/
├── main.py                     # Entry point shim
├── core/                       # 🏗️ Core Orchestration Layer (Domain/Ports)
│   ├── main.py                 # Multi-modal pipeline logic
│   ├── registry.py             # Master component registry
│   ├── experiment_runner.py    # Execution engine
│   └── model_utils.py          # Shared ML utilities
│
├── components/                 # 🧩 Modular Components (Hexagonal Expansion)
│   ├── adapters/               # 🔌 External Interface Handlers
│   │   ├── ingestion/          # e.g., tabular, vision, text modalities
│   │   │   ├── models/         # 🧠 Internal architectures (Encoders, Segmenters)
│   │   │   ├── utils/          # 🛠️ Component-specific tools (Sweeps, Extractors)
│   │   │   ├── configs/        # 📋 Component mapping schemas
│   │   │   └── experiments/    # ⚙️ Modality-specific experiment YAMLs
│   │   ├── channels/           # Future integrations
│   │   └── bridges/            # Future integrations
│   │
│   ├── procesors/              # ⚙️ Core processing and task logic
│   │   ├── fusion/             # Concatenation and aggregation strategies
│   │   └── prognosis/          # Linear Cox prediction
│   │
│   ├── explainers/             # 🔍 Interpretability (e.g. GraphRAG)
│   ├── monitors/               # 📊 System monitoring
│   └── external/               # 🏛️ Non-compliant baselines for reference
│
├── experiments/                # ⚙️ Global Experiment Configs
```

## Quick Start

1.  **Configure**: select your components in `code/experiments/experiment_config.yaml`.
2.  **Run**:
    ```bash
    python3 code/main.py --config experiments/experiment_config.yaml
    ```
3.  **Inspect**: Results are stored in `results/{run_id}/`.

## Multi-Modal Adapters

### VISION
- **`stunet`**: Uses STU-Net (SOTA medical segmentation) and TotalSegmentator.
- **`vision_resnet18_2d`**: Frozen ResNet18 over three central anatomical views.
- **`vision_resnet50_2d`**: Frozen ResNet50 with a fixed leakage-safe 768D projection.
- **`vision_resnet18_2p5d`**: Frozen ResNet18 with neighboring slices as RGB context.
- **`mock`**: Architectural validation with synthetic masks.

See [VISION-L0 integration](docs/vision_resnet_integration.md) for raw DICOM/NIfTI
and precomputed-Colab workflows.
See [ResNet18 sequence Fast Proof](docs/resnet18_sequence_mamba_fastproof.md)
for the paired baseline, attention-pooling and Mamba survival benchmark.
The completed exploratory run is reported in
[ResNet18 sequence preliminary results](docs/resnet18_sequence_mamba_preliminary_results.md).
The symmetric nested repeated-CV follow-up is reported in
[ResNet18 sequence confirmatory results](docs/resnet18_sequence_mamba_confirmatory_results.md).
The append-only experiment history and review checklist are maintained in the
[research decision log](docs/research_decision_log.md).
The Mamba risk integration is summarized in [trimodal Mamba fusion results](docs/trimodal_mamba_fusion_results.md).
The repeated-CV correction is in [trimodal sequence nested-CV results](docs/trimodal_sequence_nested_cv_results.md).
The post-hoc training-stability analysis is in [Mamba epoch stability](docs/mamba_epoch_stability_diagnostic.md).
The paired architecture/token/position study is in
[sequence factorial ablation](docs/sequence_factorial_ablation_results.md).
The shared-weight directionality test is reported in
[Mamba bidirectional ablation](docs/mamba_bidirectional_ablation_results.md).
The post-hoc complementarity check is in
[fixed sequence ensemble](docs/fixed_sequence_ensemble_results.md).
The leakage-safe train-derived scaling follow-up is in
[train-scaled sequence ensemble](docs/train_scaled_sequence_ensemble_results.md).

For the complete TCGA-KIRC download, embedding-cache and three-modality run,
see [trimodal data preparation](docs/trimodal_data_preparation.md).

### TEXT
- **`clinicalbert`**: Docling extraction + ClinicalBERT embeddings.

### TABULAR
- **`cox_baseline`**: Cox Proportional Hazards baseline.
- **`tabpfn`**: Large In-Context Learning for tabular data.
- **`linear_compact`**: Resource-efficient linear encoder (formerly linear_fpga).

## Adding Components

1.  Implement your class in `code/components/<modality>/models/`.
2.  Register it in `code/core/registry.py`.
3.  Update your `.yaml` experiment config.

## What's New (v5 Refactor: Modality Centric)

The architecture has transitioned from a file-type grouping to **Modality-Centric Modularity**:
- **Hexagonal Alignment**: External interfaces are now strictly treated as **Adapters**, isolating the core domain logic.
- **Granular Components**: Each modality now explicitly separates its `models/` from its `utils/`.
- **Centralized Core**: Orchestrators moved to `core/` to leave the root clean.
- **Lazy Loading**: Heavy models (BERT, STU-Net) now use lazy initialization to speed up registry lookups.
