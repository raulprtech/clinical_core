# CLINICAL-CORE Architecture

The project follows a layered design to ensure that research components (models) are decoupled from the infrastructure (orchestration).

## The Four Layers

### 1. Ingestion Layer (Connectors)
**Path:** `code/components/connectors/`

Connectors are responsible for transforming raw clinical modalities into standard 768-dimensional embeddings + a confidence score (0.0 to 1.0).

| Component | Responsibility | Examples |
|-----------|----------------|----------|
| **Tabular** | Imputation & Scaling | TabPFN, Cox-Baseline |
| **Vision** | Segmentation & Radiomics | STU-Net, TotalSegmentator |
| **Text** | Extraction & Embedding | ClinicalBERT, Docling |

**The Contract:** Every connector MUST satisfy the `encode()` method:
- **Input:** Path to data or raw features.
- **Output:** `(torch.Tensor[768], float)`

### 2. Reasoning Layer (Processors)
**Path:** `code/components/processors/`

Processors handle cross-modal interaction and final prediction.

- **Fusion**: Merges N embeddings into a single fused representation.
- **Prognosis**: Maps the fused representation to a survival risk score or probability.

### 3. Registry (The Master of Keys)
**Path:** `code/registry.py`

A central registry that maps string names in YAML configs to Python classes. This allows swapping any logic without modifying the core `main.py` pipeline.

### 4. Orchestration Layer
**Path:** `code/main.py`

Discovers data, applies the requested connectors, caches embeddings, and runs the ablation studies/experiments. It is strictly config-driven.

## Data Flow

1.  **Discovery**: Identify matching XMLs, PDFs, and NIfTI files per subject.
2.  **Encoding**: Modalities are processed in parallel (or cached) into embeddings.
3.  **Fusion**: Embeddings are combined based on the experiment's ablation plan.
4.  **Prediction**: The prognosis head estimates hazard/risk.
5.  **Validation**: Metrics (C-index) are computed typically via cross-validation.
