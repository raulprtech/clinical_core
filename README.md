# CLINICAL-CORE / RENAL-CORE (Multimodal Baseline)

End-to-end multimodal baseline for the first-quarter implementation of CLINICAL-CORE, validated on TCGA-KIRC. Implements the full pipeline:

```
TABULAR-CONN ─┐
TEXT-CONN ────┼──→ FUSION-PROC ──→ PROGNOSIS-PROC ──→ C-index
VISION-CONN ──┘
```

This version is designed as infrastructure-seed for a future auto-improvement lab. It respects three rules that cost zero now and save weeks later.

## The three rules

**1. Declarative configs.** Every decision about what runs, with what parameters, on what data, lives in `experiment_config.yaml`. Zero hardcoding in Python. To change experiment behavior, edit the YAML — never a `.py`.

**2. Structured outputs with provenance.** Each run creates a unique directory `{output_base}/{timestamp}_{config_hash}/` containing an exact copy of the config used, all metric CSVs, and a `run_metadata.json` with environment info. The config hash uniquely identifies which result came from which configuration.

**3. Swaps via config only.** Adding a new component (variant, imputation, text encoder, vision backend, fusion strategy, prognosis model) requires two changes: registering it in `registry.py` and listing it in the config. Zero modifications to `experiment_runner.py` or `multimodal_pipeline.py`.

## File structure

```
config_v2.yaml             # Clinical feature/target schema (what to extract from XML)
experiment_config.yaml     # Declarative experiment definition (what to run)
registry.py                # Registry of swappable components (6 categories)

# CONNs (ingestion)
extractor.py               # TCGA-KIRC XML parser (TABULAR raw extraction)
imputation_benchmark.py    # Imputation strategies + preprocessor
encoders.py                # TABULAR-CONN encoder variants (Cox / TabPFN / Linear)
text_conn.py               # TEXT-CONN (Docling + ClinicalBERT) 
vision_conn.py             # VISION-CONN (TotalSegmentator/STU-Net + radiomics)

# PROCs (fusion + prediction)
fusion_proc.py             # FUSION-PROC (confidence-weighted concatenation)
prognosis_proc.py          # PROGNOSIS-PROC (linear Cox-PH on fused embedding)

# Orchestration
multimodal_pipeline.py     # End-to-end pipeline + ablation runner
experiment_runner.py       # Config-driven phase orchestrator
run_colab.ipynb            # Colab execution notebook

# Standalone scripts (not part of experiment_runner)
finetune_vision_kits23.py  # Optional: fine-tune STU-Net for tumor segmentation
```

## Five experimental phases

The runner executes up to 5 phases, each independently enableable in the config:

| Phase | Purpose | Output |
|-------|---------|--------|
| 1 | Imputation benchmark (K-S + downstream C-index with fixed Cox) | `phase1_imputation.csv` |
| 2 | TABULAR-CONN variant comparison (3 variants) | `phase2_variants.csv` |
| 3 | Efficiency benchmark (latency, memory, FLOPs) | `phase3_efficiency.csv` |
| 4 | Stress test (20% noise injection + outliers) | `phase4_stress.csv` |
| 5 | **Multimodal end-to-end ablation** | `phase5_multimodal_ablation.csv` |

Phase 5 is the new addition in v4. It runs the full multimodal pipeline and reports C-index for several modality combinations:
- `[tabular]` (floor)
- `[tabular, text]`
- `[tabular, vision]`
- `[tabular, text, vision]` (full)

## VISION-CONN backends

The vision connector has three backends, selected via `vision_backend` in the config:

**`totalseg`** (recommended for the first run). Uses the [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) python package, which wraps nnU-Net pretrained on >1000 CT scans annotated with 104 anatomical structures including kidney. Single pip install:
```bash
pip install totalsegmentator nibabel
```
Weights download automatically on first inference. This is the path that works out of the box.

**`stunet`**. Uses STU-Net (Huang et al. 2023) checkpoints. Requires manual setup:
1. Install nnUNetv2: `pip install nnunetv2`
2. Clone STU-Net: `git clone https://github.com/uni-medical/STU-Net`
3. Patch the STUNetTrainer files into your nnunetv2 installation following the instructions in the STU-Net repo
4. Download a pretrained checkpoint (B/L/H) from the STU-Net release page
5. Set environment variable: `export CLINICAL_CORE_STUNET_CHECKPOINT=/path/to/checkpoint`
6. Complete the `_run_inference` method in `vision_conn.py:STUNetSegmenter` (it's a single nnUNetv2 predictor call once the trainer is registered — see the docstring for the exact code shape)

The scaffold is in place; the integration point is one method.

**`mock`**. Synthetic ellipsoid mask centered in the volume. Used automatically when neither real backend is available. Allows architectural validation but produces meaningless C-index values — do not report.

**`auto`** (default). Tries `totalseg` → `stunet` → `mock` in order.

## Tumor segmentation (separate from baseline)

The baseline does NOT segment tumor explicitly. It segments the whole kidney as ROI and extracts radiomics features from the organ. Tumor-specific prognostic information enters via TABULAR-CONN (TNM, grade) and TEXT-CONN (pathology report).

To add explicit tumor segmentation, run `finetune_vision_kits23.py` separately. This is a one-time fine-tuning of STU-Net on KiTS23 (which has tumor labels). Requires GPU and is independent of the experiment runner.

```bash
# Verify environment first
python finetune_vision_kits23.py --check_only

# Launch fine-tuning (requires manual KiTS23 setup in nnUNetv2 format)
python finetune_vision_kits23.py \
    --pretrained /path/to/stunet_b.pth \
    --dataset_id 220 \
    --epochs 100
```

## Execution in Colab (4 steps)

1. Upload the 13 source files to `/MyDrive/data_tesis/code/`
2. Place TCGA-KIRC XMLs in `/MyDrive/data_tesis/clinicalsupplement/`
3. (Optional) Place pathology PDFs and CT volumes in their respective directories and update `text_dir` / `vision_dir` in the config
4. Open `run_colab.ipynb` and execute all cells

For the very first run, the recommended config is:
- Phases 1, 2, 3, 4 enabled (the original tabular-only protocol)
- Phase 5 enabled with `modalities: [tabular]` and `text_dir: null`, `vision_dir: null`

This validates the infrastructure end-to-end without requiring text or vision data. Once you have those data sources, point the config to them and rerun — without touching any code.

## Adding a new component (worked example)

Suppose you want to add a new fusion strategy called `attention_fusion`.

**Step 1.** Implement the class in a new file `fusion_proc_attention.py`:
```python
class FusionProc_Attention:
    name = "fusion_attention"
    def __init__(self, modalities, modality_dims=None, **kwargs):
        ...
    def fuse_one(self, modality_outputs):
        ...
        return fused_embedding, aggregate_confidence
```

**Step 2.** Register in `registry.py`:
```python
from fusion_proc_attention import FusionProc_Attention

FUSION_PROC_REGISTRY = {
    'fusion_baseline_concat': ...,
    'fusion_attention': lambda modalities, modality_dims=None, **kw: FusionProc_Attention(
        modalities=modalities, modality_dims=modality_dims, **kw
    ),
}
```

**Step 3.** Activate in `experiment_config.yaml`:
```yaml
phase_5_multimodal:
  fusion_proc: "fusion_attention"
```

That's it. The runner picks it up via the registry. No changes to orchestration code.

## What changed from v3

| Aspect | v3 | v4 |
|--------|----|----|
| Modalities | TABULAR only | TABULAR + TEXT + VISION |
| Components | 2 categories (imputation, variants) | 6 categories (+ text_conn, vision_conn, fusion_proc, prognosis_proc) |
| End-to-end | Not yet possible | Phase 5 runs the full pipeline + ablations |
| Vision | N/A | Three backends (totalseg, stunet, mock) |
| Caching | N/A | Per-modality embedding cache for efficient ablations |
| Provenance | Already had it | Same — every run has unique directory + config hash |

## Notes for the future you

When Tournament-CORE arrives, the only thing the automated system needs to do is generate new `experiment_config.yaml` files, run `run_experiment(path)`, and read the `summary.json`. The architecture is already prepared for that transition. Don't rewrite this when you get there — just add an agent on top.

The integration points where automation will plug in are:
- `registry.py` for registering newly-generated components
- `experiment_config.yaml` as the action space the agent searches over
- `summary.json` as the reward signal for whatever optimizer the agent uses
