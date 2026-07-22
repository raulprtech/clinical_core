# Anti-leakage protocol for survival prediction on TCGA-KIRC

Reproduction code for the manuscript on three-layer anti-leakage protocol for survival prediction in clear-cell renal cell carcinoma. The code reproduces every numerical claim in the paper from the TCGA-KIRC clinical XMLs.

This repository is self-contained: it does **not** depend on any external multimodal pipeline, text encoder, vision encoder, fusion module, or quantization stage. The three tabular models (Cox PH, `linear_compact`, FT-Transformer), the anti-leakage protocol, the imputation strategies restricted to the train pool, the Mahootiha replication, and the bootstrap / calibration / SHAP analyses are all included.

## What's inside

```
leakage-survival-protocol/
├── configs/
│   ├── experiment_config.yaml             # single source of truth
│   ├── tabular_mapping_no_leakage.yaml    # 19-vars  (limpio)
│   ├── tabular_mapping_22_with_leakage.yaml # 22-vars (permisivo)
│   └── tabular_mapping_19_plus_ecog.yaml  # 20-vars  (intermediate)
├── src/
│   ├── runner.py                          # main experiment driver
│   ├── registry.py                        # variant + imputation registries
│   ├── model_utils.py                     # Cox PL loss, training loop
│   ├── models/
│   │   ├── cox_baseline.py                # lifelines Cox PH wrapper
│   │   ├── linear_compact.py              # 2-layer MLP (103K params)
│   │   └── ft_transformer.py              # FT-Transformer (1.49M params)
│   ├── preprocessing/
│   │   ├── extractor.py                   # TCGA BCR-XML parser
│   │   └── imputation.py                  # mean/median, KNN, MICE
│   └── analysis/
│       ├── statistical_tests.py           # bootstrap CI + paired delta test
│       └── calibration.py                 # Breslow + IBS + reliability bins
├── tools/
│   ├── run_significance_tests.py          # post-hoc: bootstrap + paired delta
│   ├── run_calibration.py                 # post-hoc: IBS + curves
│   └── run_shap_attributions.py           # post-hoc: SHAP (Cox + FT-T)
├── data/
│   └── manifests/
│       └── gdc_modality_manifest_TCGA-KIRC_20260528.csv
├── results/                               # populated by runs
├── requirements.txt
└── README.md
```

## Setting up

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`scikit-survival` requires a working C compiler. On Ubuntu: `sudo apt install build-essential`.

## Getting the data

The TCGA-KIRC BCR Clinical Supplement XMLs are available through the GDC portal:

1. Go to https://portal.gdc.cancer.gov/projects/TCGA-KIRC
2. Add to cart: **Data Category = "Clinical"**, **Data Format = "BCR XML"**
3. Download the cart and extract under `data/raw/clinicalsupplement/`.

The included `data/manifests/gdc_modality_manifest_TCGA-KIRC_20260528.csv` records, per case, the presence of the four modalities (WSI, mRNA-seq, miRNA-seq, methylation 450K) needed for the dense-subcohort ablation (paper §4.3). It was generated against the GDC API on 2026-05-28 and is shipped as a fixed snapshot so reviewers can reproduce the n=224 subcohort exactly.

## Reproducing the paper

### Table 1 — sparse cohort (n=444)

```bash
# Edit configs/experiment_config.yaml:
#   cohort_filter.enabled: false   (default)
#   phase_2_holdout.enabled: true
#   phase_2_mahootiha.enabled: false
python -m src.runner configs/experiment_config.yaml
```

### Table 2 — dense cohort (n=224)

```bash
# Edit configs/experiment_config.yaml:
#   cohort_filter.enabled: true
python -m src.runner configs/experiment_config.yaml
```

### Table 3 — Mahootiha replication

```bash
# Edit configs/experiment_config.yaml:
#   phase_2_mahootiha.enabled: true
python -m src.runner configs/experiment_config.yaml
```

Each run writes `results/{timestamp}_{hash}/` containing:

- `phase2_holdout.csv` + `_summary.csv` — per (protocol, seed, variant) C-index, ECE, Brier.
- `phase2_mahootiha.csv` + `_summary.csv` + `_feature_ranking.csv` — per K and per seed.
- `phase2_artifacts/predictions/{protocol}_seed{N}_{variant}.npz` — per-patient risk scores + survival functions (Cox only) for every run.
- `phase2_artifacts/checkpoints/{protocol}_seed{N}_{variant}.pkl` — pickled CoxPHFitter / encoder state dict.
- `cohort_manifest.json` — audit trail of the cohort filter.
- `summary.json`, `run_metadata.json`, `experiment_config.yaml` — provenance.

### Post-hoc analyses

```bash
RUN_DIR=results/<timestamp>_<hash>

# Bootstrap 95% CI per (protocol, seed, variant) and paired bootstrap
# test of permisivo vs limpio for each variant.
python tools/run_significance_tests.py "$RUN_DIR" --n-iter 1000

# Integrated Brier Score (sksurv) + reliability curves at 1/3/5 years.
# For neural variants the Breslow estimator is fit on the saved train
# risk scores.
python tools/run_calibration.py "$RUN_DIR"

# SHAP attributions for Cox (LinearExplainer) and FT-Transformer
# (KernelExplainer, ~10-30 min per protocol on CPU).
python tools/run_shap_attributions.py "$RUN_DIR" \
    --seed 42 --n-explain 100 --n-background 50 --n-iter 100
```

Each tool writes its own CSV + PNG into the same `$RUN_DIR`.

## Protocol summary

Three layers, all enforced by the runner:

1. **Temporal-provenance filter.** The 22 BCR clinical fields are reduced to 19 by excluding `ecog_score`, `karnofsky_score`, `tumor_status` — variables whose values are only known after the follow-up window. Selected from `configs/tabular_mapping_no_leakage.yaml` (limpio protocol). The permisivo protocol keeps all 22 fields for comparison.
2. **Imputation restricted to the train pool.** Every preprocessor (`mean_median`, `knn_5`, `knn_10`, `mice`) is fit on the 80% train set only and applied blindly to the 20% held-out. The held-out is never seen by the imputer.
3. **Strict held-out evaluation.** Stratified 80/20 split per seed (5 seeds), no leakage of held-out into hyperparameter selection. C-index reported as mean ± std over seeds.

A paired-bootstrap test (`tools/run_significance_tests.py`) is then applied to the difference in C-index between limpio and permisivo on the **same** held-out patients per seed, which is the right statistical structure (the classical DeLong test is defined only for AUC, not for right-censored C-index).

## What is *not* in this release

To keep the release focused, the parent project's TEXT-IN (Bio_ClinicalBERT), VISION-IN (nnU-Net), TurboLatent quantization, generative-VAE fusion, late-fusion diagnostic, Mahootiha-style late-fusion ablation, RSF/TabPFN external baselines, and explainer (GraphRAG) modules are all excluded. None of them affect any number in the paper.

## License

To be announced.
