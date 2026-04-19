"""
TABULAR-CONN: Config-Driven Experiment Runner
=============================================

Orchestrates all experimental phases. Driven entirely by experiment_config.yaml.

Three rules this runner enforces:

  1. DECLARATIVE CONFIGS: All paths, hyperparameters, variant choices, and
     phase enable/disable flags live in experiment_config.yaml. Zero hardcoding
     in this file.
  
  2. STRUCTURED OUTPUTS WITH PROVENANCE: Every run produces a unique directory
     {output.base_dir}/{timestamp}_{config_hash}/ containing:
       - experiment_config.yaml         (exact copy of config used)
       - feature_config.yaml            (exact copy of feature schema)
       - run_metadata.json              (timestamp, hash, environment)
       - phase{N}_{name}.csv            (canonical column names)
       - summary.json                   (high-level results)
  
  3. SWAPS VIA CONFIG ONLY: Adding a new variant/imputation requires
     (a) registering it in registry.py and (b) listing it in the config.
     Zero changes to this runner.

Usage:
    python experiment_runner.py experiment_config.yaml
    
Or from a notebook:
    from experiment_runner import run_experiment
    run_experiment("experiment_config.yaml")
"""

import json
import sys
import time
import hashlib
import shutil
import platform
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional, Union
import os

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import StratifiedKFold
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

warnings.filterwarnings('ignore')

# Add parent directory to path to allow absolute imports from 'code'
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from components.adapters.ingestion.tabular.utils.extractor import TCGAExtractor
from components.adapters.ingestion.tabular.utils.imputation_benchmark import TabularPreprocessor
from core.model_utils import (
    verify_ingestion_contract,
    train_variant_c,
    benchmark_efficiency,
    cox_partial_likelihood_loss,
)
from components.adapters.ingestion.tabular.models.linear_compact import VariantC_LinearEncoder
from core.registry import get_imputation, get_variant, list_components
from core.main import MultimodalPipeline, discover_modality_files




# ============================================================
# PROVENANCE & RUN DIRECTORY MANAGEMENT
# ============================================================

def compute_config_hash(config_dict: dict) -> str:
    """Deterministic hash of canonicalized config. First 8 hex chars."""
    canonical = yaml.dump(config_dict, sort_keys=True, default_flow_style=False)
    return hashlib.sha256(canonical.encode()).hexdigest()[:8]


def create_run_directory(config: dict, config_path: Optional[Union[str, Path]] = None) -> Path:
    """Create timestamped + hashed run directory and save config into it."""
    base_dir = Path(config['output']['base_dir'])
    base_dir.mkdir(parents=True, exist_ok=True)
    
    config_hash = compute_config_hash(config)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_id = f"{timestamp}_{config_hash}"
    run_dir = base_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    
    # Save experiment config
    if config_path and Path(config_path).exists():
        shutil.copy2(config_path, run_dir / "experiment_config.yaml")
    else:
        with open(run_dir / "experiment_config.yaml", 'w') as f:
            yaml.dump(config, f, sort_keys=False)
    
    # Copy feature config verbatim
    feature_config_path = Path(config['data']['feature_config'])
    if feature_config_path.exists():
        shutil.copy2(feature_config_path, run_dir / "feature_config.yaml")
    
    # Write run metadata
    metadata = {
        'run_id': run_id,
        'timestamp': timestamp,
        'config_hash': config_hash,
        'experiment_name': config['experiment']['name'],
        'protocol_version': config['experiment'].get('protocol_version', 'unknown'),
        'environment': {
            'python': platform.python_version(),
            'platform': platform.platform(),
            'torch': torch.__version__,
            'numpy': np.__version__,
            'pandas': pd.__version__,
        },
        'registered_components': list_components(),
    }
    with open(run_dir / "run_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return run_dir


def log(msg: str, verbosity: str = "normal", level: str = "info"):
    """Verbosity-aware logging."""
    if verbosity == "silent":
        return
    if level == "debug" and verbosity != "verbose":
        return
    print(msg)


# ============================================================
# CALIBRATION METRICS
# ============================================================

def expected_calibration_error(predicted: np.ndarray, observed: np.ndarray, n_bins: int = 10) -> float:
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(predicted)
    for i in range(n_bins):
        mask = (predicted >= bin_edges[i]) & (predicted < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / n) * abs(observed[mask].mean() - predicted[mask].mean())
    return ece


def brier_score(predicted: np.ndarray, observed: np.ndarray) -> float:
    return float(np.mean((predicted - observed) ** 2))


# ============================================================
# PHASE 1 — IMPUTATION BENCHMARK
# ============================================================

def phase_1_imputation(
    df_features: pd.DataFrame,
    df_targets: pd.DataFrame,
    config: dict,
    run_dir: Path,
) -> Tuple[pd.DataFrame, str]:
    phase_cfg = config['phase_1_imputation']
    if not phase_cfg.get('enabled', False):
        log("[PHASE 1] DISABLED — skipping imputation benchmark")
        fallback = config.get('phase_2_variants', {}).get('imputation_for_variants', 'knn_5')
        return None, 'knn_5' if fallback == 'auto' else fallback
    
    log("\n[PHASE 1] Imputation benchmark")
    
    valid = df_targets['survival_days'].notna() & (df_targets['survival_days'] > 0)
    X = df_features.loc[valid].copy()
    y = df_targets.loc[valid].copy()
    log(f"  Cases with valid survival: {len(X)} (events: {int(y['event'].sum())})")
    
    seeds = config['random']['seeds']
    n_folds = config['random']['n_folds']
    cox_penalizer = phase_cfg.get('cox_penalizer', 0.1)
    strategies = phase_cfg['strategies']
    
    rows = []
    for strategy_name in strategies:
        log(f"  → {strategy_name}")
        seed_ci, seed_ks = [], []
        
        for seed in seeds:
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
            fold_ci, fold_ks = [], []
            
            for tr_idx, va_idx in skf.split(X, y['event']):
                X_tr, X_va = X.iloc[tr_idx].copy(), X.iloc[va_idx].copy()
                y_tr, y_va = y.iloc[tr_idx].copy(), y.iloc[va_idx].copy()
                
                # Stash original distributions for K-S test
                original_dists = {
                    c: X_tr[c].dropna().values
                    for c in X_tr.select_dtypes(include=[np.number]).columns
                    if X_tr[c].dropna().shape[0] > 10
                }
                
                strategy = get_imputation(strategy_name)
                prep = TabularPreprocessor()
                X_tr_p, _, _ = prep.fit_transform(X_tr, strategy)
                X_va_p, _, _ = prep.transform(X_va)
                
                # K-S fidelity
                ks_scores = []
                from scipy import stats
                for c, orig in original_dists.items():
                    if c in X_tr_p.columns:
                        s, _ = stats.ks_2samp(orig, X_tr_p[c].values)
                        ks_scores.append(s)
                fold_ks.append(np.mean(ks_scores) if ks_scores else np.nan)
                
                # C-index downstream with FIXED Cox predictor
                try:
                    cox_df = X_tr_p.copy()
                    cox_df['T'] = y_tr['survival_days'].values
                    cox_df['E'] = y_tr['event'].values
                    valid_cols = [
                        c for c in cox_df.columns
                        if c not in ['T', 'E'] and cox_df[c].std() > 1e-8
                    ]
                    cox_df = cox_df[valid_cols + ['T', 'E']].replace(
                        [np.inf, -np.inf], np.nan
                    ).dropna()
                    
                    cph = CoxPHFitter(penalizer=cox_penalizer)
                    cph.fit(cox_df, duration_col='T', event_col='E')
                    
                    X_va_cox = X_va_p[valid_cols].replace(
                        [np.inf, -np.inf], np.nan
                    ).fillna(0)
                    risk = cph.predict_partial_hazard(X_va_cox).values.ravel()
                    ci = concordance_index(
                        y_va['survival_days'].values, -risk, y_va['event'].values
                    )
                    fold_ci.append(ci)
                except Exception as e:
                    log(f"    Cox failed: {e}", level="debug")
                    fold_ci.append(np.nan)
            
            seed_ci.append(np.nanmean(fold_ci))
            seed_ks.append(np.nanmean(fold_ks))
        
        rows.append({
            'strategy': strategy_name,
            'cindex_mean': float(np.nanmean(seed_ci)),
            'cindex_std': float(np.nanstd(seed_ci)),
            'ks_mean': float(np.nanmean(seed_ks)),
            'ks_std': float(np.nanstd(seed_ks)),
            'n_seeds': len(seeds),
            'n_folds': n_folds,
        })
        log(f"     C-index {rows[-1]['cindex_mean']:.4f} ± {rows[-1]['cindex_std']:.4f} | "
            f"K-S {rows[-1]['ks_mean']:.4f}")
    
    df_results = pd.DataFrame(rows)
    df_results.to_csv(run_dir / "phase1_imputation.csv", index=False)
    
    best = df_results.loc[df_results['cindex_mean'].idxmax(), 'strategy']
    log(f"  WINNER: {best}")
    return df_results, best


# ============================================================
# PHASE 2 — VARIANT COMPARISON
# ============================================================

def _build_mask_aligned(mask_df: pd.DataFrame, feature_df: pd.DataFrame) -> torch.Tensor:
    """Align missingness mask columns to feature column order."""
    aligned = torch.ones(len(feature_df), len(feature_df.columns), dtype=torch.float32)
    for i, col in enumerate(feature_df.columns):
        mask_col = f"mask__{col}"
        if mask_col in mask_df.columns:
            aligned[:, i] = torch.tensor(mask_df[mask_col].values, dtype=torch.float32)
    return aligned


def phase_2_variants(
    df_features: pd.DataFrame,
    df_targets: pd.DataFrame,
    config: dict,
    run_dir: Path,
    best_imputation: str,
) -> Optional[pd.DataFrame]:
    phase_cfg = config['phase_2_variants']
    if not phase_cfg['enabled']:
        log("[PHASE 2] DISABLED")
        return None
    
    log("\n[PHASE 2] Variant comparison")
    
    valid = df_targets['survival_days'].notna() & (df_targets['survival_days'] > 0)
    X = df_features.loc[valid].copy()
    y = df_targets.loc[valid].copy()
    
    median_surv = y['survival_days'].median()
    y['risk_group'] = (y['survival_days'] < median_surv).astype(int)
    
    seeds = config['random']['seeds']
    n_folds = config['random']['n_folds']
    output_dim = phase_cfg['output_dim']
    variants = phase_cfg['variants']
    variant_params = phase_cfg.get('variant_params', {})
    
    imp_for_variants = phase_cfg['imputation_for_variants']
    if imp_for_variants == "auto":
        imp_for_variants = best_imputation
    imp_for_baseline = phase_cfg.get('imputation_for_baseline', 'mean_median')
    
    log(f"  Cases: {len(X)}, output_dim: {output_dim}")
    log(f"  Variants: {variants}")
    log(f"  Imputation for baseline: {imp_for_baseline}")
    log(f"  Imputation for advanced variants: {imp_for_variants}")
    
    rows = []
    
    for seed in seeds:
        log(f"  Seed {seed}")
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        
        for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X, y['event'])):
            X_tr_raw, X_va_raw = X.iloc[tr_idx].copy(), X.iloc[va_idx].copy()
            y_tr, y_va = y.iloc[tr_idx].copy(), y.iloc[va_idx].copy()
            
            # Two preprocessing passes: one for baseline, one for advanced variants
            prep_base = TabularPreprocessor()
            X_tr_b, mask_tr_b, conf_tr_b = prep_base.fit_transform(
                X_tr_raw, get_imputation(imp_for_baseline)
            )
            X_va_b, mask_va_b, conf_va_b = prep_base.transform(X_va_raw)
            
            prep_adv = TabularPreprocessor()
            X_tr_a, mask_tr_a, conf_tr_a = prep_adv.fit_transform(
                X_tr_raw, get_imputation(imp_for_variants)
            )
            X_va_a, mask_va_a, conf_va_a = prep_adv.transform(X_va_raw)
            
            input_dim = X_tr_a.shape[1]
            
            for variant_name in variants:
                # Baseline uses its own preprocessing
                if variant_name == 'cox_baseline':
                    X_tr_use, X_va_use = X_tr_b, X_va_b
                    conf_va_use = conf_va_b
                else:
                    X_tr_use, X_va_use = X_tr_a, X_va_a
                    conf_va_use = conf_va_a
                
                ci, ece, bs, contract_ok = _evaluate_variant(
                    variant_name=variant_name,
                    input_dim=input_dim,
                    output_dim=output_dim,
                    variant_params=variant_params.get(variant_name, {}),
                    X_tr=X_tr_use, X_va=X_va_use,
                    y_tr=y_tr, y_va=y_va,
                    mask_tr=mask_tr_a, mask_va=mask_va_a,
                    conf_tr=conf_tr_a, conf_va=conf_va_use,
                    median_surv=median_surv,
                )
                
                rows.append({
                    'seed': seed,
                    'fold': fold_idx,
                    'variant': variant_name,
                    'cindex': ci,
                    'ece': ece,
                    'brier_score': bs,
                    'contract_satisfied': contract_ok,
                })
    
    df_results = pd.DataFrame(rows)
    df_results.to_csv(run_dir / "phase2_variants.csv", index=False)
    
    summary = df_results.groupby('variant').agg(
        cindex_mean=('cindex', 'mean'),
        cindex_std=('cindex', 'std'),
        ece_mean=('ece', 'mean'),
        brier_mean=('brier_score', 'mean'),
        contract_satisfied=('contract_satisfied', 'all'),
    ).round(4)
    summary.to_csv(run_dir / "phase2_variants_summary.csv")
    log("\n  SUMMARY:")
    log(summary.to_string())
    
    return df_results


def _evaluate_variant(
    variant_name, input_dim, output_dim, variant_params,
    X_tr, X_va, y_tr, y_va,
    mask_tr, mask_va, conf_tr, conf_va,
    median_surv,
) -> Tuple[float, float, float, bool]:
    """Evaluate a single variant on a single fold. Returns (cindex, ece, brier, contract_ok)."""
    try:
        if variant_name == 'cox_baseline':
            return _eval_cox_baseline(X_tr, X_va, y_tr, y_va, conf_va, output_dim, median_surv)
        elif variant_name == 'linear_compact':
            return _eval_linear_compact(X_tr, X_va, y_tr, y_va, mask_tr, mask_va, conf_va, output_dim, variant_params)
        elif variant_name == 'ft_transformer':
            return _eval_ft_transformer(
                X_tr, X_va, y_tr, y_va,
                mask_tr, mask_va, conf_va, output_dim,
                variant_params.get(variant_name, {}),
            )
        else:
            # Generic fallback for future-registered variants
            log(f"    {variant_name}: no evaluator registered, skipping", level="debug")
            return np.nan, np.nan, np.nan, False
    except Exception as e:
        log(f"    {variant_name} failed: {e}", level="debug")
        return np.nan, np.nan, np.nan, False


def _eval_cox_baseline(X_tr, X_va, y_tr, y_va, conf_va, output_dim, median_surv):
    cox_df = X_tr.copy()
    cox_df['T'] = y_tr['survival_days'].values
    cox_df['E'] = y_tr['event'].values
    valid_cols = [c for c in cox_df.columns
                  if c not in ['T', 'E'] and cox_df[c].std() > 1e-8]
    cox_df = cox_df[valid_cols + ['T', 'E']].replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(cox_df, duration_col='T', event_col='E')
    
    X_va_cox = X_va[valid_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    risk = cph.predict_partial_hazard(X_va_cox).values.ravel()
    ci = concordance_index(y_va['survival_days'], -risk, y_va['event'])
    
    surv_func = cph.predict_survival_function(X_va_cox)
    pred_probs = (1 - surv_func.loc[median_surv].values
                  if median_surv in surv_func.index
                  else np.full(len(X_va_cox), 0.5))
    pred_probs = np.clip(pred_probs, 0.01, 0.99)
    
    ece = expected_calibration_error(pred_probs, y_va['event'].values)
    bs = brier_score(pred_probs, y_va['event'].values)
    
    # Contract verification on a sample
    variant = get_variant('cox_baseline', X_va.shape[1], output_dim)
    emb, conf_t = variant.encode(
        X_va.values.astype(np.float32),
        conf_va.values.astype(np.float32)
    )
    contract = verify_ingestion_contract(emb, conf_t, output_dim, verbose=False)
    return ci, ece, bs, contract.get('contract_satisfied', False)


def _eval_linear_compact(X_tr, X_va, y_tr, y_va, mask_tr, mask_va, conf_va, output_dim, params):
    encoder = get_variant(
        'linear_compact',
        input_dim=X_tr.shape[1],
        output_dim=output_dim,
        hidden_dim=params.get('hidden_dim', 128),
    )
    
    X_tr_t = torch.tensor(X_tr.values, dtype=torch.float32)
    X_va_t = torch.tensor(X_va.values, dtype=torch.float32)
    M_tr = _build_mask_aligned(mask_tr, X_tr)
    M_va = _build_mask_aligned(mask_va, X_va)
    T_tr = torch.tensor(y_tr['survival_days'].values, dtype=torch.float32)
    E_tr = torch.tensor(y_tr['event'].values, dtype=torch.float32)
    T_va = torch.tensor(y_va['survival_days'].values, dtype=torch.float32)
    E_va = torch.tensor(y_va['event'].values, dtype=torch.float32)
    
    result = train_variant_c(
        encoder, X_tr_t, M_tr, T_tr, E_tr, X_va_t, M_va, T_va, E_va,
        epochs=params.get('epochs', 200),
        lr=params.get('lr', 0.001),
        patience=params.get('patience', 20),
        verbose=False,
    )
    ci = result['best_val_cindex']
    
    encoder.eval()
    with torch.no_grad():
        emb, conf_t = encoder(X_va_t, M_va)
        risk = result['risk_head'](emb).squeeze(-1).numpy()
    pred_probs = 1 / (1 + np.exp(-risk))
    pred_probs = np.clip(pred_probs, 0.01, 0.99)
    ece = expected_calibration_error(pred_probs, y_va['event'].values)
    bs = brier_score(pred_probs, y_va['event'].values)
    
    contract = verify_ingestion_contract(emb, conf_t, output_dim, verbose=False)
    return ci, ece, bs, contract.get('contract_satisfied', False)


def _eval_ft_transformer(
    X_tr, X_va, y_tr, y_va,
    mask_tr, mask_va, conf_va,
    output_dim, params,
):
    """
    Evaluate the FT-Transformer variant on one fold.

    Mirrors _eval_linear_compact in structure so the two encoders are directly
    comparable: identical training loop, identical loss, identical contract
    verification. The only thing that changes is the encoder architecture
    itself (registered under name 'ft_transformer' in the registry).

    Parameters
    ----------
    X_tr, X_va : pd.DataFrame
        Preprocessed training and validation features (advanced-variant
        imputation, typically KNN k=5).
    y_tr, y_va : pd.DataFrame
        Targets with columns ['survival_days', 'event', 'risk_group'].
    mask_tr, mask_va : pd.DataFrame
        Missingness masks with `mask__<feature>` columns.
    conf_va : pd.DataFrame
        Per-case confidence columns used for the ingestion contract check.
    output_dim : int
        Contract-fixed embedding dim (typically 768).
    params : dict
        Variant-specific hyperparameters from the YAML.

    Returns
    -------
    (cindex, ece, brier_score, contract_satisfied) : tuple
        Same signature as the other _eval_* helpers in the runner.
    """
    # Encoder construction via registry — same pattern as linear_compact
    encoder = get_variant(
        'ft_transformer',
        input_dim=X_tr.shape[1],
        output_dim=output_dim,
        d_token=params.get('d_token', 192),
        n_blocks=params.get('n_blocks', 3),
        n_heads=params.get('n_heads', 8),
        d_ff=params.get('d_ff', None),
        dropout=params.get('dropout', 0.1),
    )

    # Tensor conversion — identical to _eval_linear_compact
    X_tr_t = torch.tensor(X_tr.values, dtype=torch.float32)
    X_va_t = torch.tensor(X_va.values, dtype=torch.float32)
    M_tr = _build_mask_aligned(mask_tr, X_tr)
    M_va = _build_mask_aligned(mask_va, X_va)
    T_tr = torch.tensor(y_tr['survival_days'].values, dtype=torch.float32)
    E_tr = torch.tensor(y_tr['event'].values, dtype=torch.float32)
    T_va = torch.tensor(y_va['survival_days'].values, dtype=torch.float32)
    E_va = torch.tensor(y_va['event'].values, dtype=torch.float32)

    # Training — Cox partial-likelihood, same as linear_compact for fair
    # architectural comparison. FT-Transformer benefits from slightly gentler
    # learning rates than a linear MLP (Gorishniy 2021 reports lr≈3e-4 as
    # a strong default for transformers of this size on small tabular data).
    result = train_variant_c(
        encoder, X_tr_t, M_tr, T_tr, E_tr,
        X_va_t, M_va, T_va, E_va,
        epochs=params.get('epochs', 200),
        lr=params.get('lr', 3e-4),
        patience=params.get('patience', 20),
        weight_decay=params.get('weight_decay', 1e-4),
        verbose=False,
    )
    ci = result['best_val_cindex']

    # Calibration & contract verification on validation set — same pattern
    encoder.eval()
    with torch.no_grad():
        emb, conf_t = encoder(X_va_t, M_va)
        risk = result['risk_head'](emb).squeeze(-1).numpy()

    pred_probs = 1 / (1 + np.exp(-risk))
    pred_probs = np.clip(pred_probs, 0.01, 0.99)
    ece = expected_calibration_error(pred_probs, y_va['event'].values)
    bs = brier_score(pred_probs, y_va['event'].values)

    contract = verify_ingestion_contract(emb, conf_t, output_dim, verbose=False)
    return ci, ece, bs, contract.get('contract_satisfied', False)


# ============================================================
# PHASE 2 EXTERNAL — non-compliant SOTA baselines
# ============================================================

def phase_2_external_baselines(
    df_features: pd.DataFrame,
    df_targets: pd.DataFrame,
    config: dict,
    run_dir: Path,
    best_imputation: str,
) -> Optional[pd.DataFrame]:
    """
    Run external (non-contract-compliant) SOTA baselines for tabular survival.

    Uses the SAME preprocessing as phase_2_variants advanced variants
    (imputation_for_variants, typically KNN k=5) and the SAME StratifiedKFold
    partition with the SAME seeds, so that every external baseline row is
    evaluated on the same validation set as every compliant variant row.
    """
    phase_cfg = config.get('phase_2_external_baselines', {})
    if not phase_cfg.get('enabled', False):
        log("[PHASE 2-EXTERNAL] DISABLED")
        return None

    log("\n[PHASE 2-EXTERNAL] External non-compliant baselines")

    baselines_cfg = phase_cfg.get('baselines', {})
    active = [name for name, cfg in baselines_cfg.items() if cfg.get('enabled', True)]
    if not active:
        log("  No external baselines enabled. Skipping.")
        return None

    log(f"  Enabled baselines: {active}")

    # --------------------------------------------------------
    # Data setup — MUST mirror phase_2_variants exactly
    # --------------------------------------------------------
    valid = df_targets['survival_days'].notna() & (df_targets['survival_days'] > 0)
    X_all = df_features.loc[valid].copy()
    y_all = df_targets.loc[valid].copy()

    seeds = config['random']['seeds']
    n_folds = config['random']['n_folds']

    # Imputation strategy — same as the advanced variants of phase_2_variants
    imp_for_variants = config['phase_2_variants']['imputation_for_variants']
    if imp_for_variants == "auto":
        imp_for_variants = best_imputation

    log(f"  Cases: {len(X_all)}")
    log(f"  Imputation: {imp_for_variants}")
    log(f"  Seeds: {seeds}")
    log(f"  N folds: {n_folds}")

    rows = []
    for seed in seeds:
        log(f"  Seed {seed}")
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

        for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X_all, y_all['event'])):
            X_tr_raw, X_va_raw = X_all.iloc[tr_idx].copy(), X_all.iloc[va_idx].copy()
            y_tr, y_va = y_all.iloc[tr_idx].copy(), y_all.iloc[va_idx].copy()

            prep = TabularPreprocessor()
            X_tr, _, _ = prep.fit_transform(X_tr_raw, get_imputation(imp_for_variants))
            X_va, _, _ = prep.transform(X_va_raw)

            # Fill any remaining NaNs (some imputers can leave them at edges)
            X_tr = X_tr.replace([np.inf, -np.inf], np.nan).fillna(0)
            X_va = X_va.replace([np.inf, -np.inf], np.nan).fillna(0)

            # -- TabPFN --
            if 'tabpfn_external' in active:
                row = _eval_tabpfn_external(
                    X_tr, X_va, y_tr, y_va,
                    cfg=baselines_cfg.get('tabpfn_external', {}),
                    seed=seed, fold=fold_idx,
                )
                rows.append(row)

            # -- RSF --
            if 'rsf_external' in active:
                row = _eval_rsf_external(
                    X_tr, X_va, y_tr, y_va,
                    cfg=baselines_cfg.get('rsf_external', {}),
                    seed=seed, fold=fold_idx,
                )
                rows.append(row)

    if not rows:
        return None

    df_results = pd.DataFrame(rows)
    # Drop the nested model_summary from the CSV (it's kept only for the JSON summary).
    csv_view = df_results.drop(columns=['model_summary'], errors='ignore')
    csv_view.to_csv(run_dir / "phase2_external_baselines.csv", index=False)

    # Aggregated summary
    summary = (
        csv_view
        .groupby('baseline')
        .agg(
            cindex_mean=('cindex', 'mean'),
            cindex_std=('cindex', 'std'),
            n_folds=('fold', 'count'),
        )
        .round(4)
    )
    summary['contract_compliant'] = False
    summary.to_csv(run_dir / "phase2_external_baselines_summary.csv")

    log("\n  EXTERNAL BASELINES SUMMARY:")
    log(summary.to_string())

    return df_results


def _eval_tabpfn_external(X_tr, X_va, y_tr, y_va, cfg, seed, fold):
    """Fit TabPFN on binary-at-median survival task, compute C-index."""
    try:
        from components.external.tabpfn_external import TabPFNExternalBaseline
    except ImportError:
        # Fallback if structure varies slightly
        try:
            from tabpfn_external import TabPFNExternalBaseline
        except ImportError:
            log(f"    tabpfn_external fold={fold} seed={seed} FAILED: module not found")
            return {'baseline': 'tabpfn_external', 'seed': seed, 'fold': fold, 'cindex': np.nan, 'contract_compliant': False, 'model_summary': {'error': 'ImportError'}}

    try:
        model = TabPFNExternalBaseline(
            device=cfg.get('device', 'auto'),
            n_estimators=cfg.get('n_estimators', 4),
            random_state=seed,
        )
        model.fit(
            X_tr,
            survival_days=y_tr['survival_days'].values,
            event=y_tr['event'].values,
        )
        risk = model.predict_risk(X_va)
        ci = concordance_index(
            y_va['survival_days'].values, -risk, y_va['event'].values
        )
        model_sum = model.summary()
        return {
            'baseline': 'tabpfn_external',
            'seed': seed,
            'fold': fold,
            'cindex': float(ci),
            'contract_compliant': False,
            'model_summary': model_sum,
        }
    except Exception as e:
        log(f"    tabpfn_external fold={fold} seed={seed} FAILED: {e}", level="debug")
        return {
            'baseline': 'tabpfn_external',
            'seed': seed, 'fold': fold,
            'cindex': float('nan'),
            'contract_compliant': False,
            'model_summary': {'error': str(e)},
        }


def _eval_rsf_external(X_tr, X_va, y_tr, y_va, cfg, seed, fold):
    """Fit Random Survival Forest, compute C-index."""
    try:
        from components.external.rsf_external import RSFExternalBaseline
    except ImportError:
        try:
            from rsf_external import RSFExternalBaseline
        except ImportError:
            log(f"    rsf_external fold={fold} seed={seed} FAILED: module not found")
            return {'baseline': 'rsf_external', 'seed': seed, 'fold': fold, 'cindex': np.nan, 'contract_compliant': False, 'model_summary': {'error': 'ImportError'}}

    try:
        model = RSFExternalBaseline(
            n_estimators=cfg.get('n_estimators', 100),
            min_samples_split=cfg.get('min_samples_split', 10),
            min_samples_leaf=cfg.get('min_samples_leaf', 15),
            max_features=cfg.get('max_features', 'sqrt'),
            n_jobs=cfg.get('n_jobs', -1),
            random_state=seed,
        )
        model.fit(
            X_tr,
            survival_days=y_tr['survival_days'].values,
            event=y_tr['event'].values,
        )
        risk = model.predict_risk(X_va)
        ci = concordance_index(
            y_va['survival_days'].values, -risk, y_va['event'].values
        )
        model_sum = model.summary()
        return {
            'baseline': 'rsf_external',
            'seed': seed,
            'fold': fold,
            'cindex': float(ci),
            'contract_compliant': False,
            'model_summary': model_sum,
        }
    except Exception as e:
        log(f"    rsf_external fold={fold} seed={seed} FAILED: {e}", level="debug")
        return {
            'baseline': 'rsf_external',
            'seed': seed, 'fold': fold,
            'cindex': float('nan'),
            'contract_compliant': False,
            'model_summary': {'error': str(e)},
        }


# ============================================================
# PHASE 3 — EFFICIENCY BENCHMARK
# ============================================================

def phase_3_efficiency(
    input_dim: int,
    config: dict,
    run_dir: Path,
) -> Optional[pd.DataFrame]:
    phase_cfg = config['phase_3_efficiency']
    if not phase_cfg['enabled']:
        log("[PHASE 3] DISABLED")
        return None
    
    log("\n[PHASE 3] Efficiency benchmark")
    
    output_dim = config['phase_2_variants']['output_dim']
    variant_params = config['phase_2_variants'].get('variant_params', {})
    
    rows = []
    sample_x = torch.randn(1, input_dim)
    sample_m = torch.ones(1, input_dim)
    
    for variant_name in phase_cfg['variants']:
        try:
            params = variant_params.get(variant_name, {})
            encoder = get_variant(
                variant_name, input_dim=input_dim, output_dim=output_dim,
                hidden_dim=params.get('hidden_dim', 128),
            )
            metrics = benchmark_efficiency(
                encoder, sample_x, sample_m,
                n_warmup=phase_cfg.get('n_warmup', 10),
                n_runs=phase_cfg.get('n_runs', 100),
            )
            metrics['variant'] = variant_name
            rows.append(metrics)
            log(f"  {variant_name}: {metrics['latency_ms']:.3f}ms, "
                f"{metrics['memory_mb']:.2f}MB, params={metrics['n_parameters']}")
        except Exception as e:
            log(f"  {variant_name} failed: {e}", level="debug")
    
    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "phase3_efficiency.csv", index=False)
    return df


# ============================================================
# PHASE 4 — STRESS TEST
# ============================================================

def phase_4_stress(
    df_features: pd.DataFrame,
    df_targets: pd.DataFrame,
    config: dict,
    run_dir: Path,
    best_imputation: str,
) -> Optional[pd.DataFrame]:
    phase_cfg = config['phase_4_stress']
    if not phase_cfg['enabled']:
        log("[PHASE 4] DISABLED")
        return None
    
    log("\n[PHASE 4] Stress test")
    
    df_noisy = df_features.copy()
    rng = np.random.default_rng(seed=42)
    
    # MCAR injection
    for col in df_noisy.columns:
        mask = rng.random(len(df_noisy)) < phase_cfg['noise_fraction']
        df_noisy.loc[mask, col] = np.nan
    
    # Outlier injection
    for col in df_noisy.select_dtypes(include=[np.number]).columns:
        valid = df_noisy[col].dropna()
        if len(valid) > 0:
            outlier_mask = rng.random(len(df_noisy)) < phase_cfg['outlier_fraction']
            extreme = valid.mean() + 5 * valid.std()
            non_null = df_noisy[col].notna()
            df_noisy.loc[outlier_mask & non_null, col] = extreme
    
    # Run variant comparison on noisy data with reduced seed list
    stress_config = {**config}
    stress_config['random'] = {
        'seeds': phase_cfg.get('seeds', config['random']['seeds'][:3]),
        'n_folds': config['random']['n_folds'],
    }
    
    (run_dir / "_stress_clean").mkdir(parents=True, exist_ok=True)
    (run_dir / "_stress_noisy").mkdir(parents=True, exist_ok=True)
    
    log("  Running clean evaluation...")
    clean_results = phase_2_variants(df_features, df_targets, stress_config, run_dir / "_stress_clean", best_imputation)
    log("  Running noisy evaluation...")
    noisy_results = phase_2_variants(df_noisy, df_targets, stress_config, run_dir / "_stress_noisy", best_imputation)
    
    if clean_results is None or noisy_results is None:
        return None
    
    clean_summary = clean_results.groupby('variant')['cindex'].mean()
    noisy_summary = noisy_results.groupby('variant')['cindex'].mean()
    
    degradation = pd.DataFrame({
        'variant': clean_summary.index,
        'clean_cindex': clean_summary.values,
        'noisy_cindex': noisy_summary.values,
        'cindex_drop': clean_summary.values - noisy_summary.values,
        'pct_drop': ((clean_summary.values - noisy_summary.values) / clean_summary.values * 100),
    })
    degradation.to_csv(run_dir / "phase4_stress.csv", index=False)
    log("\n  DEGRADATION:")
    log(degradation.round(4).to_string(index=False))
    
    return degradation


# ============================================================
# PHASE 5 — MULTIMODAL BASELINE (END-TO-END)
# ============================================================

def phase_5_multimodal(
    df_features: pd.DataFrame,
    df_targets: pd.DataFrame,
    config: dict,
    run_dir: Path,
) -> Optional[pd.DataFrame]:
    """
    End-to-end CLINICAL-CORE / RENAL-CORE baseline:
      TABULAR-CONN + TEXT-CONN + VISION-CONN → FUSION-PROC → PROGNOSIS-PROC
    
    Reports C-index per modality combination (ablation table).
    """
    phase_cfg = config.get('phase_5_multimodal', {})
    if not phase_cfg.get('enabled', False):
        log("[PHASE 5] DISABLED")
        return None
    
    log("\n[PHASE 5] Multimodal end-to-end baseline")
    
    # Discover modality files for each case
    case_ids = list(df_features.index)
    data_dirs = {
        'text_dir': phase_cfg.get('text_dir'),
        'vision_dir': phase_cfg.get('vision_dir'),
    }
    
    log(f"  Modalities enabled: {phase_cfg['modalities']}")
    log(f"  Text data dir: {data_dirs['text_dir'] or '(disabled — using mock)'}")
    log(f"  Vision data dir: {data_dirs['vision_dir'] or '(disabled — using mock)'}")
    
    modality_files = discover_modality_files(data_dirs, case_ids)
    
    n_text = modality_files['text_path'].notna().sum()
    n_vision = modality_files['vision_path'].notna().sum()
    log(f"  Cases with text data:   {n_text}/{len(modality_files)}")
    log(f"  Cases with vision data: {n_vision}/{len(modality_files)}")
    
    # Save modality file manifest for traceability
    modality_files.to_csv(run_dir / "phase5_modality_manifest.csv")
    
    # Run the pipeline
    pipeline = MultimodalPipeline(config)
    
    seeds = config['random']['seeds']
    n_folds = config['random']['n_folds']
    
    results = pipeline.run_ablation(
        df_features=df_features,
        df_targets=df_targets,
        modality_files=modality_files,
        seeds=seeds,
        n_folds=n_folds,
        ablations=phase_cfg.get('ablations'),
    )
    
    results.to_csv(run_dir / "phase5_multimodal_ablation.csv", index=False)
    
    log("\n  ABLATION RESULTS:")
    log(results[['subset_label', 'n_cases', 'cindex_mean', 'cindex_std']].to_string(index=False))
    
    return results


# ============================================================
# ARTIFACT MANAGEMENT (shared by phases 6/7/8)
# ============================================================
 
def get_artifacts_dir(run_dir: Path) -> Path:
    """Canonical artifacts directory inside a run directory."""
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    return artifacts_dir
 
 
def find_most_recent_artifact(
    base_dir: Path,
    artifact_name: str,
    within_n_runs: int = 10,
) -> Optional[Path]:
    """
    Search {base_dir}/*/artifacts/{artifact_name} across recent runs.
 
    Returns the most recent matching path, or None if none exists.
    Only looks at the last `within_n_runs` runs (sorted by directory name,
    which embeds the timestamp — this is why the run_dir format matters).
    """
    if not base_dir.exists():
        return None
    run_dirs = sorted(
        [d for d in base_dir.iterdir() if d.is_dir()],
        reverse=True,
    )[:within_n_runs]
    for rd in run_dirs:
        candidate = rd / "artifacts" / artifact_name
        if candidate.exists():
            return candidate
    return None
 
 
def resolve_artifact_path(
    artifact_name: str,
    current_run_dir: Path,
    phase_cfg: dict,
    output_base_dir: Path,
    explicit_key: str = "source_artifact_path",
) -> Optional[Path]:
    """
    Precedence-ordered artifact discovery.
 
    Args:
        artifact_name: canonical filename (e.g. 'phase_6_latent_z.npz').
        current_run_dir: {output.base_dir}/{timestamp_hash}/
        phase_cfg: the config block of the phase requesting the artifact.
        output_base_dir: {output.base_dir}/
        explicit_key: config key under which an explicit path may be set.
    """
    # 1. Current run (same-session)
    current = current_run_dir / "artifacts" / artifact_name
    if current.exists():
        return current
 
    # 2. Explicit path from YAML
    explicit = phase_cfg.get(explicit_key)
    if explicit:
        explicit_path = Path(explicit)
        if explicit_path.exists():
            return explicit_path
 
    # 3. Most recent across prior runs
    recent = find_most_recent_artifact(output_base_dir, artifact_name)
    if recent is not None:
        return recent
 
    return None
 
 
# ============================================================
# PHASE 6 — FUSION-PROC (VAE generative, 2-stage training)
# ============================================================
 
def phase_6_fusion_proc(
    df_features: "pd.DataFrame",
    df_targets: "pd.DataFrame",
    config: dict,
    run_dir: "Path",
    best_imputation: str,
) -> "Optional[pd.DataFrame]":
    """
    FUSION-PROC: train the generative VAE in two stages on a trained-tabular +
    mock-modality input and emit a frozen latent Z as artifact for downstream
    phases (TurboLatent, Prognosis benchmark).
 
    Pipeline:
      1. TabularPreprocessor produces X_tab [N, n_features] + mask.
      2. Train a linear_compact encoder (single split, all N cases) to map
         [n_features] → [modality_dim]. This is the "production embedder".
      3. Apply the trained encoder to the full cohort → tab_emb [N, modality_dim].
      4. Assemble trimodal input (tabular real + mock text/vision zeros).
      5. Train fusion_vae_generative (Stage A + Stage B) on the assembled tensor.
      6. Extract frozen Z from the VAE encoder and persist as artifact.
 
    Rationale for Option A (trained embedder vs random projection):
      The random projection used in the initial Phase 6 implementation
      caused a ~0.08 C-index drop downstream because it destroyed the
      discriminative structure of the 19 clinical features. Training a
      linear_compact encoder end-to-end with Cox loss restores that
      structure before the VAE sees the data. This matches the flow used
      in the Colab experiments where tabular_emb.npz was produced by a
      trained encoder.
    """
    phase_cfg = config.get('phase_6_fusion_proc', {})
    if not phase_cfg.get('enabled', False):
        log("[PHASE 6] DISABLED")
        return None
 
    log("\n[PHASE 6] FUSION-PROC (VAE generative, 2-stage)")
 
    from components.processors.fusion.models.vae_generative import (
        VAEGenTrainConfig,
    )
    from core.registry import get_fusion_proc, get_imputation
    from components.adapters.ingestion.tabular.utils.imputation_benchmark import (
        TabularPreprocessor,
    )
    from sklearn.model_selection import train_test_split
 
    seed = config['random'].get('seed', 42)
    np.random.seed(seed)
    torch.manual_seed(seed)
 
    # --- Filter valid cases (with survival data) ---
    valid = df_targets['survival_days'].notna() & (df_targets['survival_days'] > 0)
    X_raw = df_features.loc[valid].copy()
    y = df_targets.loc[valid].copy()
    case_ids = np.array(X_raw.index.tolist())
 
    # --- Preprocess tabular with best imputation ---
    imp_name = phase_cfg.get('tabular_imputation', 'auto')
    if imp_name == 'auto':
        imp_name = best_imputation
    prep = TabularPreprocessor()
    X_tab, mask, conf_tab = prep.fit_transform(X_raw, get_imputation(imp_name))
    N = X_tab.shape[0]
 
    log(f"  Cases: {N}  (events: {int(y['event'].sum())})")
    log(f"  Imputation: {imp_name}")
 
    # --- Build trimodal input (tabular real + mock text/vision) ---
    modality_dim = phase_cfg.get('modality_dim', 768)
    modalities = phase_cfg.get('modalities', ['tabular', 'text', 'vision'])
    n_mod = len(modalities)
 
    # Train a linear_compact encoder on the full cohort. Single stratified
    # split, not CV — this is an embedder, not a predictor benchmark.
    if X_tab.shape[1] == modality_dim:
        tab_emb = X_tab.values.astype(np.float32)
        log(f"  Tabular: already at modality_dim={modality_dim}, no encoder needed")
    else:
        log(f"  Training linear_compact embedder ({X_tab.shape[1]} → "
            f"{modality_dim}) on {N} cases...")
 
        enc_train_idx, enc_val_idx = train_test_split(
            np.arange(N),
            test_size=phase_cfg.get('encoder_val_fraction', 0.15),
            random_state=seed,
            stratify=y['event'].values,
        )
 
        X_tab_t = torch.tensor(X_tab.values, dtype=torch.float32)
        M_tab_t = _build_mask_aligned(mask, X_tab)
        T_all = torch.tensor(y['survival_days'].values, dtype=torch.float32)
        E_all = torch.tensor(y['event'].values, dtype=torch.float32)
 
        encoder_params = phase_cfg.get('encoder_params', {})
        encoder = VariantC_LinearEncoder(
            input_dim=X_tab.shape[1],
            hidden_dim=encoder_params.get('hidden_dim', 128),
            output_dim=modality_dim,
        )
 
        t0_enc = time.time()
        train_variant_c(
            encoder,
            X_tab_t[enc_train_idx], M_tab_t[enc_train_idx],
            T_all[enc_train_idx],    E_all[enc_train_idx],
            X_tab_t[enc_val_idx],    M_tab_t[enc_val_idx],
            T_all[enc_val_idx],      E_all[enc_val_idx],
            epochs=encoder_params.get('epochs', 200),
            lr=encoder_params.get('lr', 1e-3),
            patience=encoder_params.get('patience', 20),
            verbose=False,
        )
        elapsed_enc = time.time() - t0_enc
        log(f"  Encoder trained in {elapsed_enc:.1f}s")
 
        # Extract embeddings for the FULL cohort using the trained encoder
        encoder.eval()
        with torch.no_grad():
            tab_emb_t, _ = encoder(X_tab_t, M_tab_t)
        tab_emb = tab_emb_t.cpu().numpy().astype(np.float32)
        log(f"  Tabular embedding produced: shape={tab_emb.shape}  "
            f"range=[{tab_emb.min():.3f}, {tab_emb.max():.3f}]")
 
    # --- Assemble flat input tensor for the VAE (trimodal with mocks) ---
    X_flat = torch.zeros(N, modality_dim * n_mod, dtype=torch.float32)
    confs_matrix = np.zeros((N, n_mod), dtype=np.float32)
    for i, mod_name in enumerate(modalities):
        if mod_name == 'tabular':
            X_flat[:, i * modality_dim:(i + 1) * modality_dim] = torch.tensor(tab_emb)
            confs_matrix[:, i] = 1.0
        # text/vision remain zeros with confidence 0
 
    conf = torch.tensor(confs_matrix, dtype=torch.float32)
    T = torch.tensor(y['survival_days'].values, dtype=torch.float32)
    E = torch.tensor(y['event'].values, dtype=torch.float32)
 
    # --- Train/val split for the VAE (separate from the encoder split) ---
    val_frac = phase_cfg.get('val_fraction', 0.15)
    idx_tr, idx_va = train_test_split(
        np.arange(N), test_size=val_frac, random_state=seed,
        stratify=E.numpy(),
    )
    log(f"  VAE train/val: {len(idx_tr)}/{len(idx_va)}")
 
    # --- Instantiate VAE via registry ---
    model_params = phase_cfg.get('model_params', {})
    vae = get_fusion_proc(
        phase_cfg.get('fusion_proc', 'fusion_vae_generative'),
        modalities=modalities,
        modality_dims={m: modality_dim for m in modalities},
        d_latent=model_params.get('d_latent', 128),
        hidden_dims=tuple(model_params.get('hidden_dims', [512, 256])),
        dropout=model_params.get('dropout', 0.1),
    )
 
    # --- Build training config from YAML ---
    train_params = phase_cfg.get('training', {})
    stage_a = train_params.get('stage_a', {})
    stage_b = train_params.get('stage_b', {})
    loss_w = train_params.get('loss_weights', {})
    train_cfg = VAEGenTrainConfig(
        epochs_stage_a      = stage_a.get('epochs', 100),
        lr_stage_a          = stage_a.get('lr', 1e-3),
        patience_stage_a    = stage_a.get('patience', 20),
        kl_anneal_epochs    = stage_a.get('kl_anneal_epochs', 30),
        epochs_stage_b      = stage_b.get('epochs', 60),
        lr_stage_b          = stage_b.get('lr', 3e-4),
        patience_stage_b    = stage_b.get('patience', 15),
        triplet_margin      = stage_b.get('triplet_margin', 1.0),
        time_similar_window = stage_b.get('time_similar_window', 180.0),
        alpha_recon         = loss_w.get('alpha_recon', 1.0),
        beta_kl             = loss_w.get('beta_kl', 0.01),
        delta_contra        = loss_w.get('delta_contra', 0.5),
        train_with_masking  = train_params.get('train_with_masking', False),
        modality_dropout_prob = train_params.get('modality_dropout_prob', 0.3),
        weight_decay        = train_params.get('weight_decay', 1e-4),
        batch_size          = train_params.get('batch_size', 64),
        seed                = seed,
        verbose             = False,
    )
 
    # --- Train VAE ---
    t0 = time.time()
    result = vae.fit(
        X_train=X_flat[idx_tr],  conf_train=conf[idx_tr],
        T_train=T[idx_tr],        E_train=E[idx_tr],
        X_val=X_flat[idx_va],     conf_val=conf[idx_va],
        T_val=T[idx_va],          E_val=E[idx_va],
        cfg=train_cfg,
    )
    elapsed = time.time() - t0
    log(f"  VAE training elapsed: {elapsed:.1f}s  "
        f"(Stage A: {len(result['stage_A_history'])}ep, "
        f"Stage B: {len(result['stage_B_history'])}ep)")
 
    # --- Extract frozen Z for full cohort ---
    vae.eval()
    with torch.no_grad():
        Z, conf_full = vae.extract_latent_space(X_flat, conf)
    Z = Z.cpu().numpy()
    conf_full = conf_full.cpu().numpy()
 
    # --- Persist artifact ---
    artifacts_dir = get_artifacts_dir(run_dir)
    latent_path = artifacts_dir / "phase_6_latent_z.npz"
    np.savez(
        latent_path,
        Z=Z, conf=conf_full,
        T=T.numpy(), E=E.numpy(),
        case_ids=case_ids,
        train_idx=idx_tr, val_idx=idx_va,
    )
    log(f"  Latent Z saved: {latent_path}  (shape: {Z.shape})")
 
    # --- Persist checkpoint + history ---
    ckpt_path = artifacts_dir / "phase_6_vae_checkpoint.pt"
    torch.save({
        'model_state': vae.state_dict(),
        'model_name':  vae.name,
        'n_parameters': vae.n_parameters(),
        'train_cfg': train_cfg.__dict__,
    }, ckpt_path)
 
    history_path = artifacts_dir / "phase_6_vae_history.json"
    with open(history_path, 'w') as f:
        json.dump({
            'stage_A': result['stage_A_history'],
            'stage_B': result['stage_B_history'],
        }, f, indent=2, default=str)
 
    # --- Summary dataframe ---
    summary_row = {
        'model':            vae.name,
        'n_parameters':     vae.n_parameters(),
        'd_latent':         Z.shape[1],
        'n_cases':          N,
        'n_events':         int(y['event'].sum()),
        'stage_a_epochs':   len(result['stage_A_history']),
        'stage_b_epochs':   len(result['stage_B_history']),
        'elapsed_s':        round(elapsed, 2),
        'artifact_path':    str(latent_path),
    }
    results_df = pd.DataFrame([summary_row])
    results_df.to_csv(run_dir / "phase_6_fusion_proc.csv", index=False)
    log(f"  Summary: {summary_row}")
 
    return results_df
 
 
# ============================================================
# PHASE 7 — TURBOLATENT (rotation + PTQ over frozen Z)
# ============================================================
 
def phase_7_turbolatent(
    df_features: pd.DataFrame,
    df_targets: pd.DataFrame,
    config: dict,
    run_dir: Path,
) -> Optional[pd.DataFrame]:
    """
    TurboLatent: apply rotation (Hadamard or SVD variant) + uniform PTQ
    to the frozen latent Z, measure C-index degradation as a function of bit width.
 
    Resolves Z artifact via artifact discovery precedence — if Phase 6 ran
    in the current session, uses that; otherwise falls back to explicit
    path from config, or the most recent prior run's artifact.
    """
    phase_cfg = config.get('phase_7_turbolatent', {})
    if not phase_cfg.get('enabled', False):
        log("[PHASE 7] DISABLED")
        return None
 
    log("\n[PHASE 7] TurboLatent (rotation + PTQ on frozen Z)")
 
    from core.registry import get_prognosis_proc
    from sklearn.model_selection import StratifiedKFold
 
    # --- Resolve Z artifact ---
    latent_path = resolve_artifact_path(
        artifact_name='phase_6_latent_z.npz',
        current_run_dir=run_dir,
        phase_cfg=phase_cfg,
        output_base_dir=Path(config['output']['base_dir']),
    )
    if latent_path is None:
        log("[PHASE 7] SKIPPED — no Z artifact found. Run Phase 6 first or "
            "specify source_artifact_path in the config.")
        return None
    log(f"  Z artifact: {latent_path}")
 
    data = np.load(latent_path, allow_pickle=True)
    Z = data['Z'].astype(np.float32)
    T = data['T']; E = data['E']
    N, D = Z.shape
 
    variants = phase_cfg.get('variants', ['hadamard', 'svd'])
    bit_widths = phase_cfg.get('bit_widths', [8, 6, 4, 3])
    include_baseline = phase_cfg.get('include_baseline_no_rotation', True)
 
    seeds = config['random']['seeds']
    n_folds = config['random']['n_folds']
    prognosis_name = phase_cfg.get(
        'prognosis_proc', 'prognosis_baseline_linear_cox'
    )
 
    # --- Rotation helpers (kept local to the runner, mirror turbolatent.py) ---
    def make_hadamard(d: int) -> np.ndarray:
        """Block-diagonal Walsh-Hadamard of dimension d (d need not be power of 2)."""
        def H_pow2(n: int) -> np.ndarray:
            assert n & (n - 1) == 0
            H = np.array([[1.0]])
            while H.shape[0] < n:
                H = np.block([[H, H], [H, -H]])
            return H / np.sqrt(n)
        # Largest power of 2 ≤ d as primary block, remainder as smaller block
        k = 1
        while (k << 1) <= d: k <<= 1
        remainder = d - k
        if remainder == 0:
            return H_pow2(d)
        k2 = 1
        while (k2 << 1) <= remainder: k2 <<= 1
        # Block-diag
        R = np.zeros((d, d), dtype=np.float32)
        R[:k, :k] = H_pow2(k)
        if remainder > 0:
            if remainder == k2:
                R[k:, k:] = H_pow2(k2)
            else:
                # Pad to k2 via identity on the residual
                R[k:k+k2, k:k+k2] = H_pow2(k2)
                R[k+k2:, k+k2:] = np.eye(remainder - k2)
        return R
 
    def make_svd_rotation(Z: np.ndarray) -> np.ndarray:
        """Data-driven rotation via SVD of the centered Z."""
        Zc = Z - Z.mean(axis=0, keepdims=True)
        _, _, Vt = np.linalg.svd(Zc, full_matrices=False)
        return Vt  # [D, D] orthogonal
 
    def quantize_uniform(x: np.ndarray, bits: int) -> np.ndarray:
        """Per-dimension min-max uniform PTQ to int + dequantize."""
        x_min = x.min(axis=0, keepdims=True)
        x_max = x.max(axis=0, keepdims=True)
        scale = (x_max - x_min) / max(1, (2 ** bits - 1))
        scale = np.where(scale > 0, scale, 1.0)
        q = np.round((x - x_min) / scale)
        q = np.clip(q, 0, 2 ** bits - 1)
        return q * scale + x_min
 
    # --- Evaluation harness ---
    def eval_cox_cv(X: np.ndarray) -> Tuple[float, float]:
        """Return (mean C-index, std) across seeds × folds."""
        seed_means = []
        for s in seeds:
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=s)
            fold_cis = []
            for tr_idx, va_idx in skf.split(np.zeros(N), E):
                X_tr = torch.tensor(X[tr_idx], dtype=torch.float32)
                X_va = torch.tensor(X[va_idx], dtype=torch.float32)
                T_tr = torch.tensor(T[tr_idx], dtype=torch.float32)
                T_va = torch.tensor(T[va_idx], dtype=torch.float32)
                E_tr = torch.tensor(E[tr_idx], dtype=torch.float32)
                E_va = torch.tensor(E[va_idx], dtype=torch.float32)
                model = get_prognosis_proc(prognosis_name, fused_dim=D)
                res = model.fit(
                    X_tr, T_tr, E_tr, X_va, T_va, E_va,
                    epochs=phase_cfg.get('epochs', 200),
                    patience=phase_cfg.get('patience', 20),
                    verbose=False,
                )
                fold_cis.append(res['best_val_cindex'])
            seed_means.append(float(np.mean(fold_cis)))
        return float(np.mean(seed_means)), float(np.std(seed_means))
 
    # --- Run all (variant × bits) combinations ---
    rows = []
 
    if include_baseline:
        log("  Baseline: no rotation, FP32")
        ci_mean, ci_std = eval_cox_cv(Z)
        rows.append({
            'variant': 'baseline', 'bits': 'fp32',
            'cindex_mean': ci_mean, 'cindex_std': ci_std,
        })
        log(f"    C-index: {ci_mean:.4f} ± {ci_std:.4f}")
 
    for variant in variants:
        if variant == 'hadamard':
            R = make_hadamard(D)
            Z_rot = Z @ R
        elif variant == 'svd':
            R = make_svd_rotation(Z)
            Z_rot = Z @ R.T  # right-multiply by V^T
        else:
            log(f"  Unknown variant: {variant}, skipping")
            continue
 
        # FP32 rotated (rotation-only, no quantization)
        ci_mean, ci_std = eval_cox_cv(Z_rot)
        rows.append({
            'variant': variant, 'bits': 'fp32_rotated',
            'cindex_mean': ci_mean, 'cindex_std': ci_std,
        })
        log(f"  {variant} FP32-rotated: C-index {ci_mean:.4f} ± {ci_std:.4f}")
 
        for bits in bit_widths:
            Z_q = quantize_uniform(Z_rot, bits=bits)
            ci_mean, ci_std = eval_cox_cv(Z_q)
            rows.append({
                'variant': variant, 'bits': int(bits),
                'cindex_mean': ci_mean, 'cindex_std': ci_std,
            })
            log(f"  {variant} INT{bits}:     C-index {ci_mean:.4f} ± {ci_std:.4f}")
 
    results_df = pd.DataFrame(rows)
    results_df.to_csv(run_dir / "phase_7_turbolatent.csv", index=False)
    return results_df
 
 
# ============================================================
# PHASE 8 — PROGNOSIS-PROC BENCHMARK (Cox vs Weibull on frozen Z)
# ============================================================
 
def phase_8_prognosis_benchmark(
    df_features: pd.DataFrame,
    df_targets: pd.DataFrame,
    config: dict,
    run_dir: Path,
) -> Optional[pd.DataFrame]:
    """
    Benchmark multiple PROGNOSIS-PROC implementations on the frozen Z.
    Default set: linear_cox vs weibull_head. Extend via config 'models' list.
    """
    phase_cfg = config.get('phase_8_prognosis_benchmark', {})
    if not phase_cfg.get('enabled', False):
        log("[PHASE 8] DISABLED")
        return None
 
    log("\n[PHASE 8] PROGNOSIS-PROC benchmark")
 
    from core.registry import get_prognosis_proc
    from sklearn.model_selection import StratifiedKFold
 
    latent_path = resolve_artifact_path(
        artifact_name='phase_6_latent_z.npz',
        current_run_dir=run_dir,
        phase_cfg=phase_cfg,
        output_base_dir=Path(config['output']['base_dir']),
    )
    if latent_path is None:
        log("[PHASE 8] SKIPPED — no Z artifact found.")
        return None
    log(f"  Z artifact: {latent_path}")
 
    data = np.load(latent_path, allow_pickle=True)
    Z = torch.tensor(data['Z'].astype(np.float32))
    T = torch.tensor(data['T'].astype(np.float32))
    E = torch.tensor(data['E'].astype(np.float32))
    N, D = Z.shape
 
    seeds = config['random']['seeds']
    n_folds = config['random']['n_folds']
    epochs = phase_cfg.get('epochs', 200)
    patience = phase_cfg.get('patience', 20)
    models_cfg = phase_cfg.get('models', [
        {'name': 'prognosis_baseline_linear_cox'},
        {'name': 'prognosis_weibull_head'},
    ])
 
    rows = []
    for mcfg in models_cfg:
        model_name = mcfg['name']
        model_params = mcfg.get('params', {})
        seed_means = []
        for s in seeds:
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=s)
            fold_cis = []
            for tr_idx, va_idx in skf.split(np.zeros(N), E.numpy()):
                model = get_prognosis_proc(
                    model_name, fused_dim=D, **model_params
                )
                res = model.fit(
                    Z[tr_idx], T[tr_idx], E[tr_idx],
                    Z[va_idx], T[va_idx], E[va_idx],
                    epochs=epochs, patience=patience, verbose=False,
                )
                fold_cis.append(res['best_val_cindex'])
            seed_means.append(float(np.mean(fold_cis)))
            rows.append({
                'model': model_name, 'seed': int(s),
                'cindex_mean_folds': float(np.mean(fold_cis)),
            })
        m_arr = np.array(seed_means)
        log(f"  {model_name:35s}  C-index {m_arr.mean():.4f} ± "
            f"{m_arr.std():.4f}  (median {np.median(m_arr):.4f})")
 
    results_df = pd.DataFrame(rows)
    results_df.to_csv(run_dir / "phase_8_prognosis_benchmark.csv", index=False)
    return results_df



# ============================================================
# MAIN ENTRY POINT
# ============================================================

def run_experiment(config_input: Union[str, Path, dict] = "experiment_config.yaml") -> dict:
    """
    Main experiment runner. Accepts a path to a YAML config or a config dictionary directly.
    Executes all enabled phases and returns a summary dictionary.
    """
    if isinstance(config_input, dict):
        config = config_input
        config_path = None
    else:
        config_path = Path(config_input).resolve()
        if not config_path.exists():
            raise FileNotFoundError(f"Experiment config not found: {config_path}")
        
        with open(config_path) as f:
            config = yaml.safe_load(f)
    
    # Resolve feature_config relative to experiment_config dir if not absolute
    feat_path = Path(config['data']['feature_config'])
    if not feat_path.is_absolute():
        # If we have a config_path, resolve relative to it. 
        # Otherwise assume relative to current working directory.
        base_dir = config_path.parent if config_path else Path.cwd()
        feat_path = (base_dir / feat_path).resolve()
        config['data']['feature_config'] = str(feat_path)
    
    verbosity = config.get('runtime', {}).get('verbosity', 'normal')
    
    # ---- Run setup ----
    run_dir = create_run_directory(config, config_path)
    
    print("=" * 70)
    print(f"CLINICAL-CORE / TABULAR-CONN EXPERIMENT")
    print(f"Name:      {config['experiment']['name']}")
    print(f"Hash:      {compute_config_hash(config)}")
    print(f"Run dir:   {run_dir}")
    print("=" * 70)
    
    t_start = time.time()
    summary = {'phases': {}, 'errors': []}
    
    # ---- Step 0: extraction ----
    log("\n[STEP 0] Extracting clinical data from XMLs")
    extractor = TCGAExtractor(config['data']['feature_config'])
    df_features, df_targets = extractor.extract_cohort(config['data']['xml_dir'])
    
    if config['output'].get('save_raw_extraction', True):
        df_features.to_csv(run_dir / "raw_features.csv")
        df_targets.to_csv(run_dir / "raw_targets.csv")
    
    summary['n_cases'] = int(len(df_features))
    summary['n_features'] = int(df_features.shape[1])
    summary['n_events'] = int(df_targets['event'].sum())
    
    # ---- Phases ----
    fail_fast = config.get('runtime', {}).get('fail_fast', False)
    
    try:
        ph1, best_imp = phase_1_imputation(df_features, df_targets, config, run_dir)
        if ph1 is not None:
            summary['phases']['phase_1'] = {
                'best_strategy': best_imp,
                'best_cindex': float(ph1.loc[ph1['cindex_mean'].idxmax(), 'cindex_mean']),
            }
    except Exception as e:
        summary['errors'].append({'phase': 1, 'error': str(e)})
        if fail_fast: raise
        best_imp = 'knn_5'
    
    try:
        ph2 = phase_2_variants(df_features, df_targets, config, run_dir, best_imp)
        if ph2 is not None:
            ph2_summary = ph2.groupby('variant')['cindex'].agg(['mean', 'std']).round(4)
            summary['phases']['phase_2'] = ph2_summary.to_dict()
        
        # New Phase 2 External Baselines
        ph2_ext = phase_2_external_baselines(df_features, df_targets, config, run_dir, best_imp)
        if ph2_ext is not None:
            ph2_ext_summary = (
                ph2_ext
                .drop(columns=['model_summary'], errors='ignore')
                .groupby('baseline')['cindex']
                .agg(['mean', 'std'])
                .round(4)
            )
            summary['phases']['phase_2_external'] = {
                'cindex_summary': ph2_ext_summary.to_dict(),
                'contract_compliant': False,
                'model_summaries': (
                    ph2_ext
                    .groupby('baseline')['model_summary']
                    .first()
                    .to_dict()
                ),
            }
    except Exception as e:
        summary['errors'].append({'phase': '2_external', 'error': str(e)})
        if fail_fast: raise
    
    try:
        ph3 = phase_3_efficiency(df_features.shape[1], config, run_dir)
        if ph3 is not None:
            summary['phases']['phase_3'] = ph3.to_dict(orient='records')
    except Exception as e:
        summary['errors'].append({'phase': 3, 'error': str(e)})
        if fail_fast: raise
    
    try:
        ph4 = phase_4_stress(df_features, df_targets, config, run_dir, best_imp)
        if ph4 is not None:
            summary['phases']['phase_4'] = ph4.to_dict(orient='records')
    except Exception as e:
        summary['errors'].append({'phase': 4, 'error': str(e)})
        if fail_fast: raise
    
    try:
        ph5 = phase_5_multimodal(df_features, df_targets, config, run_dir)
        if ph5 is not None:
            summary['phases']['phase_5'] = ph5.to_dict(orient='records')
    except Exception as e:
        summary['errors'].append({'phase': 5, 'error': str(e)})
        if fail_fast: raise

    try:
        ph6 = phase_6_fusion_proc(df_features, df_targets, config, run_dir, best_imp)
        if ph6 is not None:
            summary['phases']['phase_6'] = ph6.to_dict(orient='records')
    except Exception as e:
        summary['errors'].append({'phase': 6, 'error': str(e)})
        if fail_fast: raise
 
    try:
        ph7 = phase_7_turbolatent(df_features, df_targets, config, run_dir)
        if ph7 is not None:
            summary['phases']['phase_7'] = ph7.to_dict(orient='records')
    except Exception as e:
        summary['errors'].append({'phase': 7, 'error': str(e)})
        if fail_fast: raise
 
    try:
        ph8 = phase_8_prognosis_benchmark(df_features, df_targets, config, run_dir)
        if ph8 is not None:
            summary['phases']['phase_8'] = ph8.to_dict(orient='records')
    except Exception as e:
        summary['errors'].append({'phase': 8, 'error': str(e)})
        if fail_fast: raise


    #     # --- NUEVO BLOQUE EXPLAINABILITY ---
    # try:
    #     ph_explain = phase_explainability_benchmark(config, run_dir)
    #     if ph_explain is not None:
    #         summary['phases']['phase_explain'] = ph_explain.to_dict(orient='records')
    # except Exception as e:
    #     summary['errors'].append({'phase': 'explain', 'error': str(e)})
    #     if fail_fast: raise
    
    # ---- Final summary ----
    summary['runtime_seconds'] = round(time.time() - t_start, 2)
    summary['run_dir'] = str(run_dir)
    
    with open(run_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print("\n" + "=" * 70)
    print(f"EXPERIMENT COMPLETE in {summary['runtime_seconds']}s")
    print(f"Results: {run_dir}")
    print(f"Errors:  {len(summary['errors'])}")
    print("=" * 70)
    
    return summary


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "experiment_config.yaml"
    run_experiment(config_path)
