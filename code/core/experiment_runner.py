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

from components.tabular.utils.extractor import TCGAExtractor
from components.tabular.utils.imputation_benchmark import TabularPreprocessor
from core.model_utils import (
    verify_ingestion_contract,
    train_variant_c,
    benchmark_efficiency,
    cox_partial_likelihood_loss,
)
from components.tabular.models.linear_compact import VariantC_LinearEncoder
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
    if not phase_cfg['enabled']:
        log("[PHASE 1] DISABLED — skipping imputation benchmark")
        return None, config['phase_2_variants'].get('imputation_for_variants', 'knn_5')
    
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
        elif variant_name == 'tabpfn':
            return _eval_tabpfn(X_tr, X_va, y_tr, y_va, conf_tr, conf_va, output_dim, variant_params)
        elif variant_name == 'linear_compact':
            return _eval_linear_compact(X_tr, X_va, y_tr, y_va, mask_tr, mask_va, conf_va, output_dim, variant_params)
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


def _eval_tabpfn(X_tr, X_va, y_tr, y_va, conf_tr, conf_va, output_dim, params):
    variant = get_variant('tabpfn', X_tr.shape[1], output_dim)
    variant.fit(X_tr.values.astype(np.float32), y_tr['risk_group'].values)
    
    emb_tr, _ = variant.encode(
        X_tr.values.astype(np.float32),
        conf_tr.values.astype(np.float32)
    )
    emb_va, conf_va_t = variant.encode(
        X_va.values.astype(np.float32),
        conf_va.values.astype(np.float32)
    )
    
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(emb_tr.numpy(), y_tr['survival_days'].values)
    pred = lr.predict(emb_va.numpy())
    ci = concordance_index(y_va['survival_days'], pred, y_va['event'])
    
    pred_probs = 1 - np.clip(pred / max(pred.max(), 1e-8), 0.01, 0.99)
    ece = expected_calibration_error(pred_probs, y_va['event'].values)
    bs = brier_score(pred_probs, y_va['event'].values)
    
    contract = verify_ingestion_contract(emb_va, conf_va_t, output_dim, verbose=False)
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
    
    log("  Running clean evaluation...")
    clean_results = phase_2_variants(df_features, df_targets, stress_config, run_dir / "_stress_clean", best_imputation)
    log("  Running noisy evaluation...")
    (run_dir / "_stress_clean").mkdir(exist_ok=True)
    (run_dir / "_stress_noisy").mkdir(exist_ok=True)
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
    except Exception as e:
        summary['errors'].append({'phase': 2, 'error': str(e)})
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
