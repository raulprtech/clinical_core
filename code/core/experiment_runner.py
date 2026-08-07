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
from sklearn.preprocessing import StandardScaler
from lifelines import CoxPHFitter, KaplanMeierFitter
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


def validate_clinical_moment(
    config: dict,
    phase_cfg: dict,
    modalities,
) -> str:
    """Enforce that pathology-derived modalities are post-surgery only."""
    context_cfg = config.get('clinical_context', {})
    moment = phase_cfg.get(
        'clinical_moment', context_cfg.get('moment', 'post_surgery')
    )
    if moment not in {'pre_surgery', 'post_surgery'}:
        raise ValueError(
            "clinical moment must be 'pre_surgery' or 'post_surgery'"
        )
    pathology_modalities = set(
        phase_cfg.get(
            'pathology_modalities',
            context_cfg.get('pathology_modalities', ['text']),
        )
    )
    forbidden = pathology_modalities.intersection(modalities)
    if moment == 'pre_surgery' and forbidden:
        raise ValueError(
            "Pre-surgery evaluation cannot use pathology-derived modalities: "
            f"{sorted(forbidden)}"
        )
    return moment




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


def survival_ipcw_calibration(
    predicted_event_probability: np.ndarray,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    horizon: float,
    n_bins: int = 5,
) -> Tuple[float, float]:
    """IPCW Brier score and weighted calibration error at a fixed horizon."""
    train_time = y_train['survival_days'].to_numpy(dtype=float)
    train_event = y_train['event'].to_numpy(dtype=int)
    test_time = y_test['survival_days'].to_numpy(dtype=float)
    test_event = y_test['event'].to_numpy(dtype=int)
    predicted = np.asarray(predicted_event_probability, dtype=float)

    censoring_km = KaplanMeierFitter().fit(
        train_time,
        event_observed=1 - train_event,
    )
    epsilon = 1e-8
    g_horizon = max(float(censoring_km.predict(horizon)), epsilon)
    event_before = (test_event == 1) & (test_time <= horizon)
    known_survivor = test_time > horizon
    weights = np.zeros(len(test_time), dtype=float)
    observed = np.zeros(len(test_time), dtype=float)
    observed[event_before] = 1.0
    if event_before.any():
        just_before = np.nextafter(test_time[event_before], -np.inf)
        g_event = np.asarray(censoring_km.predict(just_before), dtype=float)
        weights[event_before] = 1.0 / np.maximum(g_event, epsilon)
    weights[known_survivor] = 1.0 / g_horizon

    brier = float(np.sum(weights * (predicted - observed) ** 2) / len(test_time))
    total_weight = float(weights.sum())
    if total_weight <= 0:
        return float('nan'), brier
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for idx in range(n_bins):
        upper = predicted <= edges[idx + 1] if idx == n_bins - 1 else predicted < edges[idx + 1]
        mask = (predicted >= edges[idx]) & upper & (weights > 0)
        if not mask.any():
            continue
        bin_weight = float(weights[mask].sum())
        predicted_mean = float(np.average(predicted[mask], weights=weights[mask]))
        observed_mean = float(np.average(observed[mask], weights=weights[mask]))
        ece += (bin_weight / total_weight) * abs(observed_mean - predicted_mean)
    return float(ece), brier


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
                    curves_ctx={                              # NEW
                        'run_dir': run_dir,
                        'seed': seed,
                        'fold': fold_idx,
                        'variant_name': variant_name,
                    },
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

def _maybe_dump_curves(result: dict, curves_ctx: Optional[dict]) -> None:
    """
    If curves_ctx is provided with all required keys, dump the per-epoch
    training and validation losses to a JSON file under
    `<run_dir>/training_curves/seed{seed}_fold{fold}_{variant}.json`.

    The dump is a no-op if curves_ctx is None or any required key is missing,
    so existing call sites that pass `curves_ctx=None` (or omit it) remain
    fully backward-compatible.
    """
    if not curves_ctx:
        return
    required = {'run_dir', 'seed', 'fold', 'variant_name'}
    if not required.issubset(curves_ctx.keys()):
        return
    import json
    out_dir = Path(curves_ctx['run_dir']) / 'training_curves'
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"seed{curves_ctx['seed']}_fold{curves_ctx['fold']}_{curves_ctx['variant_name']}.json"
    payload = {
        'seed': curves_ctx['seed'],
        'fold': curves_ctx['fold'],
        'variant': curves_ctx['variant_name'],
        'train_loss_history': result.get('train_loss_history', []),
        'val_loss_history': result.get('val_loss_history', []),
        'best_epoch': result.get('best_epoch', None),
        'n_epochs_run': result.get('n_epochs_run', None),
        'best_val_cindex': result.get('best_val_cindex', None),
    }
    with open(out_dir / fname, 'w') as f:
        json.dump(payload, f, indent=2)

def _evaluate_variant(
    variant_name, input_dim, output_dim, variant_params,
    X_tr, X_va, y_tr, y_va,
    mask_tr, mask_va, conf_tr, conf_va,
    median_surv,
    curves_ctx: Optional[dict] = None,
    artifacts_dir: Optional[Path] = None,
    artifacts_key: str = '',
) -> Tuple[float, float, float, bool]:
    """Evaluate a single variant on a single fold. Returns (cindex, ece, brier, contract_ok)."""
    try:
        if variant_name == 'cox_baseline':
            return _eval_cox_baseline(
                X_tr, X_va, y_tr, y_va, conf_va, output_dim, median_surv,
                artifacts_dir=artifacts_dir, artifacts_key=artifacts_key,
            )
        elif variant_name == 'linear_compact':
            return _eval_linear_compact(
                X_tr, X_va, y_tr, y_va, mask_tr, mask_va, conf_va,
                output_dim, variant_params,
                curves_ctx=curves_ctx,
                artifacts_dir=artifacts_dir, artifacts_key=artifacts_key,
            )
        elif variant_name == 'ft_transformer':
            return _eval_ft_transformer(
                X_tr, X_va, y_tr, y_va,
                mask_tr, mask_va, conf_va, output_dim,
                variant_params.get(variant_name, {}),
                curves_ctx=curves_ctx,
                artifacts_dir=artifacts_dir, artifacts_key=artifacts_key,
            )
        else:
            log(f"    {variant_name}: no evaluator registered, skipping", level="debug")
            return np.nan, np.nan, np.nan, False
    except Exception as e:
        log(f"    {variant_name} failed: {e}", level="debug")
        return np.nan, np.nan, np.nan, False


def _save_predictions_npz(
    artifacts_dir: Optional[Path],
    artifacts_key: str,
    *,
    case_ids_train, risk_train, times_train, events_train,
    case_ids_holdout, risk_holdout, times_holdout, events_holdout,
    surv_func_holdout=None, surv_time_grid=None,
    extra: Optional[dict] = None,
) -> None:
    """Persist per-patient predictions for post-hoc bootstrap / DeLong / calibration."""
    if artifacts_dir is None or not artifacts_key:
        return
    out_dir = Path(artifacts_dir) / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        'case_ids_train': np.asarray(case_ids_train).astype('U64'),
        'risk_train': np.asarray(risk_train, dtype=np.float64),
        'times_train': np.asarray(times_train, dtype=np.float64),
        'events_train': np.asarray(events_train, dtype=np.int64),
        'case_ids_holdout': np.asarray(case_ids_holdout).astype('U64'),
        'risk_holdout': np.asarray(risk_holdout, dtype=np.float64),
        'times_holdout': np.asarray(times_holdout, dtype=np.float64),
        'events_holdout': np.asarray(events_holdout, dtype=np.int64),
    }
    if surv_func_holdout is not None:
        payload['surv_func_holdout'] = np.asarray(surv_func_holdout, dtype=np.float64)
        payload['surv_time_grid'] = np.asarray(surv_time_grid, dtype=np.float64)
    if extra:
        for k, v in extra.items():
            payload[k] = np.asarray(v)
    np.savez_compressed(out_dir / f"{artifacts_key}.npz", **payload)


def _save_cox_checkpoint(artifacts_dir, artifacts_key, cph, scaler, feature_cols):
    if artifacts_dir is None or not artifacts_key:
        return
    import pickle
    out_dir = Path(artifacts_dir) / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{artifacts_key}.pkl", 'wb') as f:
        pickle.dump({
            'kind': 'cox_baseline',
            'cph': cph,
            'scaler': scaler,
            'feature_cols': list(feature_cols),
        }, f)


def _save_neural_checkpoint(artifacts_dir, artifacts_key, encoder, risk_head, feature_cols,
                            variant_name, variant_kwargs):
    if artifacts_dir is None or not artifacts_key:
        return
    import pickle
    out_dir = Path(artifacts_dir) / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{artifacts_key}.pkl", 'wb') as f:
        pickle.dump({
            'kind': variant_name,
            'encoder_state_dict': {k: v.cpu() for k, v in encoder.state_dict().items()},
            'risk_head_state_dict': {k: v.cpu() for k, v in risk_head.state_dict().items()},
            'feature_cols': list(feature_cols),
            'variant_kwargs': variant_kwargs,
        }, f)


def _eval_cox_baseline(X_tr, X_va, y_tr, y_va, conf_va, output_dim, median_surv,
                       artifacts_dir: Optional[Path] = None, artifacts_key: str = ''):
    """
    Cox baseline aligned with diagnostic_cox_raw.py protocol.
 
    Pipeline applied per fold:
      1. Drop columns with >90% missingness in train (high-missing defense).
      2. Replace +/-inf with NaN; drop train rows with residual NaN
         (TabularPreprocessor should have imputed these; defensive).
      3. Drop columns with variance < 1e-8 post-imputation.
      4. StandardScaler fit on train, applied to val.
      5. CoxPHFitter with adaptive penalizer (escalates if Newton-Raphson
         fails to converge).
    """
    # ---- 1. Drop high-missing columns based on train fold only ----
    miss_frac = X_tr.isna().mean(axis=0)
    keep_cols = miss_frac[miss_frac < 0.90].index.tolist()
    X_tr_f = X_tr[keep_cols].copy()
    X_va_f = X_va[keep_cols].copy()
 
    # ---- 2. Sanitize inf/NaN ----
    X_tr_f = X_tr_f.replace([np.inf, -np.inf], np.nan)
    X_va_f = X_va_f.replace([np.inf, -np.inf], np.nan)
 
    valid_rows = ~X_tr_f.isna().any(axis=1)
    X_tr_f = X_tr_f.loc[valid_rows]
    y_tr_f = y_tr.loc[valid_rows]
    X_va_f = X_va_f.fillna(0)
 
    # ---- 3. Drop near-constant columns post-imputation ----
    var_cols = X_tr_f.var(axis=0)
    keep_var = var_cols[var_cols > 1e-8].index.tolist()
    X_tr_f = X_tr_f[keep_var]
    X_va_f = X_va_f[keep_var]
 
    if len(keep_var) == 0:
        return np.nan, np.nan, np.nan, False
 
    # ---- 4. Standardize: fit on train, transform on val ----
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr_f.values)
    X_va_s = sc.transform(X_va_f.values)
 
    # ---- 5. Cox with adaptive penalizer ----
    cox_df_tr = pd.DataFrame(X_tr_s, columns=keep_var)
    cox_df_tr['T'] = y_tr_f['survival_days'].values
    cox_df_tr['E'] = y_tr_f['event'].values
    cox_df_va = pd.DataFrame(X_va_s, columns=keep_var)
 
    cph = None
    for pen_try in [0.5, 1.0, 5.0, 20.0]:
        try:
            cph = CoxPHFitter(penalizer=pen_try, l1_ratio=0.0)
            cph.fit(
                cox_df_tr, duration_col='T', event_col='E',
                show_progress=False,
            )
            break
        except Exception:
            cph = None
    if cph is None:
        return np.nan, np.nan, np.nan, False
 
    # ---- 6. Risk scores and C-index on validation ----
    risk = cph.predict_partial_hazard(cox_df_va).values.ravel()
    risk_tr = cph.predict_partial_hazard(cox_df_tr).values.ravel()
    ci = concordance_index(
        y_va['survival_days'].values, -risk, y_va['event'].values
    )

    # ---- 7. Calibration metrics (ECE, Brier) on validation ----
    horizon = float(median_surv)
    surv_at_horizon = cph.predict_survival_function(
        cox_df_va, times=[horizon],
    ).iloc[0].to_numpy(dtype=float)
    pred_probs = 1.0 - surv_at_horizon
    pred_probs = np.clip(pred_probs, 0.01, 0.99)
    ece, bs = survival_ipcw_calibration(pred_probs, y_tr_f, y_va, horizon)
    surv_func = cph.predict_survival_function(cox_df_va)

    # ---- 7b. Persist artifacts (predictions + checkpoint) ----
    _save_predictions_npz(
        artifacts_dir, artifacts_key,
        case_ids_train=y_tr_f.index.astype(str).values,
        risk_train=risk_tr,
        times_train=y_tr_f['survival_days'].values,
        events_train=y_tr_f['event'].values,
        case_ids_holdout=y_va.index.astype(str).values,
        risk_holdout=risk,
        times_holdout=y_va['survival_days'].values,
        events_holdout=y_va['event'].values,
        surv_func_holdout=surv_func.values,
        surv_time_grid=surv_func.index.values,
    )
    _save_cox_checkpoint(artifacts_dir, artifacts_key, cph, sc, keep_var)

    # ---- 8. Contract verification (unchanged behavior) ----
    variant = get_variant('cox_baseline', X_va.shape[1], output_dim)
    emb, conf_t = variant.encode(
        X_va.values.astype(np.float32),
        conf_va.values.astype(np.float32),
    )
    contract = verify_ingestion_contract(emb, conf_t, output_dim, verbose=False)
    return ci, ece, bs, contract.get('contract_satisfied', False)


def _eval_linear_compact(X_tr, X_va, y_tr, y_va, mask_tr, mask_va, conf_va, output_dim, params,
                         curves_ctx: Optional[dict] = None,
                         artifacts_dir: Optional[Path] = None, artifacts_key: str = ''):
    variant_kwargs = {'hidden_dim': params.get('hidden_dim', 128)}
    encoder = get_variant(
        'linear_compact',
        input_dim=X_tr.shape[1],
        output_dim=output_dim,
        **variant_kwargs,
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
        weight_decay=params.get('weight_decay', 0.0001),
        verbose=False,
    )
    ci = result['best_val_cindex']

    # NEW: dump training curves to disk if context provided
    _maybe_dump_curves(result, curves_ctx)

    encoder.eval()
    with torch.no_grad():
        emb_tr, _ = encoder(X_tr_t, M_tr)
        risk_tr = result['risk_head'](emb_tr).squeeze(-1).numpy()
        emb, conf_t = encoder(X_va_t, M_va)
        risk = result['risk_head'](emb).squeeze(-1).numpy()
    pred_probs = 1 / (1 + np.exp(-risk))
    pred_probs = np.clip(pred_probs, 0.01, 0.99)
    ece = expected_calibration_error(pred_probs, y_va['event'].values)
    bs = brier_score(pred_probs, y_va['event'].values)

    _save_predictions_npz(
        artifacts_dir, artifacts_key,
        case_ids_train=y_tr.index.astype(str).values,
        risk_train=risk_tr,
        times_train=y_tr['survival_days'].values,
        events_train=y_tr['event'].values,
        case_ids_holdout=y_va.index.astype(str).values,
        risk_holdout=risk,
        times_holdout=y_va['survival_days'].values,
        events_holdout=y_va['event'].values,
    )
    _save_neural_checkpoint(
        artifacts_dir, artifacts_key, encoder, result['risk_head'],
        feature_cols=list(X_tr.columns),
        variant_name='linear_compact',
        variant_kwargs={**variant_kwargs, 'output_dim': output_dim, 'input_dim': X_tr.shape[1]},
    )

    contract = verify_ingestion_contract(emb, conf_t, output_dim, verbose=False)
    return ci, ece, bs, contract.get('contract_satisfied', False)
    
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
    curves_ctx: Optional[dict] = None,    # NEW
    artifacts_dir: Optional[Path] = None, artifacts_key: str = '',
):
    variant_kwargs = {
        'd_token': params.get('d_token', 192),
        'n_blocks': params.get('n_blocks', 3),
        'n_heads': params.get('n_heads', 8),
        'd_ff': params.get('d_ff', None),
        'dropout': params.get('dropout', 0.1),
    }
    encoder = get_variant(
        'ft_transformer',
        input_dim=X_tr.shape[1],
        output_dim=output_dim,
        **variant_kwargs,
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
        encoder, X_tr_t, M_tr, T_tr, E_tr,
        X_va_t, M_va, T_va, E_va,
        epochs=params.get('epochs', 200),
        lr=params.get('lr', 3e-4),
        patience=params.get('patience', 20),
        weight_decay=params.get('weight_decay', 1e-4),
        verbose=False,
    )
    ci = result['best_val_cindex']

    # NEW: dump training curves to disk if context provided
    _maybe_dump_curves(result, curves_ctx)

    encoder.eval()
    with torch.no_grad():
        emb_tr, _ = encoder(X_tr_t, M_tr)
        risk_tr = result['risk_head'](emb_tr).squeeze(-1).numpy()
        emb, conf_t = encoder(X_va_t, M_va)
        risk = result['risk_head'](emb).squeeze(-1).numpy()

    pred_probs = 1 / (1 + np.exp(-risk))
    pred_probs = np.clip(pred_probs, 0.01, 0.99)
    ece = expected_calibration_error(pred_probs, y_va['event'].values)
    bs = brier_score(pred_probs, y_va['event'].values)

    _save_predictions_npz(
        artifacts_dir, artifacts_key,
        case_ids_train=y_tr.index.astype(str).values,
        risk_train=risk_tr,
        times_train=y_tr['survival_days'].values,
        events_train=y_tr['event'].values,
        case_ids_holdout=y_va.index.astype(str).values,
        risk_holdout=risk,
        times_holdout=y_va['survival_days'].values,
        events_holdout=y_va['event'].values,
    )
    _save_neural_checkpoint(
        artifacts_dir, artifacts_key, encoder, result['risk_head'],
        feature_cols=list(X_tr.columns),
        variant_name='ft_transformer',
        variant_kwargs={**variant_kwargs, 'output_dim': output_dim, 'input_dim': X_tr.shape[1]},
    )

    contract = verify_ingestion_contract(emb, conf_t, output_dim, verbose=False)
    return ci, ece, bs, contract.get('contract_satisfied', False)


# ============================================================
# COHORT FILTER — optional global filter applied after extraction
# ============================================================

def apply_cohort_filter(
    df_features: pd.DataFrame,
    df_targets: pd.DataFrame,
    cohort_cfg: dict,
    run_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Optional cohort filter applied after extraction and before any phase.

    Filters (all optional, applied conjunctively):
      - require_dfs_valid: keep only cases with df_targets['dfs_valid'] == True
      - modality_manifest_path + require_modalities: with modality_policy set
        to ``intersection`` (legacy), keep only cases where every requested
        modality is present. With ``per_modality``, record availability but do
        not shrink the global cohort; each modality combination selects its own
        cohort later.

    YAML structure:
      cohort_filter:
        enabled: true
        require_dfs_valid: true
        modality_manifest_path: "data/manifests/gdc_modality_manifest_TCGA-KIRC_20260528.csv"
        require_modalities: [wsi, mrna_seq, mirna_seq, methylation_450k]

    Returns (df_features_filtered, df_targets_filtered, manifest_summary). Persists
    `cohort_manifest.json` under run_dir with the full audit trail.
    """
    audit = {
        'n_initial': int(len(df_features)),
        'filters_applied': [],
        'n_after_each_filter': [],
        'dropped_examples': {},
        'n_final': None,
        'case_ids_final': [],
    }

    if not cohort_cfg.get('enabled', False):
        log("[COHORT FILTER] DISABLED")
        audit['n_final'] = audit['n_initial']
        audit['case_ids_final'] = list(df_features.index)
        return df_features, df_targets, audit

    log(f"\n[COHORT FILTER] Initial n = {audit['n_initial']}")

    keep_mask = pd.Series(True, index=df_features.index)

    # Filter 1: DFS validity
    if cohort_cfg.get('require_dfs_valid', False):
        if 'dfs_valid' not in df_targets.columns:
            raise KeyError(
                "cohort_filter.require_dfs_valid=True but df_targets has no 'dfs_valid' "
                "column. Make sure the feature_config YAML includes the DFS targets and "
                "the extractor has been updated."
            )
        dfs_mask = df_targets.reindex(df_features.index)['dfs_valid'].fillna(False).astype(bool)
        n_dropped = int((~dfs_mask).sum())
        keep_mask &= dfs_mask
        n_after = int(keep_mask.sum())
        audit['filters_applied'].append('require_dfs_valid')
        audit['n_after_each_filter'].append(n_after)
        audit['dropped_examples']['require_dfs_valid'] = (
            df_features.index[~dfs_mask].tolist()[:10]
        )
        log(f"  After require_dfs_valid: {n_after} (dropped {n_dropped})")

    # Filter 2: modality availability via GDC manifest. The preferred
    # per_modality policy preserves the largest cohort for every experiment.
    manifest_path = cohort_cfg.get('modality_manifest_path')
    required_modalities = cohort_cfg.get('require_modalities', [])
    modality_policy = cohort_cfg.get('modality_policy', 'intersection')
    if modality_policy not in {'intersection', 'per_modality'}:
        raise ValueError(
            "cohort_filter.modality_policy must be 'intersection' or 'per_modality'"
        )
    if manifest_path and required_modalities:
        manifest_path = Path(manifest_path)
        if not manifest_path.is_absolute():
            manifest_path = Path.cwd() / manifest_path
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"cohort_filter.modality_manifest_path not found: {manifest_path}. "
                f"Build it first with `python3 tools/build_modality_manifest.py`."
            )
        manifest_df = pd.read_csv(manifest_path).set_index('case_id')
        log(f"  Loaded modality manifest: {manifest_path.name} ({len(manifest_df)} cases)")

        missing_cols = [f'has_{m}' for m in required_modalities if f'has_{m}' not in manifest_df.columns]
        if missing_cols:
            raise KeyError(
                f"Modality manifest is missing columns: {missing_cols}. "
                f"Available columns: {list(manifest_df.columns)}"
            )

        # A case is kept only if it's in the manifest AND every required has_<m> is True
        in_manifest = df_features.index.isin(manifest_df.index)
        per_case_pass = pd.Series(False, index=df_features.index)
        for cid in df_features.index[in_manifest]:
            row = manifest_df.loc[cid]
            per_case_pass.loc[cid] = bool(
                all(row.get(f'has_{m}', False) for m in required_modalities)
            )
        audit['modality_policy'] = modality_policy
        audit['modality_manifest_path'] = str(manifest_path)
        audit['modality_counts'] = {
            modality: int(
                manifest_df.get(
                    f'has_{modality}', pd.Series(False, index=manifest_df.index)
                ).fillna(False).astype(bool).sum()
            )
            for modality in required_modalities
        }
        if modality_policy == 'intersection':
            # Cases absent from the manifest are missing under strict policy.
            n_dropped = int((~per_case_pass).sum())
            keep_mask &= per_case_pass
            n_after = int(keep_mask.sum())
            audit['filters_applied'].append(
                f"modalities:{','.join(required_modalities)}"
            )
            audit['n_after_each_filter'].append(n_after)
            audit['dropped_examples']['modalities'] = (
                df_features.index[~per_case_pass].tolist()[:10]
            )
            log(f"  After modality intersection: {n_after} (dropped {n_dropped})")
        else:
            log(
                "  Modality policy per_modality: global cohort preserved; "
                "availability will be applied per experiment subset"
            )

    df_features_f = df_features.loc[keep_mask].copy()
    df_targets_f = df_targets.reindex(df_features_f.index).copy()

    audit['n_final'] = int(len(df_features_f))
    audit['case_ids_final'] = list(df_features_f.index)

    out_path = run_dir / "cohort_manifest.json"
    with open(out_path, 'w') as f:
        json.dump(audit, f, indent=2, default=str)
    log(f"  Final n = {audit['n_final']}. Manifest: {out_path}")

    return df_features_f, df_targets_f, audit


# ============================================================
# PHASE 2 HOLDOUT — single train/holdout 80/20 per seed
# ============================================================

def phase_2_holdout(
    df_features: pd.DataFrame,
    df_targets: pd.DataFrame,
    config: dict,
    run_dir: Path,
    best_imputation: str,
) -> Optional[pd.DataFrame]:
    """
    Held-out 80/20 evaluation of TABULAR-IN variants under the same
    preprocessing protocol as phase_2_variants, but with a single stratified
    train/holdout split per seed instead of K-fold CV.

    Replicates the protocol of diagnostic_cox_raw.py inside the formal runner
    so the defendable cifra (cox_baseline ≈ 0.8103 ± 0.044 on 5 seeds) becomes
    reproducible without an external script.
    """
    from sklearn.model_selection import train_test_split

    phase_cfg = config.get('phase_2_holdout', {})
    if not phase_cfg.get('enabled', False):
        log("[PHASE 2 HOLDOUT] DISABLED")
        return None

    log("\n[PHASE 2 HOLDOUT] Single train/holdout 80/20 per seed")

    valid = df_targets['survival_days'].notna() & (df_targets['survival_days'] > 0)
    X = df_features.loc[valid].copy()
    y = df_targets.loc[valid].copy()

    seeds = phase_cfg.get('seeds', config['random']['seeds'])
    holdout_fraction = phase_cfg.get('holdout_fraction', 0.20)
    output_dim = phase_cfg.get('output_dim',
                                config.get('phase_2_variants', {}).get('output_dim', 768))
    variants = phase_cfg.get('variants',
                              config.get('phase_2_variants', {}).get('variants',
                                                                      ['cox_baseline']))
    variant_params = phase_cfg.get('variant_params',
                                    config.get('phase_2_variants', {}).get('variant_params', {}))

    imp_for_variants = phase_cfg.get('imputation_for_variants',
                                      config.get('phase_2_variants', {}).get('imputation_for_variants', 'knn_5'))
    if imp_for_variants == "auto":
        imp_for_variants = best_imputation
    imp_for_baseline = phase_cfg.get('imputation_for_baseline',
                                      config.get('phase_2_variants', {}).get('imputation_for_baseline', 'knn_5'))

    # Multi-protocol support: each entry defines a name and an optional
    # list of columns to drop from df_features (e.g. the post-event variables
    # ecog/karnofsky/tumor_status for the "limpio" protocol). If absent,
    # behaviour collapses to a single anonymous protocol on the full feature set.
    protocols_cfg = phase_cfg.get('protocols')
    if not protocols_cfg:
        protocols_cfg = [{'name': '', 'drop_features': []}]
    save_artifacts = bool(phase_cfg.get('save_artifacts', False))
    artifacts_dir = run_dir / "phase2_artifacts" if save_artifacts else None

    log(f"  Cases: {len(X)}, output_dim: {output_dim}")
    log(f"  Holdout fraction: {holdout_fraction}, seeds: {list(seeds)}")
    log(f"  Variants: {variants}")
    log(f"  Imputation for baseline: {imp_for_baseline}")
    log(f"  Imputation for advanced variants: {imp_for_variants}")
    log(f"  Protocols: {[p.get('name','<unnamed>') for p in protocols_cfg]}")
    log(f"  save_artifacts: {save_artifacts}")

    rows = []

    for protocol in protocols_cfg:
        proto_name = str(protocol.get('name', '') or '')
        drop_cols = list(protocol.get('drop_features') or [])
        X_proto = X.drop(columns=[c for c in drop_cols if c in X.columns], errors='ignore')
        proto_tag = proto_name if proto_name else 'default'
        log(f"\n  [Protocol: {proto_tag}] n_features={X_proto.shape[1]} "
            f"(dropped={[c for c in drop_cols if c in X.columns]})")

        for seed in seeds:
            log(f"  Seed {seed}")
            idx_all = np.arange(len(X_proto))
            tr_idx, ho_idx = train_test_split(
                idx_all,
                test_size=holdout_fraction,
                stratify=y['event'].values,
                random_state=int(seed),
            )

            X_tr_raw, X_ho_raw = X_proto.iloc[tr_idx].copy(), X_proto.iloc[ho_idx].copy()
            y_tr, y_ho = y.iloc[tr_idx].copy(), y.iloc[ho_idx].copy()

            # Two preprocessing passes mirroring phase_2_variants: baseline and advanced.
            onehot_features = list(phase_cfg.get('onehot_features') or [])
            prep_base = TabularPreprocessor(onehot_columns=onehot_features)
            X_tr_b, mask_tr_b, conf_tr_b = prep_base.fit_transform(
                X_tr_raw, get_imputation(imp_for_baseline)
            )
            X_ho_b, mask_ho_b, conf_ho_b = prep_base.transform(X_ho_raw)

            prep_adv = TabularPreprocessor(onehot_columns=onehot_features)
            X_tr_a, mask_tr_a, conf_tr_a = prep_adv.fit_transform(
                X_tr_raw, get_imputation(imp_for_variants)
            )
            X_ho_a, mask_ho_a, conf_ho_a = prep_adv.transform(X_ho_raw)

            input_dim = X_tr_a.shape[1]
            calibration_horizon = float(
                phase_cfg.get('calibration_horizon_days')
                or y_tr['survival_days'].median()
            )
            n_events_ho = int(y_ho['event'].sum())
            log(f"    n_train={len(tr_idx)}, n_holdout={len(ho_idx)}, events_holdout={n_events_ho}")

            for variant_name in variants:
                if variant_name == 'cox_baseline':
                    X_tr_use, X_ho_use = X_tr_b, X_ho_b
                    conf_ho_use = conf_ho_b
                else:
                    X_tr_use, X_ho_use = X_tr_a, X_ho_a
                    conf_ho_use = conf_ho_a

                artifacts_key = (
                    f"{proto_tag}_seed{int(seed)}_{variant_name}"
                    if save_artifacts else ''
                )

                ci, ece, bs, contract_ok = _evaluate_variant(
                    variant_name=variant_name,
                    input_dim=input_dim,
                    output_dim=output_dim,
                    variant_params=variant_params.get(variant_name, {}),
                    X_tr=X_tr_use, X_va=X_ho_use,
                    y_tr=y_tr, y_va=y_ho,
                    mask_tr=mask_tr_a, mask_va=mask_ho_a,
                    conf_tr=conf_tr_a, conf_va=conf_ho_use,
                    median_surv=calibration_horizon,
                    curves_ctx={
                        'run_dir': run_dir,
                        'seed': seed,
                        'fold': f'holdout_{proto_tag}',
                        'variant_name': variant_name,
                    },
                    artifacts_dir=artifacts_dir,
                    artifacts_key=artifacts_key,
                )

                rows.append({
                    'protocol': proto_tag,
                    'seed': int(seed),
                    'variant': variant_name,
                    'cindex_holdout': float(ci),
                    'ece': float(ece) if ece == ece else float('nan'),
                    'brier_score': float(bs) if bs == bs else float('nan'),
                    'contract_satisfied': bool(contract_ok),
                    'n_train': int(len(tr_idx)),
                    'n_holdout': int(len(ho_idx)),
                    'events_holdout': n_events_ho,
                    'n_features': int(X_tr_use.shape[1]),
                    'calibration_horizon_days': calibration_horizon,
                })

    df_results = pd.DataFrame(rows)
    df_results.to_csv(run_dir / "phase2_holdout.csv", index=False)

    summary = (
        df_results
        .groupby(['protocol', 'variant'])
        .agg(
            cindex_mean=('cindex_holdout', 'mean'),
            cindex_std=('cindex_holdout', 'std'),
            cindex_median=('cindex_holdout', 'median'),
            ece_mean=('ece', 'mean'),
            brier_mean=('brier_score', 'mean'),
            contract_satisfied=('contract_satisfied', 'all'),
            n_seeds=('cindex_holdout', 'count'),
        )
        .round(4)
    )
    summary.to_csv(run_dir / "phase2_holdout_summary.csv")
    log("\n  HOLDOUT SUMMARY:")
    log(summary.to_string())

    return df_results


# ============================================================
# PHASE 2 EXTERNAL HOLDOUT — single train/holdout 80/20 per seed for externals
# ============================================================

def phase_2_external_holdout(
    df_features: pd.DataFrame,
    df_targets: pd.DataFrame,
    config: dict,
    run_dir: Path,
    best_imputation: str,
) -> Optional[pd.DataFrame]:
    """
    Held-out 80/20 evaluation of non-contractual SOTA baselines (TabPFN, RSF).

    Mirrors phase_2_holdout (stratified single split per seed) but uses the
    existing _eval_tabpfn_external / _eval_rsf_external helpers. Preprocessing
    is identical to the advanced variants of phase_2_holdout (KNN imputation
    fit on the train fold only).

    YAML structure:
      phase_2_external_holdout:
        enabled: true
        holdout_fraction: 0.20
        seeds: [42, 123, 456, 789, 1024]
        imputation: knn_5
        baselines:
          - name: tabpfn_external
            params: {device: 'auto', n_estimators: 4}
          - name: rsf_external
            params: {n_estimators: 100, min_samples_split: 10, min_samples_leaf: 15}
    """
    from sklearn.model_selection import train_test_split

    phase_cfg = config.get('phase_2_external_holdout', {})
    if not phase_cfg.get('enabled', False):
        log("[PHASE 2 EXTERNAL HOLDOUT] DISABLED")
        return None

    log("\n[PHASE 2 EXTERNAL HOLDOUT] Single train/holdout 80/20 per seed (TabPFN / RSF / ...)")

    valid = df_targets['survival_days'].notna() & (df_targets['survival_days'] > 0)
    X = df_features.loc[valid].copy()
    y = df_targets.loc[valid].copy()

    seeds = phase_cfg.get('seeds', config['random']['seeds'])
    holdout_fraction = phase_cfg.get('holdout_fraction', 0.20)
    imp_for_variants = phase_cfg.get('imputation',
                                      config.get('phase_2_variants', {}).get('imputation_for_variants', 'knn_5'))
    if imp_for_variants == "auto":
        imp_for_variants = best_imputation

    baselines = phase_cfg.get('baselines', [
        {'name': 'tabpfn_external', 'params': {'device': 'auto', 'n_estimators': 4}},
        {'name': 'rsf_external', 'params': {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 15}},
    ])

    log(f"  Cases: {len(X)}")
    log(f"  Holdout fraction: {holdout_fraction}, seeds: {list(seeds)}")
    log(f"  Imputation: {imp_for_variants}")
    log(f"  Baselines: {[b['name'] for b in baselines]}")

    rows = []

    for seed in seeds:
        log(f"  Seed {seed}")
        idx_all = np.arange(len(X))
        tr_idx, ho_idx = train_test_split(
            idx_all,
            test_size=holdout_fraction,
            stratify=y['event'].values,
            random_state=int(seed),
        )

        X_tr_raw, X_ho_raw = X.iloc[tr_idx].copy(), X.iloc[ho_idx].copy()
        y_tr, y_ho = y.iloc[tr_idx].copy(), y.iloc[ho_idx].copy()

        prep = TabularPreprocessor()
        X_tr, mask_tr, conf_tr = prep.fit_transform(X_tr_raw, get_imputation(imp_for_variants))
        X_ho, mask_ho, conf_ho = prep.transform(X_ho_raw)

        n_events_ho = int(y_ho['event'].sum())
        log(f"    n_train={len(tr_idx)}, n_holdout={len(ho_idx)}, events_holdout={n_events_ho}")

        for bcfg in baselines:
            name = bcfg['name']
            params = bcfg.get('params', {})

            if name == 'tabpfn_external':
                result = _eval_tabpfn_external(X_tr, X_ho, y_tr, y_ho, params, seed, fold='holdout')
            elif name == 'rsf_external':
                result = _eval_rsf_external(X_tr, X_ho, y_tr, y_ho, params, seed, fold='holdout')
            else:
                log(f"    UNKNOWN external baseline: {name}, skipping", level="warn")
                continue

            rows.append({
                'seed': int(seed),
                'baseline': name,
                'cindex_holdout': float(result['cindex']),
                'contract_compliant': bool(result.get('contract_compliant', False)),
                'n_train': int(len(tr_idx)),
                'n_holdout': int(len(ho_idx)),
                'events_holdout': n_events_ho,
                'model_summary': result.get('model_summary', {}),
            })

    if not rows:
        log("  No external baselines were successfully evaluated.")
        return None

    df_results = pd.DataFrame(rows)
    # Drop nested model_summary from the CSV for cleanliness; keep it in summary.json
    df_results.drop(columns=['model_summary'], errors='ignore').to_csv(
        run_dir / "phase2_external_holdout.csv", index=False
    )

    summary = (
        df_results
        .groupby('baseline')
        .agg(
            cindex_mean=('cindex_holdout', 'mean'),
            cindex_std=('cindex_holdout', 'std'),
            cindex_median=('cindex_holdout', 'median'),
            n_seeds=('cindex_holdout', 'count'),
        )
        .round(4)
    )
    summary.to_csv(run_dir / "phase2_external_holdout_summary.csv")
    log("\n  EXTERNAL HOLDOUT SUMMARY:")
    log(summary.to_string())

    return df_results


# ============================================================
# PHASE 2 MAHOOTIHA — Spearman + RF feature ranking, Cox over top-K
# ============================================================

def phase_2_mahootiha(
    df_features: pd.DataFrame,
    df_targets: pd.DataFrame,
    config: dict,
    run_dir: Path,
    best_imputation: str,
) -> Optional[pd.DataFrame]:
    """
    Replicate the Mahootiha 2024-style feature ranking method on our cohort.

    Method:
      1. Per seed, stratified train/holdout 80/20 split.
      2. On the TRAIN fold only:
         - Spearman correlation of each feature vs survival_days (abs value).
         - Random Forest classifier (binary-at-median target) feature importance.
         - Combined rank = average of (Spearman rank, RF rank), lower = better.
      3. For each K in the configured `k_values` (default [5, 10, 15, all]):
         - Select top-K features by combined rank.
         - Train cox_baseline on top-K (same protocol as phase_2_holdout).
         - Evaluate held-out C-index.
      4. Aggregate per K across seeds.
      5. Persist the per-seed combined rank for every feature so the §5.3 table
         (where ecog/karnofsky/tumor_status would appear) becomes reproducible.

    YAML structure:
      phase_2_mahootiha:
        enabled: true
        holdout_fraction: 0.20
        seeds: [42, 123, 456, 789, 1024]
        k_values: [5, 10, 15, null]    # null → use all features
        imputation: knn_5
        rf_n_estimators: 200

    Per the user's choice (2026-05-27), this phase is intended to be run twice:
      - on the 19 anti-leakage features (to validate Mahootiha's 0.84 under our protocol),
      - on the 22 with-leakage features (to replicate their original setup).
    The relevant comparison is K=10 (Mahootiha's typical cutoff) on both feature sets.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from scipy.stats import spearmanr

    phase_cfg = config.get('phase_2_mahootiha', {})
    if not phase_cfg.get('enabled', False):
        log("[PHASE 2 MAHOOTIHA] DISABLED")
        return None

    log("\n[PHASE 2 MAHOOTIHA] Spearman + RF ranking + Cox per top-K, held-out 80/20 per seed")

    valid = df_targets['survival_days'].notna() & (df_targets['survival_days'] > 0)
    X = df_features.loc[valid].copy()
    y = df_targets.loc[valid].copy()
    median_surv = float(y['survival_days'].median())

    seeds = phase_cfg.get('seeds', config['random']['seeds'])
    holdout_fraction = phase_cfg.get('holdout_fraction', 0.20)
    n_total_features = X.shape[1]

    # k_values: allow None as a sentinel for "all features".
    k_raw = phase_cfg.get('k_values', [5, 10, 15, None])
    k_values = sorted(set(
        n_total_features if k is None else min(int(k), n_total_features) for k in k_raw
    ))

    rf_n_estimators = phase_cfg.get('rf_n_estimators', 200)
    imp_name = phase_cfg.get('imputation', 'knn_5')
    if imp_name == 'auto':
        imp_name = best_imputation
    output_dim = phase_cfg.get('output_dim',
                                config.get('phase_2_variants', {}).get('output_dim', 768))

    log(f"  Cases: {len(X)}, total features: {n_total_features}")
    log(f"  Seeds: {list(seeds)}, K values: {k_values}")
    log(f"  RF n_estimators: {rf_n_estimators}, imputation: {imp_name}")

    # Multi-protocol support (mirrors phase_2_holdout): each entry drops a
    # list of columns from df_features BEFORE the 80/20 split so train/holdout
    # indices stay aligned across protocols for the same seed.
    protocols_cfg = phase_cfg.get('protocols')
    if not protocols_cfg:
        protocols_cfg = [{'name': '', 'drop_features': []}]
    log(f"  Protocols: {[p.get('name','<unnamed>') for p in protocols_cfg]}")

    rows = []
    rankings_per_seed: Dict[str, Dict[int, pd.DataFrame]] = {}

    for protocol in protocols_cfg:
        proto_name = str(protocol.get('name', '') or '')
        drop_cols = list(protocol.get('drop_features') or [])
        X_proto = X.drop(columns=[c for c in drop_cols if c in X.columns], errors='ignore')
        proto_tag = proto_name if proto_name else 'default'
        log(f"\n  [Protocol: {proto_tag}] n_features={X_proto.shape[1]} "
            f"(dropped={[c for c in drop_cols if c in X.columns]})")
        rankings_per_seed[proto_tag] = {}

        # Re-derive k_values per protocol because n_features changes
        proto_k_values = sorted(set(
            X_proto.shape[1] if k is None else min(int(k), X_proto.shape[1]) for k in k_raw
        ))

        for seed in seeds:
            log(f"  Seed {seed}")
            tr_idx, ho_idx = train_test_split(
                np.arange(len(X_proto)),
                test_size=holdout_fraction,
                stratify=y['event'].values,
                random_state=int(seed),
            )
            X_tr_raw, X_ho_raw = X_proto.iloc[tr_idx].copy(), X_proto.iloc[ho_idx].copy()
            y_tr, y_ho = y.iloc[tr_idx].copy(), y.iloc[ho_idx].copy()

            # Impute with the same KNN strategy as phase_2_holdout (fit on train only)
            prep = TabularPreprocessor()
            X_tr_imp, _, _ = prep.fit_transform(X_tr_raw, get_imputation(imp_name))
            X_ho_imp, _, conf_ho = prep.transform(X_ho_raw)

            # 1. Spearman correlation (abs value) of each feature vs survival_days
            spearman_corrs = {}
            for col in X_tr_imp.columns:
                try:
                    corr, _ = spearmanr(X_tr_imp[col].values, y_tr['survival_days'].values)
                    spearman_corrs[col] = abs(corr) if not np.isnan(corr) else 0.0
                except Exception:
                    spearman_corrs[col] = 0.0

            # 2. RF feature importance via binary-at-median classifier
            survival_arr = y_tr['survival_days'].values
            event_arr = y_tr['event'].values
            y_bin = (survival_arr < median_surv).astype(int)
            keep_for_rf = ~((event_arr == 0) & (survival_arr < median_surv))
            rf = RandomForestClassifier(
                n_estimators=rf_n_estimators,
                random_state=int(seed),
                n_jobs=-1,
            )
            rf.fit(X_tr_imp.values[keep_for_rf], y_bin[keep_for_rf])
            rf_imp = dict(zip(X_tr_imp.columns, rf.feature_importances_))

            # 3. Combined rank (lower = more important)
            spearman_rank = pd.Series(spearman_corrs).rank(ascending=False, method='average')
            rf_rank = pd.Series(rf_imp).rank(ascending=False, method='average')
            combined_rank = ((spearman_rank + rf_rank) / 2).sort_values()
            rankings_per_seed[proto_tag][int(seed)] = pd.DataFrame({
                'spearman_corr_abs': pd.Series(spearman_corrs),
                'rf_importance': pd.Series(rf_imp),
                'spearman_rank': spearman_rank,
                'rf_rank': rf_rank,
                'combined_rank': combined_rank,
            }).reindex(combined_rank.index)

            # 4. Train Cox over top-K features for each K
            for K in proto_k_values:
                top_features = combined_rank.head(K).index.tolist()
                X_tr_K = X_tr_imp[top_features]
                X_ho_K = X_ho_imp[top_features]
                try:
                    ci, ece, bs, _ = _eval_cox_baseline(
                        X_tr_K, X_ho_K, y_tr, y_ho,
                        conf_va=conf_ho,
                        output_dim=output_dim,
                        median_surv=median_surv,
                    )
                except Exception as e:
                    log(f"    K={K} seed={seed} cox FAILED: {e}", level="warn")
                    ci, ece, bs = float('nan'), float('nan'), float('nan')

                rows.append({
                    'protocol': proto_tag,
                    'seed': int(seed),
                    'K': int(K),
                    'cindex_holdout': float(ci) if ci == ci else float('nan'),
                    'ece': float(ece) if ece == ece else float('nan'),
                    'brier_score': float(bs) if bs == bs else float('nan'),
                    'n_train': int(len(tr_idx)),
                    'n_holdout': int(len(ho_idx)),
                    'n_features': int(X_proto.shape[1]),
                    'top_features': ','.join(top_features),
                })

    df_results = pd.DataFrame(rows)
    df_results.to_csv(run_dir / "phase2_mahootiha.csv", index=False)

    group_cols = ['protocol', 'K'] if 'protocol' in df_results.columns else ['K']
    summary = (
        df_results
        .groupby(group_cols)['cindex_holdout']
        .agg(cindex_mean='mean', cindex_std='std', cindex_median='median', n_seeds='count')
        .round(4)
    )
    summary.to_csv(run_dir / "phase2_mahootiha_summary.csv")
    log("\n  MAHOOTIHA top-K HOLDOUT SUMMARY:")
    log(summary.to_string())

    # Aggregated feature ranking across seeds (per protocol)
    for proto_tag, seed_rankings in rankings_per_seed.items():
        if not seed_rankings:
            continue
        rank_matrix = pd.DataFrame({
            f'rank_seed{s}': r['combined_rank'] for s, r in seed_rankings.items()
        })
        rank_matrix['mean_combined_rank'] = rank_matrix.mean(axis=1)
        rank_matrix['std_combined_rank'] = rank_matrix.std(axis=1)
        rank_matrix = rank_matrix.sort_values('mean_combined_rank')
        suffix = f"_{proto_tag}" if proto_tag != 'default' else ''
        rank_matrix.to_csv(run_dir / f"phase2_mahootiha_feature_ranking{suffix}.csv")
        log(f"\n  TOP 10 FEATURES BY MEAN COMBINED RANK (protocol={proto_tag}):")
        log(rank_matrix[['mean_combined_rank', 'std_combined_rank']].head(10).to_string())

    return df_results


# ============================================================
# PHASE 2 LATE FUSION HOLDOUT — diagnostic: skip the VAE
# ============================================================

def phase_2_late_fusion_holdout(
    df_features: pd.DataFrame,
    df_targets: pd.DataFrame,
    config: dict,
    run_dir: Path,
    best_imputation: str,
) -> Optional[pd.DataFrame]:
    """
    Diagnostic experiment: skip the VAE entirely and train a single linear
    Cox head on raw concatenated embeddings (late fusion).

    For each seed:
      1. Stratified train/holdout 80/20 split.
      2. Train linear_compact encoder on train pool ONLY → tab_emb (768).
      3. Load text_emb (768) from the precomputed cache, aligned to cohort.
      4. Train three Cox heads (PrognosisProc_LinearCox) and report held-out
         C-index for each:
            - tab_only:  fused_dim = 768, input = tab_emb
            - text_only: fused_dim = 768, input = text_emb
            - concat:    fused_dim = 1536, input = [tab_emb ‖ text_emb]

    Diagnostic interpretation:
      - If concat > tab_only:        VAE was diluting good text signal.
        Action: redesign FUSION-PROC (gating / attention / confidence weights).
      - If concat ≈ tab_only:        Text adds little under off-the-shelf BERT.
        Action: V2 fine-tune the text encoder.
      - If concat < tab_only:        Late fusion has its own overfitting,
        but the VAE result was likely upper-bounded by the same issue.

    YAML structure:
      phase_2_late_fusion_holdout:
        enabled: true
        holdout_fraction: 0.20
        seeds: [42, 123, 456, 789, 1024]
        text_embeddings_cache: /path/to/text_embeddings_*.npz
        encoder_params:
          hidden_dim: 128
          epochs: 200
          lr: 0.001
          patience: 20
        cox_head:
          epochs: 200
          patience: 20
          lr: 1e-3
          weight_decay: 1e-3
    """
    from sklearn.model_selection import train_test_split
    from components.adapters.ingestion.tabular.models.linear_compact import VariantC_LinearEncoder
    from components.processors.prognosis.models.linear_cox import PrognosisProc_LinearCox

    phase_cfg = config.get('phase_2_late_fusion_holdout', {})
    if not phase_cfg.get('enabled', False):
        log("[PHASE 2 LATE FUSION HOLDOUT] DISABLED")
        return None

    validate_clinical_moment(config, phase_cfg, ['text'])

    log("\n[PHASE 2 LATE FUSION HOLDOUT] Skip-VAE diagnostic: tab-only / text-only / concat")

    valid = df_targets['survival_days'].notna() & (df_targets['survival_days'] > 0)
    X = df_features.loc[valid].copy()
    y = df_targets.loc[valid].copy()

    seeds = phase_cfg.get('seeds', config['random']['seeds'])
    holdout_fraction = phase_cfg.get('holdout_fraction', 0.20)
    imp_name = phase_cfg.get('imputation', 'knn_5')
    if imp_name == 'auto':
        imp_name = best_imputation

    encoder_params = phase_cfg.get('encoder_params', {})
    cox_cfg = phase_cfg.get('cox_head', {})
    cox_epochs = cox_cfg.get('epochs', 200)
    cox_patience = cox_cfg.get('patience', 20)
    cox_lr = cox_cfg.get('lr', 1e-3)
    cox_wd = cox_cfg.get('weight_decay', 1e-3)

    text_cache_path = phase_cfg.get('text_embeddings_cache')
    if not text_cache_path:
        log("  No text_embeddings_cache configured — text-only and concat will be skipped.")
    text_emb_cache = None
    text_conf_cache = None
    text_cache_ids = None
    if text_cache_path:
        p = Path(text_cache_path)
        if not p.is_absolute():
            p = Path.cwd() / p
        if not p.exists():
            log(f"  text_embeddings_cache not found: {p}", level="warn")
        else:
            data = np.load(p, allow_pickle=True)
            text_emb_cache = data['embeddings']
            text_conf_cache = data['confidence']
            text_cache_ids = [str(c) for c in data['case_ids']]
            log(f"  Text cache loaded: {p.name} (shape {text_emb_cache.shape})")

    log(f"  Cases: {len(X)}, seeds: {list(seeds)}, holdout_fraction={holdout_fraction}")
    log(f"  Imputation: {imp_name}")
    log(f"  Cox head: epochs={cox_epochs}, patience={cox_patience}, lr={cox_lr}, wd={cox_wd}")

    rows = []
    case_id_to_text_idx = (
        {cid: j for j, cid in enumerate(text_cache_ids)} if text_cache_ids else None
    )

    for seed in seeds:
        log(f"  Seed {seed}")
        tr_idx, ho_idx = train_test_split(
            np.arange(len(X)),
            test_size=holdout_fraction,
            stratify=y['event'].values,
            random_state=int(seed),
        )
        X_tr_raw, X_ho_raw = X.iloc[tr_idx].copy(), X.iloc[ho_idx].copy()
        y_tr, y_ho = y.iloc[tr_idx].copy(), y.iloc[ho_idx].copy()
        cohort_case_ids = list(X.index)

        # ---- 1. Train linear_compact encoder on train pool only (Cox loss) ----
        prep = TabularPreprocessor()
        X_tr_imp, mask_tr, _ = prep.fit_transform(X_tr_raw, get_imputation(imp_name))
        X_ho_imp, mask_ho, _ = prep.transform(X_ho_raw)

        input_dim = X_tr_imp.shape[1]
        encoder = VariantC_LinearEncoder(
            input_dim=input_dim,
            hidden_dim=encoder_params.get('hidden_dim', 128),
            output_dim=768,
        )
        X_tr_t = torch.tensor(X_tr_imp.values, dtype=torch.float32)
        X_ho_t = torch.tensor(X_ho_imp.values, dtype=torch.float32)
        M_tr_t = _build_mask_aligned(mask_tr, X_tr_imp)
        M_ho_t = _build_mask_aligned(mask_ho, X_ho_imp)
        T_tr_t = torch.tensor(y_tr['survival_days'].values, dtype=torch.float32)
        E_tr_t = torch.tensor(y_tr['event'].values, dtype=torch.float32)
        T_ho_t = torch.tensor(y_ho['survival_days'].values, dtype=torch.float32)
        E_ho_t = torch.tensor(y_ho['event'].values, dtype=torch.float32)

        # Use train_variant_c (already GPU-aware) to fit the encoder.
        # We discard its risk_head — we want fresh Cox heads per fusion mode.
        train_variant_c(
            encoder, X_tr_t, M_tr_t, T_tr_t, E_tr_t,
            X_ho_t, M_ho_t, T_ho_t, E_ho_t,
            epochs=encoder_params.get('epochs', 200),
            lr=encoder_params.get('lr', 1e-3),
            patience=encoder_params.get('patience', 20),
            weight_decay=encoder_params.get('weight_decay', 1e-4),
            verbose=False,
        )
        encoder.eval()
        with torch.no_grad():
            tab_emb_tr, _ = encoder(X_tr_t, M_tr_t)
            tab_emb_ho, _ = encoder(X_ho_t, M_ho_t)
        tab_emb_tr = tab_emb_tr.cpu()
        tab_emb_ho = tab_emb_ho.cpu()

        # ---- 2. Align text embeddings to current cohort indices ----
        if case_id_to_text_idx is not None:
            tr_ids = [cohort_case_ids[i] for i in tr_idx]
            ho_ids = [cohort_case_ids[i] for i in ho_idx]
            txt_tr_np = np.zeros((len(tr_idx), 768), dtype=np.float32)
            txt_ho_np = np.zeros((len(ho_idx), 768), dtype=np.float32)
            for k, cid in enumerate(tr_ids):
                j = case_id_to_text_idx.get(str(cid))
                if j is not None:
                    txt_tr_np[k] = text_emb_cache[j]
            for k, cid in enumerate(ho_ids):
                j = case_id_to_text_idx.get(str(cid))
                if j is not None:
                    txt_ho_np[k] = text_emb_cache[j]
            text_tr_t = torch.tensor(txt_tr_np, dtype=torch.float32)
            text_ho_t = torch.tensor(txt_ho_np, dtype=torch.float32)
        else:
            text_tr_t = text_ho_t = None

        # ---- 3. Train three Cox heads ----
        def _train_eval(name, X_tr_emb, X_ho_emb):
            fused_dim = X_tr_emb.shape[1]
            head = PrognosisProc_LinearCox(fused_dim=fused_dim, lr=cox_lr, weight_decay=cox_wd)
            res = head.fit(
                X_tr_emb, T_tr_t, E_tr_t,
                X_ho_emb, T_ho_t, E_ho_t,
                epochs=cox_epochs, patience=cox_patience, verbose=False,
            )
            return float(res['best_val_cindex'])

        ci_tab = _train_eval('tab_only', tab_emb_tr, tab_emb_ho)
        log(f"    tab_only:  cindex={ci_tab:.4f}")
        rows.append({'seed': int(seed), 'fusion': 'tab_only',
                     'cindex_holdout': ci_tab, 'fused_dim': 768,
                     'n_train': len(tr_idx), 'n_holdout': len(ho_idx)})

        if text_tr_t is not None:
            ci_txt = _train_eval('text_only', text_tr_t, text_ho_t)
            log(f"    text_only: cindex={ci_txt:.4f}")
            rows.append({'seed': int(seed), 'fusion': 'text_only',
                         'cindex_holdout': ci_txt, 'fused_dim': 768,
                         'n_train': len(tr_idx), 'n_holdout': len(ho_idx)})

            concat_tr = torch.cat([tab_emb_tr, text_tr_t], dim=1)
            concat_ho = torch.cat([tab_emb_ho, text_ho_t], dim=1)
            ci_cat = _train_eval('concat', concat_tr, concat_ho)
            log(f"    concat:    cindex={ci_cat:.4f}")
            rows.append({'seed': int(seed), 'fusion': 'concat',
                         'cindex_holdout': ci_cat, 'fused_dim': 1536,
                         'n_train': len(tr_idx), 'n_holdout': len(ho_idx)})

    df_results = pd.DataFrame(rows)
    df_results.to_csv(run_dir / "phase2_late_fusion.csv", index=False)

    summary = (
        df_results
        .groupby('fusion')['cindex_holdout']
        .agg(cindex_mean='mean', cindex_std='std', cindex_median='median', n_seeds='count')
        .round(4)
    )
    summary.to_csv(run_dir / "phase2_late_fusion_summary.csv")
    log("\n  LATE FUSION HOLDOUT SUMMARY:")
    log(summary.to_string())
    return df_results


def phase_2_text_only_nested_cv(
    df_targets: pd.DataFrame,
    config: dict,
    run_dir: Path,
) -> Optional[pd.DataFrame]:
    """Evaluate pathology-report embeddings with honest nested CV.

    The outer fold is used once for the final C-index. Early stopping happens
    on an inner validation split. Only cases with a real text embedding and
    valid overall survival enter this modality-specific cohort.
    """
    from sklearn.model_selection import train_test_split
    from components.processors.prognosis.models.linear_cox import (
        PrognosisProc_LinearCox,
    )

    phase_cfg = config.get('phase_2_text_only_nested_cv', {})
    if not phase_cfg.get('enabled', False):
        log("[PHASE 2 TEXT-ONLY NESTED CV] DISABLED")
        return None

    clinical_moment = validate_clinical_moment(config, phase_cfg, ['text'])

    cache_path = Path(phase_cfg['text_embeddings_cache'])
    if not cache_path.is_absolute():
        cache_path = Path.cwd() / cache_path
    if not cache_path.exists():
        raise FileNotFoundError(f"Text embedding cache not found: {cache_path}")

    cache = np.load(cache_path, allow_pickle=True)
    embeddings = np.asarray(cache['embeddings'], dtype=np.float32)
    confidence = np.asarray(cache['confidence'], dtype=np.float32)
    cache_ids = [str(cid) for cid in cache['case_ids']]
    if len(cache_ids) != len(embeddings) or len(confidence) != len(embeddings):
        raise ValueError("Text cache arrays have inconsistent row counts")

    min_confidence = float(phase_cfg.get('min_confidence', 0.0))
    id_to_row = {cid: row for row, cid in enumerate(cache_ids)}
    cohort_ids = []
    cohort_rows = []
    survival = []
    events = []
    for cid in df_targets.index.astype(str):
        row = id_to_row.get(cid)
        if row is None or not confidence[row] > min_confidence:
            continue
        vector = embeddings[row]
        if not np.isfinite(vector).all() or float(np.linalg.norm(vector)) == 0.0:
            continue
        target_row = df_targets.loc[cid]
        time_value = pd.to_numeric(
            pd.Series([target_row.get('survival_days')]), errors='coerce'
        ).iloc[0]
        event_value = pd.to_numeric(
            pd.Series([target_row.get('event')]), errors='coerce'
        ).iloc[0]
        if pd.isna(time_value) or float(time_value) <= 0:
            continue
        if pd.isna(event_value) or int(event_value) not in {0, 1}:
            continue
        cohort_ids.append(cid)
        cohort_rows.append(row)
        survival.append(float(time_value))
        events.append(int(event_value))

    if len(cohort_ids) < 50:
        raise ValueError(
            f"Too few text cases with valid survival ({len(cohort_ids)})"
        )

    X_all = torch.tensor(embeddings[cohort_rows], dtype=torch.float32)
    survival = np.asarray(survival, dtype=np.float32)
    events = np.asarray(events, dtype=np.int64)
    cohort_ids = np.asarray(cohort_ids, dtype=object)

    seeds = phase_cfg.get('seeds', config['random']['seeds'])
    n_folds = int(phase_cfg.get('n_folds', config['random'].get('n_folds', 5)))
    validation_fraction = float(phase_cfg.get('validation_fraction', 0.20))
    epochs = int(phase_cfg.get('epochs', 200))
    patience = int(phase_cfg.get('patience', 20))
    lr = float(phase_cfg.get('lr', 1e-3))
    weight_decay = float(phase_cfg.get('weight_decay', 1e-3))

    log("\n[PHASE 2 TEXT-ONLY NESTED CV] Post-surgery pathology reports")
    log(
        f"  Modality-specific cohort: n={len(cohort_ids)}, "
        f"events={int(events.sum())}, cache={cache_path.name}"
    )
    log(
        f"  Outer folds={n_folds}, seeds={list(seeds)}, "
        f"inner validation={validation_fraction:.0%}"
    )

    fold_rows = []
    prediction_rows = []
    seed_rows = []
    for seed in seeds:
        splitter = StratifiedKFold(
            n_splits=n_folds, shuffle=True, random_state=int(seed)
        )
        oof_risk = np.full(len(cohort_ids), np.nan, dtype=np.float64)

        for fold, (outer_train_idx, test_idx) in enumerate(
            splitter.split(np.zeros(len(cohort_ids)), events)
        ):
            inner_train_local, val_local = train_test_split(
                np.arange(len(outer_train_idx)),
                test_size=validation_fraction,
                stratify=events[outer_train_idx],
                random_state=int(seed) * 100 + fold,
            )
            train_idx = outer_train_idx[inner_train_local]
            val_idx = outer_train_idx[val_local]

            torch.manual_seed(int(seed) * 1000 + fold)
            head = PrognosisProc_LinearCox(
                fused_dim=X_all.shape[1], lr=lr, weight_decay=weight_decay,
            )
            fit_result = head.fit(
                X_all[train_idx],
                torch.tensor(survival[train_idx]),
                torch.tensor(events[train_idx], dtype=torch.float32),
                X_all[val_idx],
                torch.tensor(survival[val_idx]),
                torch.tensor(events[val_idx], dtype=torch.float32),
                epochs=epochs,
                patience=patience,
                verbose=False,
            )
            test_risk = head.predict_risk(X_all[test_idx])
            oof_risk[test_idx] = test_risk
            test_ci = concordance_index(
                survival[test_idx], -test_risk, events[test_idx]
            )
            fold_rows.append({
                'clinical_moment': clinical_moment,
                'modality': 'text',
                'seed': int(seed),
                'fold': int(fold),
                'cindex_test': float(test_ci),
                'cindex_inner_validation_selected': float(
                    fit_result['best_val_cindex']
                ),
                'n_train': int(len(train_idx)),
                'n_validation': int(len(val_idx)),
                'n_test': int(len(test_idx)),
                'events_train': int(events[train_idx].sum()),
                'events_validation': int(events[val_idx].sum()),
                'events_test': int(events[test_idx].sum()),
            })
            prediction_rows.extend({
                'clinical_moment': clinical_moment,
                'modality': 'text',
                'seed': int(seed),
                'fold': int(fold),
                'case_id': str(cohort_ids[idx]),
                'survival_days': float(survival[idx]),
                'event': int(events[idx]),
                'risk_oof': float(oof_risk[idx]),
            } for idx in test_idx)

        if np.isnan(oof_risk).any():
            raise RuntimeError(f"Incomplete OOF predictions for seed {seed}")
        pooled_ci = concordance_index(survival, -oof_risk, events)
        seed_fold_cis = [
            row['cindex_test'] for row in fold_rows if row['seed'] == int(seed)
        ]
        seed_rows.append({
            'clinical_moment': clinical_moment,
            'modality': 'text',
            'seed': int(seed),
            'n_cases': int(len(cohort_ids)),
            'n_events': int(events.sum()),
            'cindex_outer_fold_mean': float(np.mean(seed_fold_cis)),
            'cindex_outer_fold_std': float(np.std(seed_fold_cis, ddof=1)),
            # Diagnostic only: scores from independently trained folds are not
            # guaranteed to share a common scale, so this must not be the
            # primary cross-validation estimate.
            'cindex_pooled_oof_uncalibrated': float(pooled_ci),
        })
        log(
            f"  Seed {seed}: outer-fold mean C-index="
            f"{np.mean(seed_fold_cis):.4f} "
            f"(pooled uncalibrated diagnostic={pooled_ci:.4f})"
        )

    folds_df = pd.DataFrame(fold_rows)
    predictions_df = pd.DataFrame(prediction_rows)
    seeds_df = pd.DataFrame(seed_rows)
    folds_df.to_csv(run_dir / 'phase2_text_only_nested_cv_folds.csv', index=False)
    predictions_df.to_csv(
        run_dir / 'phase2_text_only_nested_cv_oof_predictions.csv', index=False
    )
    seeds_df.to_csv(run_dir / 'phase2_text_only_nested_cv_seeds.csv', index=False)

    summary_df = pd.DataFrame([{
        'clinical_moment': clinical_moment,
        'modality': 'text',
        'n_cases': int(len(cohort_ids)),
        'n_events': int(events.sum()),
        'n_seeds': int(len(seeds_df)),
        'n_outer_folds': n_folds,
        'cindex_outer_fold_mean': float(
            seeds_df['cindex_outer_fold_mean'].mean()
        ),
        'cindex_outer_fold_std_across_seeds': float(
            seeds_df['cindex_outer_fold_mean'].std(ddof=1)
        ),
        'cindex_outer_fold_median': float(
            seeds_df['cindex_outer_fold_mean'].median()
        ),
        'cindex_outer_fold_min': float(
            seeds_df['cindex_outer_fold_mean'].min()
        ),
        'cindex_outer_fold_max': float(
            seeds_df['cindex_outer_fold_mean'].max()
        ),
        'cindex_pooled_oof_uncalibrated_mean': float(
            seeds_df['cindex_pooled_oof_uncalibrated'].mean()
        ),
    }])
    summary_df.to_csv(
        run_dir / 'phase2_text_only_nested_cv_summary.csv', index=False
    )
    log("\n  TEXT-ONLY HONEST OUTER-FOLD SUMMARY:")
    log(summary_df.to_string(index=False))
    return seeds_df


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
    if phase_cfg.get('text_embeddings_npz'):
        log(f"  Precomputed text embeddings: {phase_cfg['text_embeddings_npz']}")
    if phase_cfg.get('vision_embeddings_csv'):
        log(f"  Precomputed vision embeddings: {phase_cfg['vision_embeddings_csv']}")
    
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
    N = len(X_raw)
 
    # --- FIX (Apr 2026, BUG 5): Hold-out split BEFORE any data-dependent
    # fitting, to provide a clean evaluation set for Phase 7 and remove
    # downstream imputation leakage between train pool and test set.
    # The split is stratified by event and seed-controlled.
    holdout_frac = phase_cfg.get('holdout_fraction', 0.20)
    pool_idx, holdout_idx = train_test_split(
        np.arange(N),
        test_size=holdout_frac,
        random_state=seed,
        stratify=y['event'].values,
    )
    log(f"  Hold-out split: train pool = {len(pool_idx)} | held-out = {len(holdout_idx)}")
    log(f"  Events in pool: {int(y['event'].iloc[pool_idx].sum())} | "
        f"events in held-out: {int(y['event'].iloc[holdout_idx].sum())}")
 
    X_raw_pool = X_raw.iloc[pool_idx]
    X_raw_holdout = X_raw.iloc[holdout_idx]
 
    # --- Preprocess tabular with best imputation ---
    # FIX (Apr 2026, BUG 5): fit imputer ONLY on train pool; transform held-out.
    imp_name = phase_cfg.get('tabular_imputation', 'auto')
    if imp_name == 'auto':
        imp_name = best_imputation
    prep = TabularPreprocessor()
    X_tab_pool, mask_pool, conf_tab_pool = prep.fit_transform(
        X_raw_pool, get_imputation(imp_name))
    X_tab_holdout, mask_holdout, conf_tab_holdout = prep.transform(X_raw_holdout)
 
    # Reassemble in original order so downstream code can index by [0..N-1]
    # but preserves train-pool-only fit semantics.
    X_tab = pd.concat([X_tab_pool, X_tab_holdout]).iloc[
        np.argsort(np.concatenate([pool_idx, holdout_idx]))
    ]
    mask = pd.concat([mask_pool, mask_holdout]).iloc[
        np.argsort(np.concatenate([pool_idx, holdout_idx]))
    ]
 
    log(f"  Cases: {N}  (events: {int(y['event'].sum())})")
    log(f"  Imputation: {imp_name}")
 
    # --- Build trimodal input (tabular real + mock text/vision) ---
    modality_dim = phase_cfg.get('modality_dim', 768)
    modalities = phase_cfg.get('modalities', ['tabular', 'text', 'vision'])
    clinical_moment = validate_clinical_moment(config, phase_cfg, modalities)
    log(f"  Clinical moment: {clinical_moment}")
    n_mod = len(modalities)
 
    # Train a linear_compact encoder on the full cohort. Single stratified
    # split, not CV — this is an embedder, not a predictor benchmark.
    if X_tab.shape[1] == modality_dim:
        tab_emb = X_tab.values.astype(np.float32)
        log(f"  Tabular: already at modality_dim={modality_dim}, no encoder needed")
    else:
        log(f"  Training linear_compact embedder ({X_tab.shape[1]} → "
            f"{modality_dim}) on {N} cases...")
 
        # FIX (Apr 2026, BUG 5): encoder train/val split is taken FROM the
        # train pool only -- the held-out set never participates in encoder
        # training or its loss-driven Cox supervision.
        pool_local = pool_idx
        enc_train_local, enc_val_local = train_test_split(
            np.arange(len(pool_local)),
            test_size=phase_cfg.get('encoder_val_fraction', 0.15),
            random_state=seed,
            stratify=y['event'].iloc[pool_local].values,
        )
        enc_train_idx = pool_local[enc_train_local]
        enc_val_idx   = pool_local[enc_val_local]
 
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
 
    # --- Load optional precomputed text / vision embedding caches ---
    # These are produced offline by tools/build_text_embeddings.py and (future)
    # tools/build_vision_embeddings.py. Each .npz must carry three arrays
    # aligned by case_id: embeddings[M, modality_dim], confidence[M], case_ids[M].
    # If the cache path is absent or the file is missing, the corresponding
    # modality stays as zeros with confidence 0 (mock — backward-compatible).
    def _load_modality_cache(cache_path: Optional[str], mod_label: str):
        if not cache_path:
            return None, None
        p = Path(cache_path)
        if not p.is_absolute():
            p = Path.cwd() / p
        if not p.exists():
            log(f"  [{mod_label}] cache path configured but file not found: {p} — "
                f"falling back to zeros.", level="warn")
            return None, None
        data = np.load(p, allow_pickle=True)
        cache_ids = [str(c) for c in data['case_ids']]
        cache_emb = data['embeddings']
        cache_conf = data['confidence']
        if cache_emb.shape[1] != modality_dim:
            raise ValueError(
                f"[{mod_label}] cache modality_dim {cache_emb.shape[1]} != configured "
                f"modality_dim {modality_dim}: {p}"
            )
        # Reorder to current cohort case_ids; unknown cases get zeros + conf=0.
        idx_map = {cid: j for j, cid in enumerate(cache_ids)}
        aligned_emb = np.zeros((N, modality_dim), dtype=np.float32)
        aligned_conf = np.zeros(N, dtype=np.float32)
        n_found = 0
        for k, cid in enumerate(case_ids):
            j = idx_map.get(str(cid))
            if j is not None:
                aligned_emb[k] = cache_emb[j]
                aligned_conf[k] = cache_conf[j]
                if cache_conf[j] > 0:
                    n_found += 1
        log(f"  [{mod_label}] cache: {n_found}/{N} cases have data "
            f"(mean conf when present = "
            f"{aligned_conf[aligned_conf > 0].mean() if (aligned_conf > 0).any() else 0:.3f})  "
            f"source={p.name}")
        return aligned_emb, aligned_conf

    text_emb_aligned, text_conf_aligned = _load_modality_cache(
        phase_cfg.get('text_embeddings_cache'), 'text'
    )
    vision_emb_aligned, vision_conf_aligned = _load_modality_cache(
        phase_cfg.get('vision_embeddings_cache'), 'vision'
    )

    # --- Assemble flat input tensor for the VAE ---
    X_flat = torch.zeros(N, modality_dim * n_mod, dtype=torch.float32)
    confs_matrix = np.zeros((N, n_mod), dtype=np.float32)
    for i, mod_name in enumerate(modalities):
        slot = slice(i * modality_dim, (i + 1) * modality_dim)
        if mod_name == 'tabular':
            X_flat[:, slot] = torch.tensor(tab_emb)
            confs_matrix[:, i] = 1.0
        elif mod_name == 'text' and text_emb_aligned is not None:
            X_flat[:, slot] = torch.tensor(text_emb_aligned)
            confs_matrix[:, i] = text_conf_aligned
        elif mod_name == 'vision' and vision_emb_aligned is not None:
            X_flat[:, slot] = torch.tensor(vision_emb_aligned)
            confs_matrix[:, i] = vision_conf_aligned
        # Otherwise the modality remains zeros with confidence 0 (mock).
 
    conf = torch.tensor(confs_matrix, dtype=torch.float32)
    T = torch.tensor(y['survival_days'].values, dtype=torch.float32)
    E = torch.tensor(y['event'].values, dtype=torch.float32)
 
    # --- Train/val split for the VAE (separate from the encoder split) ---
    # FIX (Apr 2026, BUG 5): the VAE train/val split is taken from the train
    # pool only -- held-out cases never see the VAE's gradient updates.
    val_frac = phase_cfg.get('val_fraction', 0.15)
    pool_local_v = pool_idx
    idx_tr_local, idx_va_local = train_test_split(
        np.arange(len(pool_local_v)), test_size=val_frac, random_state=seed,
        stratify=E.numpy()[pool_local_v],
    )
    idx_tr = pool_local_v[idx_tr_local]
    idx_va = pool_local_v[idx_va_local]
    log(f"  VAE train/val (within train pool): {len(idx_tr)}/{len(idx_va)}")
    log(f"  Held-out cases excluded from VAE training: {len(holdout_idx)}")
 
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
 
    # --- Late-fusion shortcut: skip the VAE entirely and use the concat as Z ---
    # When phase_6_fusion_proc.late_fusion_mode is true, the "latent" passed
    # downstream is simply the concatenated modality block (X_flat). Phase 7
    # (TurboLatent) and Phase 8 (Cox/Weibull on Z) read d_latent dynamically,
    # so they work transparently with any D. This is the operational path
    # after the 2026-05-28 diagnostic showed the VAE was diluting text signal.
    late_fusion_mode = phase_cfg.get('late_fusion_mode', False)
    if late_fusion_mode:
        log("  late_fusion_mode=ON: skipping VAE training, Z = concat modalities")
        Z = X_flat.cpu().numpy().astype(np.float32)
        conf_full = conf.cpu().numpy().astype(np.float32).mean(axis=1)
        result = {'stage_A_history': [], 'stage_B_history': []}
    else:
        # --- Train VAE (on GPU if available; vae.fit reads device from model params) ---
        vae_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        vae = vae.to(vae_device)
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
        # Return VAE to CPU so the downstream extract_latent_space call (which
        # receives CPU tensors X_flat/conf) does not hit a device mismatch.
        vae = vae.to('cpu')

        # --- Extract frozen Z for full cohort ---
        vae.eval()
        with torch.no_grad():
            Z, conf_full = vae.extract_latent_space(X_flat, conf)
        Z = Z.cpu().numpy()
        conf_full = conf_full.cpu().numpy()
 
    # --- Persist artifact ---
    # FIX (Apr 2026, BUG 5): persist BOTH the train-pool/held-out partition
    # used for clean evaluation AND the inner train/val split used for
    # VAE training. Phase 7 must consume train_pool_idx for CV and
    # holdout_idx for the single clean held-out evaluation.
    artifacts_dir = get_artifacts_dir(run_dir)
    latent_path = artifacts_dir / "phase_6_latent_z.npz"
    np.savez(
        latent_path,
        Z=Z, conf=conf_full,
        T=T.numpy(), E=E.numpy(),
        case_ids=case_ids,
        train_pool_idx=pool_idx,
        holdout_idx=holdout_idx,
        vae_train_idx=idx_tr,
        vae_val_idx=idx_va,
        train_idx=idx_tr, val_idx=idx_va,
    )
    log(f"  Latent Z saved: {latent_path}  (shape: {Z.shape})")
    log(f"  Train pool: {len(pool_idx)} cases | Held-out: {len(holdout_idx)} cases")
 
    # --- Persist checkpoint + history (only when a VAE was actually trained) ---
    if not late_fusion_mode:
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
    if late_fusion_mode:
        summary_row = {
            'model':            'late_fusion_concat',
            'n_parameters':     0,
            'd_latent':         Z.shape[1],
            'n_cases':          N,
            'n_events':         int(y['event'].sum()),
            'stage_a_epochs':   0,
            'stage_b_epochs':   0,
            'elapsed_s':        0.0,
            'artifact_path':    str(latent_path),
        }
    else:
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
    from sklearn.model_selection import StratifiedKFold, train_test_split

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

    # FIX (Apr 2026, BUG 5): load clean train/holdout partition for
    # leakage-free CV and held-out evaluation. Falls back to legacy
    # behaviour (CV over all N, no held-out report) if the artifact
    # was produced by a pre-fix runner.
    if 'train_pool_idx' in data.files and 'holdout_idx' in data.files:
        train_pool_idx = data['train_pool_idx']
        holdout_idx = data['holdout_idx']
        has_clean_partition = True
        log(f"  Clean partition detected: pool={len(train_pool_idx)} | "
            f"held-out={len(holdout_idx)}")
    else:
        train_pool_idx = np.arange(N)
        holdout_idx = np.array([], dtype=int)
        has_clean_partition = False
        log("  WARN: artifact has no train_pool/holdout split. CV will run on "
            "ALL cases and held-out C-index will not be reported. "
            "Re-run Phase 6 to obtain a leakage-free evaluation.")
 
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
        """Data-driven rotation via SVD of the centered Z.

        Uses `full_matrices=True` so the right-singular-vector matrix Vt is
        always [D, D] orthogonal. With `full_matrices=False` SVD returns
        Vt with shape [min(N, D), D]; in the concat-fusion case where
        D > N, that's [N, D] and is NOT a valid rotation matrix (the
        downstream `Z @ R.T` then collapses to [N, N] and the Cox head
        receives the wrong feature count). Caveat: when N < D, only the
        first N singular directions are data-driven; the rest of Vt
        is an arbitrary orthonormal basis of the null space. The SVD
        baseline in this regime is therefore a fair-but-noisy comparator.
        """
        Zc = Z - Z.mean(axis=0, keepdims=True)
        _, _, Vt = np.linalg.svd(Zc, full_matrices=True)
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
        """Return (mean C-index, std) across seeds x folds.
        FIX (Apr 2026, BUG 5): CV is restricted to the train pool. The held-out
        set is evaluated separately by eval_cox_holdout below.
        """
        N_pool = len(train_pool_idx)
        E_pool = E[train_pool_idx]
        seed_means = []
        for s in seeds:
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=s)
            fold_cis = []
            for tr_local, va_local in skf.split(np.zeros(N_pool), E_pool):
                tr_idx = train_pool_idx[tr_local]
                va_idx = train_pool_idx[va_local]
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

    def eval_cox_holdout(X: np.ndarray) -> Tuple[float, float]:
        """Train Cox on the full train pool once per seed, evaluate on the
        fixed held-out set. Variance comes from model-init randomness only.

        Critical when D > N (e.g. concat-1536 with N=355 train pool): the
        in-pool CV severely overfits because each CV fold has only ~284
        training cases for 1536 covariates. The held-out cifra is the
        honest one for Paper 3 rotation/quantization comparisons.

        Protocol note: this uses the FIXED train_pool_idx / holdout_idx from
        the Phase 6 artifact (set by random.seed=42). This is internally
        consistent for Paper 3 (all rotation variants on the same Z and the
        same held-out 45/89 cases). It is intentionally different from the
        per-seed stratified protocol used by phase_2_late_fusion_holdout
        (Hallazgo 6 in Paper 2). Don't try to "fix" by re-splitting per seed
        in this function: the encoder that produced Z saw pool_42 already,
        so any per-seed re-split of Z leaks the encoder's training data
        into the held-out. The legitimate way to obtain a per-seed cifra is
        to re-train the encoder per seed (which is what phase_2_late_fusion_holdout
        does and what would require 5 separate Phase 6 artifacts here).
        """
        if len(holdout_idx) == 0:
            return float('nan'), float('nan')
        cis = []
        for s in seeds:
            torch.manual_seed(int(s))
            np.random.seed(int(s))
            X_tr = torch.tensor(X[train_pool_idx], dtype=torch.float32)
            X_ho = torch.tensor(X[holdout_idx],    dtype=torch.float32)
            T_tr = torch.tensor(T[train_pool_idx], dtype=torch.float32)
            T_ho = torch.tensor(T[holdout_idx],    dtype=torch.float32)
            E_tr = torch.tensor(E[train_pool_idx], dtype=torch.float32)
            E_ho = torch.tensor(E[holdout_idx],    dtype=torch.float32)
            model = get_prognosis_proc(prognosis_name, fused_dim=D)
            res = model.fit(
                X_tr, T_tr, E_tr, X_ho, T_ho, E_ho,
                epochs=phase_cfg.get('epochs', 200),
                patience=phase_cfg.get('patience', 20),
                verbose=False,
            )
            cis.append(float(res['best_val_cindex']))
        return float(np.mean(cis)), float(np.std(cis))
 
    # --- Run all (variant × bits) combinations ---
    rows = []

    def _record(variant: str, bits, X_eval: np.ndarray) -> None:
        ci_cv, std_cv = eval_cox_cv(X_eval)
        ci_ho, std_ho = eval_cox_holdout(X_eval)
        rows.append({
            'variant': variant, 'bits': bits,
            # CV in-pool (original semantics — kept for backward compat)
            'cindex_mean': ci_cv, 'cindex_std': std_cv,
            'cindex_cv_mean': ci_cv, 'cindex_cv_std': std_cv,
            # Held-out (NEW — honest cifra when D > N)
            'cindex_holdout_mean': ci_ho, 'cindex_holdout_std': std_ho,
        })
        # One log line with both numbers; NaN held-out renders cleanly.
        ho_str = (f"  |  held-out {ci_ho:.4f} ± {std_ho:.4f}"
                  if ci_ho == ci_ho else "  |  held-out (none)")
        log(f"  {variant:<10} {str(bits):<14} cv {ci_cv:.4f} ± {std_cv:.4f}{ho_str}")

    if include_baseline:
        log("  Baseline: no rotation, FP32")
        _record('baseline', 'fp32', Z)

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

        _record(variant, 'fp32_rotated', Z_rot)
        for bits in bit_widths:
            Z_q = quantize_uniform(Z_rot, bits=bits)
            _record(variant, int(bits), Z_q)
 
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

    # FIX (Apr 2026, BUG 5): load clean train/holdout partition produced by
    # Phase 6. CV is restricted to the train pool; the held-out set is
    # evaluated separately as a single clean number per model. Falls back to
    # legacy behaviour if the artifact predates the fix.
    if 'train_pool_idx' in data.files and 'holdout_idx' in data.files:
        train_pool_idx = data['train_pool_idx']
        holdout_idx = data['holdout_idx']
        has_clean_partition = True
        log(f"  Clean partition detected: pool={len(train_pool_idx)} | "
            f"held-out={len(holdout_idx)}")
    else:
        train_pool_idx = np.arange(N)
        holdout_idx = np.array([], dtype=int)
        has_clean_partition = False
        log("  WARN: artifact has no train_pool/holdout split. CV will run on "
            "ALL cases and held-out C-index will not be reported. "
            "Re-run Phase 6 to obtain a leakage-free evaluation.")

    seeds = config['random']['seeds']
    n_folds = config['random']['n_folds']
    epochs = phase_cfg.get('epochs', 200)
    patience = phase_cfg.get('patience', 20)
    models_cfg = phase_cfg.get('models', [
        {'name': 'prognosis_baseline_linear_cox'},
        {'name': 'prognosis_weibull_head'},
    ])

    rows = []
    holdout_rows = []
    for mcfg in models_cfg:
        model_name = mcfg['name']
        model_params = mcfg.get('params', {})

        # --- In-pool CV (BUG 5 fix: restricted to train pool only) ---
        N_pool = len(train_pool_idx)
        E_pool_np = E.numpy()[train_pool_idx]
        seed_means = []
        for s in seeds:
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=s)
            fold_cis = []
            for tr_local, va_local in skf.split(np.zeros(N_pool), E_pool_np):
                tr_idx = train_pool_idx[tr_local]
                va_idx = train_pool_idx[va_local]
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
                'evaluation': 'cv_in_pool',
            })
        m_arr = np.array(seed_means)
        log(f"  {model_name:35s}  in-pool CV: {m_arr.mean():.4f} +/- "
            f"{m_arr.std():.4f}  (median {np.median(m_arr):.4f})")

        # --- Single held-out evaluation per seed (one model trained on full pool,
        # evaluated on the untouched held-out set) ---
        if has_clean_partition and len(holdout_idx) > 0:
            ho_cis = []
            for s in seeds:
                torch.manual_seed(int(s))
                np.random.seed(int(s))
                model = get_prognosis_proc(
                    model_name, fused_dim=D, **model_params
                )
                # Train on the ENTIRE train pool (no inner CV), single pass.
                res = model.fit(
                    Z[train_pool_idx], T[train_pool_idx], E[train_pool_idx],
                    Z[holdout_idx],     T[holdout_idx],    E[holdout_idx],
                    epochs=epochs, patience=patience, verbose=False,
                )
                ho_cis.append(float(res['best_val_cindex']))
                holdout_rows.append({
                    'model': model_name, 'seed': int(s),
                    'cindex_holdout': float(res['best_val_cindex']),
                })
            ho_arr = np.array(ho_cis)
            log(f"  {model_name:35s}  held-out : {ho_arr.mean():.4f} +/- "
                f"{ho_arr.std():.4f}  (n_holdout={len(holdout_idx)})")

    results_df = pd.DataFrame(rows)
    results_df.to_csv(run_dir / "phase_8_prognosis_benchmark.csv", index=False)

    if holdout_rows:
        holdout_df = pd.DataFrame(holdout_rows)
        holdout_df.to_csv(run_dir / "phase_8_prognosis_holdout.csv", index=False)

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

    # ---- Step 0.5: optional cohort filter (DFS validity + GDC modality availability) ----
    cohort_cfg = config.get('cohort_filter', {})
    if cohort_cfg:
        df_features, df_targets, cohort_audit = apply_cohort_filter(
            df_features, df_targets, cohort_cfg, run_dir,
        )
        summary['cohort_filter'] = {
            'n_initial': cohort_audit['n_initial'],
            'n_final': cohort_audit['n_final'],
            'filters_applied': cohort_audit['filters_applied'],
            'n_after_each_filter': cohort_audit['n_after_each_filter'],
        }

    valid_survival = (
        df_targets['survival_days'].notna()
        & (df_targets['survival_days'] > 0)
    )
    summary['n_cases'] = int(len(df_features))
    summary['n_cases_extracted'] = int(len(df_features))
    summary['n_cases_survival'] = int(valid_survival.sum())
    summary['n_features'] = int(df_features.shape[1])
    summary['n_events'] = int(df_targets['event'].sum())
    summary['n_events_survival'] = int(
        df_targets.loc[valid_survival, 'event'].sum()
    )
    
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

        ph2_ho = phase_2_holdout(df_features, df_targets, config, run_dir, best_imp)
        if ph2_ho is not None:
            group_cols = ['protocol', 'variant'] if 'protocol' in ph2_ho.columns else ['variant']
            ph2_ho_summary = (
                ph2_ho.groupby(group_cols)['cindex_holdout']
                .agg(['mean', 'std', 'median'])
                .round(4)
                .reset_index()
            )
            summary['phases']['phase_2_holdout'] = ph2_ho_summary.to_dict(orient='records')

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

        ph2_ext_ho = phase_2_external_holdout(df_features, df_targets, config, run_dir, best_imp)
        if ph2_ext_ho is not None:
            ph2_ext_ho_summary = (
                ph2_ext_ho.groupby('baseline')['cindex_holdout']
                .agg(['mean', 'std', 'median'])
                .round(4)
            )
            summary['phases']['phase_2_external_holdout'] = ph2_ext_ho_summary.to_dict()

        ph2_mah = phase_2_mahootiha(df_features, df_targets, config, run_dir, best_imp)
        if ph2_mah is not None:
            ph2_mah_summary = (
                ph2_mah.groupby('K')['cindex_holdout']
                .agg(['mean', 'std', 'median', 'count'])
                .round(4)
            )
            summary['phases']['phase_2_mahootiha'] = ph2_mah_summary.to_dict()

        ph2_lf = phase_2_late_fusion_holdout(df_features, df_targets, config, run_dir, best_imp)
        if ph2_lf is not None:
            ph2_lf_summary = (
                ph2_lf.groupby('fusion')['cindex_holdout']
                .agg(['mean', 'std', 'median'])
                .round(4)
            )
            summary['phases']['phase_2_late_fusion_holdout'] = ph2_lf_summary.to_dict()

        ph2_text = phase_2_text_only_nested_cv(df_targets, config, run_dir)
        if ph2_text is not None:
            summary['phases']['phase_2_text_only_nested_cv'] = {
                'clinical_moment': 'post_surgery',
                'modality': 'text',
                'n_cases': int(ph2_text['n_cases'].iloc[0]),
                'n_events': int(ph2_text['n_events'].iloc[0]),
                'cindex_outer_fold_mean': float(
                    ph2_text['cindex_outer_fold_mean'].mean()
                ),
                'cindex_outer_fold_std_across_seeds': float(
                    ph2_text['cindex_outer_fold_mean'].std(ddof=1)
                ),
                'per_seed': ph2_text[
                    [
                        'seed',
                        'cindex_outer_fold_mean',
                        'cindex_outer_fold_std',
                        'cindex_pooled_oof_uncalibrated',
                    ]
                ].to_dict(orient='records'),
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
