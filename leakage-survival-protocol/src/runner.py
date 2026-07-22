"""
Anti-leakage survival-protocol experiment runner.
==================================================

Trimmed for the paper release. Reproduces all numerical claims of the
manuscript on TCGA-KIRC:

  - phase_1_imputation        (optional benchmark of imputation strategies)
  - phase_2_holdout           (limpio vs permisivo, the headline result)
  - phase_2_mahootiha         (Spearman + RF + top-K replication)
  - apply_cohort_filter       (dense n=224 subcohort via modality manifest)

All multimodal / fusion / text / vision / TurboLatent / Weibull code has
been excluded so the release stays focused and auditable. The
configuration file in `configs/experiment_config.yaml` is the single
source of truth for what runs.

Usage:
    python -m src.runner configs/experiment_config.yaml
"""
from __future__ import annotations

import hashlib
import json
import platform
import shutil
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import yaml
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# Resolve package root so absolute imports work whether invoked as a
# module (`python -m src.runner`) or as a script (`python src/runner.py`).
_PKG_ROOT = Path(__file__).resolve().parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.append(str(_PKG_ROOT))

from src.preprocessing.extractor import TCGAExtractor
from src.preprocessing.imputation import TabularPreprocessor
from src.model_utils import (
    verify_ingestion_contract,
    train_variant_c,
)
from src.registry import get_imputation, get_variant, list_components


# ============================================================
# PROVENANCE & RUN DIRECTORY MANAGEMENT
# ============================================================

def compute_config_hash(config_dict: dict) -> str:
    canonical = yaml.dump(config_dict, sort_keys=True, default_flow_style=False)
    return hashlib.sha256(canonical.encode()).hexdigest()[:8]


def create_run_directory(config: dict, config_path: Optional[Union[str, Path]] = None) -> Path:
    base_dir = Path(config['output']['base_dir'])
    base_dir.mkdir(parents=True, exist_ok=True)

    config_hash = compute_config_hash(config)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_id = f"{timestamp}_{config_hash}"
    run_dir = base_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    if config_path and Path(config_path).exists():
        shutil.copy2(config_path, run_dir / "experiment_config.yaml")
    else:
        with open(run_dir / "experiment_config.yaml", 'w') as f:
            yaml.dump(config, f, sort_keys=False)

    feature_config_path = Path(config['data']['feature_config'])
    if feature_config_path.exists():
        shutil.copy2(feature_config_path, run_dir / "feature_config.yaml")

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
    if verbosity == "silent":
        return
    if level == "debug" and verbosity != "verbose":
        return
    print(msg)


# ============================================================
# CALIBRATION METRICS (point-estimates used inside per-fold eval)
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
# PHASE 1 — IMPUTATION BENCHMARK (optional)
# ============================================================

def phase_1_imputation(df_features, df_targets, config, run_dir) -> Tuple[Optional[pd.DataFrame], str]:
    phase_cfg = config.get('phase_1_imputation', {})
    if not phase_cfg.get('enabled', False):
        log("[PHASE 1] DISABLED — skipping imputation benchmark")
        fallback = config.get('phase_2_holdout', {}).get('imputation_for_variants', 'knn_5')
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

                original_dists = {
                    c: X_tr[c].dropna().values
                    for c in X_tr.select_dtypes(include=[np.number]).columns
                    if X_tr[c].dropna().shape[0] > 10
                }
                strategy = get_imputation(strategy_name)
                prep = TabularPreprocessor()
                X_tr_p, _, _ = prep.fit_transform(X_tr, strategy)
                X_va_p, _, _ = prep.transform(X_va)

                from scipy import stats
                ks_scores = []
                for c, orig in original_dists.items():
                    if c in X_tr_p.columns:
                        s, _ = stats.ks_2samp(orig, X_tr_p[c].values)
                        ks_scores.append(s)
                fold_ks.append(np.mean(ks_scores) if ks_scores else np.nan)

                try:
                    cox_df = X_tr_p.copy()
                    cox_df['T'] = y_tr['survival_days'].values
                    cox_df['E'] = y_tr['event'].values
                    valid_cols = [
                        c for c in cox_df.columns
                        if c not in ['T', 'E'] and cox_df[c].std() > 1e-8
                    ]
                    cox_df = cox_df[valid_cols + ['T', 'E']].replace([np.inf, -np.inf], np.nan).dropna()
                    cph = CoxPHFitter(penalizer=cox_penalizer)
                    cph.fit(cox_df, duration_col='T', event_col='E')
                    X_va_cox = X_va_p[valid_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
                    risk = cph.predict_partial_hazard(X_va_cox).values.ravel()
                    fold_ci.append(concordance_index(y_va['survival_days'].values, -risk, y_va['event'].values))
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
            'n_seeds': len(seeds), 'n_folds': n_folds,
        })
        log(f"     C-index {rows[-1]['cindex_mean']:.4f} ± {rows[-1]['cindex_std']:.4f} | "
            f"K-S {rows[-1]['ks_mean']:.4f}")

    df_results = pd.DataFrame(rows)
    df_results.to_csv(run_dir / "phase1_imputation.csv", index=False)
    best = df_results.loc[df_results['cindex_mean'].idxmax(), 'strategy']
    log(f"  WINNER: {best}")
    return df_results, best


# ============================================================
# VARIANT EVALUATION (Cox / linear_compact / FT-Transformer)
# ============================================================

def _build_mask_aligned(mask_df: pd.DataFrame, feature_df: pd.DataFrame) -> torch.Tensor:
    aligned = torch.ones(len(feature_df), len(feature_df.columns), dtype=torch.float32)
    for i, col in enumerate(feature_df.columns):
        mask_col = f"mask__{col}"
        if mask_col in mask_df.columns:
            aligned[:, i] = torch.tensor(mask_df[mask_col].values, dtype=torch.float32)
    return aligned


def _maybe_dump_curves(result: dict, curves_ctx: Optional[dict]) -> None:
    if not curves_ctx:
        return
    required = {'run_dir', 'seed', 'fold', 'variant_name'}
    if not required.issubset(curves_ctx.keys()):
        return
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


def _save_predictions_npz(
    artifacts_dir: Optional[Path], artifacts_key: str, *,
    case_ids_train, risk_train, times_train, events_train,
    case_ids_holdout, risk_holdout, times_holdout, events_holdout,
    surv_func_holdout=None, surv_time_grid=None,
    extra: Optional[dict] = None,
) -> None:
    if artifacts_dir is None or not artifacts_key:
        return
    out_dir = Path(artifacts_dir) / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        'case_ids_train':   np.asarray(case_ids_train).astype('U64'),
        'risk_train':       np.asarray(risk_train, dtype=np.float64),
        'times_train':      np.asarray(times_train, dtype=np.float64),
        'events_train':     np.asarray(events_train, dtype=np.int64),
        'case_ids_holdout': np.asarray(case_ids_holdout).astype('U64'),
        'risk_holdout':     np.asarray(risk_holdout, dtype=np.float64),
        'times_holdout':    np.asarray(times_holdout, dtype=np.float64),
        'events_holdout':   np.asarray(events_holdout, dtype=np.int64),
    }
    if surv_func_holdout is not None:
        payload['surv_func_holdout'] = np.asarray(surv_func_holdout, dtype=np.float64)
        payload['surv_time_grid']    = np.asarray(surv_time_grid, dtype=np.float64)
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
            'kind': 'cox_baseline', 'cph': cph,
            'scaler': scaler, 'feature_cols': list(feature_cols),
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
            'encoder_state_dict':   {k: v.cpu() for k, v in encoder.state_dict().items()},
            'risk_head_state_dict': {k: v.cpu() for k, v in risk_head.state_dict().items()},
            'feature_cols': list(feature_cols),
            'variant_kwargs': variant_kwargs,
        }, f)


def _evaluate_variant(
    variant_name, input_dim, output_dim, variant_params,
    X_tr, X_va, y_tr, y_va,
    mask_tr, mask_va, conf_tr, conf_va,
    median_surv,
    curves_ctx: Optional[dict] = None,
    artifacts_dir: Optional[Path] = None,
    artifacts_key: str = '',
) -> Tuple[float, float, float, bool]:
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


def _eval_cox_baseline(X_tr, X_va, y_tr, y_va, conf_va, output_dim, median_surv,
                       artifacts_dir: Optional[Path] = None, artifacts_key: str = ''):
    miss_frac = X_tr.isna().mean(axis=0)
    keep_cols = miss_frac[miss_frac < 0.90].index.tolist()
    X_tr_f = X_tr[keep_cols].copy()
    X_va_f = X_va[keep_cols].copy()

    X_tr_f = X_tr_f.replace([np.inf, -np.inf], np.nan)
    X_va_f = X_va_f.replace([np.inf, -np.inf], np.nan)
    valid_rows = ~X_tr_f.isna().any(axis=1)
    X_tr_f = X_tr_f.loc[valid_rows]
    y_tr_f = y_tr.loc[valid_rows]
    X_va_f = X_va_f.fillna(0)

    var_cols = X_tr_f.var(axis=0)
    keep_var = var_cols[var_cols > 1e-8].index.tolist()
    X_tr_f = X_tr_f[keep_var]
    X_va_f = X_va_f[keep_var]
    if len(keep_var) == 0:
        return np.nan, np.nan, np.nan, False

    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr_f.values)
    X_va_s = sc.transform(X_va_f.values)

    cox_df_tr = pd.DataFrame(X_tr_s, columns=keep_var)
    cox_df_tr['T'] = y_tr_f['survival_days'].values
    cox_df_tr['E'] = y_tr_f['event'].values
    cox_df_va = pd.DataFrame(X_va_s, columns=keep_var)

    cph = None
    for pen_try in [0.5, 1.0, 5.0, 20.0]:
        try:
            cph = CoxPHFitter(penalizer=pen_try, l1_ratio=0.0)
            cph.fit(cox_df_tr, duration_col='T', event_col='E', show_progress=False)
            break
        except Exception:
            cph = None
    if cph is None:
        return np.nan, np.nan, np.nan, False

    risk    = cph.predict_partial_hazard(cox_df_va).values.ravel()
    risk_tr = cph.predict_partial_hazard(cox_df_tr).values.ravel()
    ci = concordance_index(y_va['survival_days'].values, -risk, y_va['event'].values)

    surv_func = cph.predict_survival_function(cox_df_va)
    if median_surv in surv_func.index:
        pred_probs = (1 - surv_func.loc[median_surv]).values
    else:
        pred_probs = np.full(len(cox_df_va), 0.5)
    pred_probs = np.clip(pred_probs, 0.01, 0.99)
    ece = expected_calibration_error(pred_probs, y_va['event'].values)
    bs  = brier_score(pred_probs, y_va['event'].values)

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
    encoder = get_variant('linear_compact',
                          input_dim=X_tr.shape[1], output_dim=output_dim, **variant_kwargs)
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
        epochs=params.get('epochs', 200), lr=params.get('lr', 0.001),
        patience=params.get('patience', 20),
        weight_decay=params.get('weight_decay', 0.0001), verbose=False,
    )
    ci = result['best_val_cindex']
    _maybe_dump_curves(result, curves_ctx)

    encoder.eval()
    with torch.no_grad():
        emb_tr, _ = encoder(X_tr_t, M_tr)
        risk_tr   = result['risk_head'](emb_tr).squeeze(-1).numpy()
        emb, conf_t = encoder(X_va_t, M_va)
        risk = result['risk_head'](emb).squeeze(-1).numpy()
    pred_probs = 1 / (1 + np.exp(-risk))
    pred_probs = np.clip(pred_probs, 0.01, 0.99)
    ece = expected_calibration_error(pred_probs, y_va['event'].values)
    bs  = brier_score(pred_probs, y_va['event'].values)

    _save_predictions_npz(
        artifacts_dir, artifacts_key,
        case_ids_train=y_tr.index.astype(str).values, risk_train=risk_tr,
        times_train=y_tr['survival_days'].values, events_train=y_tr['event'].values,
        case_ids_holdout=y_va.index.astype(str).values, risk_holdout=risk,
        times_holdout=y_va['survival_days'].values, events_holdout=y_va['event'].values,
    )
    _save_neural_checkpoint(
        artifacts_dir, artifacts_key, encoder, result['risk_head'],
        feature_cols=list(X_tr.columns), variant_name='linear_compact',
        variant_kwargs={**variant_kwargs, 'output_dim': output_dim, 'input_dim': X_tr.shape[1]},
    )
    contract = verify_ingestion_contract(emb, conf_t, output_dim, verbose=False)
    return ci, ece, bs, contract.get('contract_satisfied', False)


def _eval_ft_transformer(X_tr, X_va, y_tr, y_va, mask_tr, mask_va, conf_va,
                         output_dim, params,
                         curves_ctx: Optional[dict] = None,
                         artifacts_dir: Optional[Path] = None, artifacts_key: str = ''):
    variant_kwargs = {
        'd_token': params.get('d_token', 192),
        'n_blocks': params.get('n_blocks', 3),
        'n_heads': params.get('n_heads', 8),
        'd_ff': params.get('d_ff', None),
        'dropout': params.get('dropout', 0.1),
    }
    encoder = get_variant('ft_transformer',
                          input_dim=X_tr.shape[1], output_dim=output_dim, **variant_kwargs)
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
        epochs=params.get('epochs', 200), lr=params.get('lr', 3e-4),
        patience=params.get('patience', 20),
        weight_decay=params.get('weight_decay', 1e-4), verbose=False,
    )
    ci = result['best_val_cindex']
    _maybe_dump_curves(result, curves_ctx)

    encoder.eval()
    with torch.no_grad():
        emb_tr, _ = encoder(X_tr_t, M_tr)
        risk_tr   = result['risk_head'](emb_tr).squeeze(-1).numpy()
        emb, conf_t = encoder(X_va_t, M_va)
        risk = result['risk_head'](emb).squeeze(-1).numpy()
    pred_probs = 1 / (1 + np.exp(-risk))
    pred_probs = np.clip(pred_probs, 0.01, 0.99)
    ece = expected_calibration_error(pred_probs, y_va['event'].values)
    bs  = brier_score(pred_probs, y_va['event'].values)

    _save_predictions_npz(
        artifacts_dir, artifacts_key,
        case_ids_train=y_tr.index.astype(str).values, risk_train=risk_tr,
        times_train=y_tr['survival_days'].values, events_train=y_tr['event'].values,
        case_ids_holdout=y_va.index.astype(str).values, risk_holdout=risk,
        times_holdout=y_va['survival_days'].values, events_holdout=y_va['event'].values,
    )
    _save_neural_checkpoint(
        artifacts_dir, artifacts_key, encoder, result['risk_head'],
        feature_cols=list(X_tr.columns), variant_name='ft_transformer',
        variant_kwargs={**variant_kwargs, 'output_dim': output_dim, 'input_dim': X_tr.shape[1]},
    )
    contract = verify_ingestion_contract(emb, conf_t, output_dim, verbose=False)
    return ci, ece, bs, contract.get('contract_satisfied', False)


# ============================================================
# COHORT FILTER — DFS validity + GDC modality manifest (n=224 dense)
# ============================================================

def apply_cohort_filter(df_features, df_targets, cohort_cfg, run_dir):
    audit = {
        'n_initial': int(len(df_features)),
        'filters_applied': [], 'n_after_each_filter': [],
        'dropped_examples': {}, 'n_final': None, 'case_ids_final': [],
    }
    if not cohort_cfg.get('enabled', False):
        log("[COHORT FILTER] DISABLED")
        audit['n_final'] = audit['n_initial']
        audit['case_ids_final'] = list(df_features.index)
        return df_features, df_targets, audit

    log(f"\n[COHORT FILTER] Initial n = {audit['n_initial']}")
    keep_mask = pd.Series(True, index=df_features.index)

    if cohort_cfg.get('require_dfs_valid', False):
        if 'dfs_valid' not in df_targets.columns:
            raise KeyError(
                "cohort_filter.require_dfs_valid=True but df_targets has no 'dfs_valid' column."
            )
        dfs_mask = df_targets.reindex(df_features.index)['dfs_valid'].fillna(False).astype(bool)
        n_dropped = int((~dfs_mask).sum())
        keep_mask &= dfs_mask
        n_after = int(keep_mask.sum())
        audit['filters_applied'].append('require_dfs_valid')
        audit['n_after_each_filter'].append(n_after)
        audit['dropped_examples']['require_dfs_valid'] = df_features.index[~dfs_mask].tolist()[:10]
        log(f"  After require_dfs_valid: {n_after} (dropped {n_dropped})")

    manifest_path = cohort_cfg.get('modality_manifest_path')
    required_modalities = cohort_cfg.get('require_modalities', [])
    if manifest_path and required_modalities:
        manifest_path = Path(manifest_path)
        if not manifest_path.is_absolute():
            manifest_path = Path.cwd() / manifest_path
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"cohort_filter.modality_manifest_path not found: {manifest_path}."
            )
        manifest_df = pd.read_csv(manifest_path).set_index('case_id')
        log(f"  Loaded modality manifest: {manifest_path.name} ({len(manifest_df)} cases)")
        missing_cols = [f'has_{m}' for m in required_modalities if f'has_{m}' not in manifest_df.columns]
        if missing_cols:
            raise KeyError(f"Modality manifest missing columns: {missing_cols}")
        in_manifest = df_features.index.isin(manifest_df.index)
        per_case_pass = pd.Series(False, index=df_features.index)
        for cid in df_features.index[in_manifest]:
            row = manifest_df.loc[cid]
            per_case_pass.loc[cid] = bool(
                all(row.get(f'has_{m}', False) for m in required_modalities)
            )
        n_dropped = int((~per_case_pass).sum())
        keep_mask &= per_case_pass
        n_after = int(keep_mask.sum())
        audit['filters_applied'].append(f"modalities:{','.join(required_modalities)}")
        audit['n_after_each_filter'].append(n_after)
        audit['dropped_examples']['modalities'] = df_features.index[~per_case_pass].tolist()[:10]
        audit['modality_manifest_path'] = str(manifest_path)
        log(f"  After modality intersection: {n_after} (dropped {n_dropped})")

    df_features_f = df_features.loc[keep_mask].copy()
    df_targets_f  = df_targets.reindex(df_features_f.index).copy()
    audit['n_final'] = int(len(df_features_f))
    audit['case_ids_final'] = list(df_features_f.index)
    with open(run_dir / "cohort_manifest.json", 'w') as f:
        json.dump(audit, f, indent=2, default=str)
    log(f"  Final n = {audit['n_final']}.")
    return df_features_f, df_targets_f, audit


# ============================================================
# PHASE 2 HOLDOUT — single 80/20 split per (protocol, seed, variant)
# ============================================================

def phase_2_holdout(df_features, df_targets, config, run_dir, best_imputation):
    phase_cfg = config.get('phase_2_holdout', {})
    if not phase_cfg.get('enabled', False):
        log("[PHASE 2 HOLDOUT] DISABLED")
        return None

    log("\n[PHASE 2 HOLDOUT] Single train/holdout 80/20 per seed × protocol × variant")

    valid = df_targets['survival_days'].notna() & (df_targets['survival_days'] > 0)
    X = df_features.loc[valid].copy()
    y = df_targets.loc[valid].copy()
    median_surv = y['survival_days'].median()
    y['risk_group'] = (y['survival_days'] < median_surv).astype(int)

    seeds = phase_cfg.get('seeds', config['random']['seeds'])
    holdout_fraction = phase_cfg.get('holdout_fraction', 0.20)
    output_dim = phase_cfg.get('output_dim', 768)
    variants = phase_cfg.get('variants', ['cox_baseline'])
    variant_params = phase_cfg.get('variant_params', {})

    imp_for_variants = phase_cfg.get('imputation_for_variants', 'knn_5')
    if imp_for_variants == "auto":
        imp_for_variants = best_imputation
    imp_for_baseline = phase_cfg.get('imputation_for_baseline', 'mean_median')

    protocols_cfg = phase_cfg.get('protocols') or [{'name': '', 'drop_features': []}]
    save_artifacts = bool(phase_cfg.get('save_artifacts', False))
    artifacts_dir = run_dir / "phase2_artifacts" if save_artifacts else None

    log(f"  Cases: {len(X)}, output_dim: {output_dim}")
    log(f"  Holdout fraction: {holdout_fraction}, seeds: {list(seeds)}")
    log(f"  Variants: {variants}")
    log(f"  Imputation baseline / advanced: {imp_for_baseline} / {imp_for_variants}")
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
                idx_all, test_size=holdout_fraction,
                stratify=y['event'].values, random_state=int(seed),
            )
            X_tr_raw, X_ho_raw = X_proto.iloc[tr_idx].copy(), X_proto.iloc[ho_idx].copy()
            y_tr, y_ho = y.iloc[tr_idx].copy(), y.iloc[ho_idx].copy()

            prep_base = TabularPreprocessor()
            X_tr_b, mask_tr_b, conf_tr_b = prep_base.fit_transform(X_tr_raw, get_imputation(imp_for_baseline))
            X_ho_b, mask_ho_b, conf_ho_b = prep_base.transform(X_ho_raw)
            prep_adv = TabularPreprocessor()
            X_tr_a, mask_tr_a, conf_tr_a = prep_adv.fit_transform(X_tr_raw, get_imputation(imp_for_variants))
            X_ho_a, mask_ho_a, conf_ho_a = prep_adv.transform(X_ho_raw)

            input_dim = X_tr_a.shape[1]
            n_events_ho = int(y_ho['event'].sum())
            log(f"    n_train={len(tr_idx)}, n_holdout={len(ho_idx)}, events_holdout={n_events_ho}")

            for variant_name in variants:
                if variant_name == 'cox_baseline':
                    X_tr_use, X_ho_use = X_tr_b, X_ho_b
                    conf_ho_use = conf_ho_b
                else:
                    X_tr_use, X_ho_use = X_tr_a, X_ho_a
                    conf_ho_use = conf_ho_a

                artifacts_key = f"{proto_tag}_seed{int(seed)}_{variant_name}" if save_artifacts else ''
                ci, ece, bs, contract_ok = _evaluate_variant(
                    variant_name=variant_name, input_dim=input_dim, output_dim=output_dim,
                    variant_params=variant_params.get(variant_name, {}),
                    X_tr=X_tr_use, X_va=X_ho_use, y_tr=y_tr, y_va=y_ho,
                    mask_tr=mask_tr_a, mask_va=mask_ho_a, conf_tr=conf_tr_a, conf_va=conf_ho_use,
                    median_surv=median_surv,
                    curves_ctx={'run_dir': run_dir, 'seed': seed,
                                'fold': f'holdout_{proto_tag}', 'variant_name': variant_name},
                    artifacts_dir=artifacts_dir, artifacts_key=artifacts_key,
                )
                rows.append({
                    'protocol': proto_tag, 'seed': int(seed), 'variant': variant_name,
                    'cindex_holdout': float(ci),
                    'ece': float(ece) if ece == ece else float('nan'),
                    'brier_score': float(bs) if bs == bs else float('nan'),
                    'contract_satisfied': bool(contract_ok),
                    'n_train': int(len(tr_idx)), 'n_holdout': int(len(ho_idx)),
                    'events_holdout': n_events_ho, 'n_features': int(X_proto.shape[1]),
                })

    df_results = pd.DataFrame(rows)
    df_results.to_csv(run_dir / "phase2_holdout.csv", index=False)
    summary = (
        df_results.groupby(['protocol', 'variant'])
        .agg(
            cindex_mean=('cindex_holdout', 'mean'),
            cindex_std=('cindex_holdout', 'std'),
            cindex_median=('cindex_holdout', 'median'),
            ece_mean=('ece', 'mean'),
            brier_mean=('brier_score', 'mean'),
            contract_satisfied=('contract_satisfied', 'all'),
            n_seeds=('cindex_holdout', 'count'),
        ).round(4)
    )
    summary.to_csv(run_dir / "phase2_holdout_summary.csv")
    log("\n  HOLDOUT SUMMARY:")
    log(summary.to_string())
    return df_results


# ============================================================
# PHASE 2 MAHOOTIHA — Spearman + RF + top-K replication
# ============================================================

def phase_2_mahootiha(df_features, df_targets, config, run_dir, best_imputation):
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
    k_raw = phase_cfg.get('k_values', [5, 10, 15, None])
    k_values = sorted(set(
        n_total_features if k is None else min(int(k), n_total_features) for k in k_raw
    ))
    rf_n_estimators = phase_cfg.get('rf_n_estimators', 200)
    imp_name = phase_cfg.get('imputation', 'knn_5')
    if imp_name == 'auto':
        imp_name = best_imputation
    output_dim = phase_cfg.get('output_dim', 768)

    log(f"  Cases: {len(X)}, total features: {n_total_features}")
    log(f"  Seeds: {list(seeds)}, K values: {k_values}")
    log(f"  RF n_estimators: {rf_n_estimators}, imputation: {imp_name}")

    # Multi-protocol support (mirrors phase_2_holdout): each protocol drops
    # listed columns from df_features BEFORE the 80/20 split so train/holdout
    # indices stay aligned across protocols for the same seed.
    protocols_cfg = phase_cfg.get('protocols') or [{'name': '', 'drop_features': []}]
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

        proto_k_values = sorted(set(
            X_proto.shape[1] if k is None else min(int(k), X_proto.shape[1]) for k in k_raw
        ))

        for seed in seeds:
            log(f"  Seed {seed}")
            tr_idx, ho_idx = train_test_split(
                np.arange(len(X_proto)), test_size=holdout_fraction,
                stratify=y['event'].values, random_state=int(seed),
            )
            X_tr_raw, X_ho_raw = X_proto.iloc[tr_idx].copy(), X_proto.iloc[ho_idx].copy()
            y_tr, y_ho = y.iloc[tr_idx].copy(), y.iloc[ho_idx].copy()

            prep = TabularPreprocessor()
            X_tr_imp, _, _ = prep.fit_transform(X_tr_raw, get_imputation(imp_name))
            X_ho_imp, _, conf_ho = prep.transform(X_ho_raw)

            spearman_corrs = {}
            for col in X_tr_imp.columns:
                try:
                    corr, _ = spearmanr(X_tr_imp[col].values, y_tr['survival_days'].values)
                    spearman_corrs[col] = abs(corr) if not np.isnan(corr) else 0.0
                except Exception:
                    spearman_corrs[col] = 0.0

            survival_arr = y_tr['survival_days'].values
            event_arr    = y_tr['event'].values
            y_bin = (survival_arr < median_surv).astype(int)
            keep_for_rf = ~((event_arr == 0) & (survival_arr < median_surv))
            rf = RandomForestClassifier(n_estimators=rf_n_estimators, random_state=int(seed), n_jobs=-1)
            rf.fit(X_tr_imp.values[keep_for_rf], y_bin[keep_for_rf])
            rf_imp = dict(zip(X_tr_imp.columns, rf.feature_importances_))

            spearman_rank = pd.Series(spearman_corrs).rank(ascending=False, method='average')
            rf_rank       = pd.Series(rf_imp).rank(ascending=False, method='average')
            combined_rank = ((spearman_rank + rf_rank) / 2).sort_values()
            rankings_per_seed[proto_tag][int(seed)] = pd.DataFrame({
                'spearman_corr_abs': pd.Series(spearman_corrs),
                'rf_importance':     pd.Series(rf_imp),
                'spearman_rank':     spearman_rank,
                'rf_rank':           rf_rank,
                'combined_rank':     combined_rank,
            }).reindex(combined_rank.index)

            for K in proto_k_values:
                top_features = combined_rank.head(K).index.tolist()
                X_tr_K = X_tr_imp[top_features]
                X_ho_K = X_ho_imp[top_features]
                try:
                    ci, ece, bs, _ = _eval_cox_baseline(
                        X_tr_K, X_ho_K, y_tr, y_ho,
                        conf_va=conf_ho, output_dim=output_dim, median_surv=median_surv,
                    )
                except Exception as e:
                    log(f"    K={K} seed={seed} cox FAILED: {e}", level="warn")
                    ci, ece, bs = float('nan'), float('nan'), float('nan')
                rows.append({
                    'protocol': proto_tag,
                    'seed': int(seed), 'K': int(K),
                    'cindex_holdout': float(ci) if ci == ci else float('nan'),
                    'ece': float(ece) if ece == ece else float('nan'),
                    'brier_score': float(bs) if bs == bs else float('nan'),
                    'n_train': int(len(tr_idx)), 'n_holdout': int(len(ho_idx)),
                    'n_features': int(X_proto.shape[1]),
                    'top_features': ','.join(top_features),
                })

    df_results = pd.DataFrame(rows)
    df_results.to_csv(run_dir / "phase2_mahootiha.csv", index=False)
    group_cols = ['protocol', 'K'] if 'protocol' in df_results.columns else ['K']
    summary = (
        df_results.groupby(group_cols)['cindex_holdout']
        .agg(cindex_mean='mean', cindex_std='std', cindex_median='median', n_seeds='count')
        .round(4)
    )
    summary.to_csv(run_dir / "phase2_mahootiha_summary.csv")
    log("\n  MAHOOTIHA top-K HOLDOUT SUMMARY:")
    log(summary.to_string())

    for proto_tag, seed_rankings in rankings_per_seed.items():
        if not seed_rankings:
            continue
        rank_matrix = pd.DataFrame({
            f'rank_seed{s}': r['combined_rank'] for s, r in seed_rankings.items()
        })
        rank_matrix['mean_combined_rank'] = rank_matrix.mean(axis=1)
        rank_matrix['std_combined_rank']  = rank_matrix.std(axis=1)
        rank_matrix = rank_matrix.sort_values('mean_combined_rank')
        suffix = f"_{proto_tag}" if proto_tag != 'default' else ''
        rank_matrix.to_csv(run_dir / f"phase2_mahootiha_feature_ranking{suffix}.csv")
        log(f"\n  TOP 10 FEATURES BY MEAN COMBINED RANK (protocol={proto_tag}):")
        log(rank_matrix[['mean_combined_rank', 'std_combined_rank']].head(10).to_string())
    return df_results


# ============================================================
# DRIVER
# ============================================================

def run_experiment(config_path: Union[str, Path]):
    config_path = Path(config_path)
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Resolve relative feature_config path
    feat_path = Path(config['data']['feature_config'])
    if not feat_path.is_absolute():
        base_dir = config_path.parent if config_path else Path.cwd()
        feat_path = (base_dir / feat_path).resolve()
        config['data']['feature_config'] = str(feat_path)

    run_dir = create_run_directory(config, config_path)
    print("=" * 70)
    print(f"LEAKAGE-SURVIVAL-PROTOCOL EXPERIMENT")
    print(f"Name:      {config['experiment']['name']}")
    print(f"Hash:      {compute_config_hash(config)}")
    print(f"Run dir:   {run_dir}")
    print("=" * 70)

    t_start = time.time()
    summary = {'phases': {}, 'errors': []}

    # Step 0: extraction
    log("\n[STEP 0] Extracting clinical data from XMLs")
    extractor = TCGAExtractor(config['data']['feature_config'])
    df_features, df_targets = extractor.extract_cohort(config['data']['xml_dir'])
    if config['output'].get('save_raw_extraction', True):
        df_features.to_csv(run_dir / "raw_features.csv")
        df_targets.to_csv(run_dir / "raw_targets.csv")

    # Step 0.5: optional cohort filter
    cohort_cfg = config.get('cohort_filter', {})
    if cohort_cfg:
        df_features, df_targets, cohort_audit = apply_cohort_filter(
            df_features, df_targets, cohort_cfg, run_dir,
        )
        summary['cohort_filter'] = {
            'n_initial': cohort_audit['n_initial'],
            'n_final':   cohort_audit['n_final'],
            'filters_applied': cohort_audit['filters_applied'],
            'n_after_each_filter': cohort_audit['n_after_each_filter'],
        }

    summary['n_cases']    = int(len(df_features))
    summary['n_features'] = int(df_features.shape[1])
    summary['n_events']   = int(df_targets['event'].sum())

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
        ph2_ho = phase_2_holdout(df_features, df_targets, config, run_dir, best_imp)
        if ph2_ho is not None:
            group_cols = ['protocol', 'variant'] if 'protocol' in ph2_ho.columns else ['variant']
            ph2_ho_summary = (
                ph2_ho.groupby(group_cols)['cindex_holdout']
                .agg(['mean', 'std', 'median']).round(4).reset_index()
            )
            summary['phases']['phase_2_holdout'] = ph2_ho_summary.to_dict(orient='records')
    except Exception as e:
        summary['errors'].append({'phase': 'phase_2_holdout', 'error': str(e)})
        if fail_fast: raise

    try:
        ph2_mah = phase_2_mahootiha(df_features, df_targets, config, run_dir, best_imp)
        if ph2_mah is not None:
            ph2_mah_summary = (
                ph2_mah.groupby('K')['cindex_holdout']
                .agg(['mean', 'std', 'median', 'count']).round(4).reset_index()
            )
            summary['phases']['phase_2_mahootiha'] = ph2_mah_summary.to_dict(orient='records')
    except Exception as e:
        summary['errors'].append({'phase': 'phase_2_mahootiha', 'error': str(e)})
        if fail_fast: raise

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
    if len(sys.argv) < 2:
        print("Usage: python -m src.runner <config.yaml>")
        sys.exit(1)
    run_experiment(sys.argv[1])
