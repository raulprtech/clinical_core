"""
Bootstrap CIs and paired significance tests for survival C-index.

The C-index has no DeLong test in any standard library because the original
DeLong statistic is defined for the binary-classification AUC, not for the
right-censored concordance index. The paper's reference to "DeLong test" is
interpreted operationally here as the natural paired-sample analogue: a
permutation / bootstrap-based test of the difference in C-index between two
risk score vectors evaluated on the *same* held-out patients.

Two helpers are exposed:

  bootstrap_cindex_ci(risk, time, event, n_iter, seed)
      Resample patients with replacement, recompute C-index per resample,
      return mean and (lo, hi) percentile interval.

  paired_cindex_test(risk_a, risk_b, time, event, n_iter, seed)
      Bootstrap the per-resample difference C(b) - C(a). Returns observed
      delta, 95% CI of the delta, and a two-sided bootstrap p-value
      (fraction of resamples that cross zero, doubled and clipped at 1.0).
      This is the right paired test when the two risk vectors come from
      models evaluated on identical patient sets, which is exactly the
      limpio-vs-permisivo setup of phase_2_holdout.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from lifelines.utils import concordance_index


def _cindex(time: np.ndarray, risk: np.ndarray, event: np.ndarray) -> float:
    # lifelines convention: higher score => longer survival, so we pass -risk
    return float(concordance_index(time, -risk, event))


def bootstrap_cindex_ci(
    risk: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    n_iter: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> Tuple[float, float, float, float]:
    """
    Returns (point_estimate, mean_bootstrap, lo, hi).

    Stratified resampling on the event indicator keeps the censoring
    ratio constant across bootstrap samples, which is the convention used
    in survival evaluation studies.
    """
    risk = np.asarray(risk, dtype=np.float64)
    time = np.asarray(time, dtype=np.float64)
    event = np.asarray(event, dtype=np.int64)
    n = len(risk)
    if n == 0:
        return (np.nan, np.nan, np.nan, np.nan)

    point = _cindex(time, risk, event)

    rng = np.random.default_rng(seed)
    idx_pos = np.where(event == 1)[0]
    idx_neg = np.where(event == 0)[0]
    samples = np.empty(n_iter, dtype=np.float64)
    for i in range(n_iter):
        rs_pos = rng.choice(idx_pos, size=len(idx_pos), replace=True) if len(idx_pos) else idx_pos
        rs_neg = rng.choice(idx_neg, size=len(idx_neg), replace=True) if len(idx_neg) else idx_neg
        idx = np.concatenate([rs_pos, rs_neg])
        try:
            samples[i] = _cindex(time[idx], risk[idx], event[idx])
        except Exception:
            samples[i] = np.nan
    samples = samples[~np.isnan(samples)]
    if len(samples) == 0:
        return (point, np.nan, np.nan, np.nan)
    lo = float(np.percentile(samples, 100 * alpha / 2))
    hi = float(np.percentile(samples, 100 * (1 - alpha / 2)))
    return (point, float(samples.mean()), lo, hi)


def paired_cindex_test(
    risk_a: np.ndarray,
    risk_b: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    n_iter: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict:
    """
    Paired bootstrap test for the difference in C-index when the same
    patients (time, event) are scored by two different risk vectors.

    Returns a dict with:
      delta_observed: C(b) - C(a) on the observed sample
      delta_mean    : mean of bootstrap deltas
      delta_lo, delta_hi : 95% percentile CI of the delta
      p_value       : two-sided bootstrap p-value
      n_resamples   : number of non-degenerate resamples that contributed
    """
    risk_a = np.asarray(risk_a, dtype=np.float64)
    risk_b = np.asarray(risk_b, dtype=np.float64)
    time = np.asarray(time, dtype=np.float64)
    event = np.asarray(event, dtype=np.int64)
    n = len(time)
    if not (len(risk_a) == len(risk_b) == n):
        raise ValueError("risk_a, risk_b, time and event must have identical length")

    obs_a = _cindex(time, risk_a, event)
    obs_b = _cindex(time, risk_b, event)
    delta_obs = obs_b - obs_a

    rng = np.random.default_rng(seed)
    idx_pos = np.where(event == 1)[0]
    idx_neg = np.where(event == 0)[0]
    deltas = np.empty(n_iter, dtype=np.float64)
    for i in range(n_iter):
        rs_pos = rng.choice(idx_pos, size=len(idx_pos), replace=True) if len(idx_pos) else idx_pos
        rs_neg = rng.choice(idx_neg, size=len(idx_neg), replace=True) if len(idx_neg) else idx_neg
        idx = np.concatenate([rs_pos, rs_neg])
        try:
            ca = _cindex(time[idx], risk_a[idx], event[idx])
            cb = _cindex(time[idx], risk_b[idx], event[idx])
            deltas[i] = cb - ca
        except Exception:
            deltas[i] = np.nan
    deltas = deltas[~np.isnan(deltas)]
    if len(deltas) == 0:
        return {
            'delta_observed': delta_obs,
            'delta_mean': np.nan, 'delta_lo': np.nan, 'delta_hi': np.nan,
            'p_value': np.nan, 'n_resamples': 0,
        }

    delta_lo = float(np.percentile(deltas, 100 * alpha / 2))
    delta_hi = float(np.percentile(deltas, 100 * (1 - alpha / 2)))
    # Two-sided bootstrap p-value: probability that the resampled delta
    # is on the opposite side of zero relative to the observed delta.
    if delta_obs >= 0:
        p = float(np.mean(deltas <= 0)) * 2
    else:
        p = float(np.mean(deltas >= 0)) * 2
    p = min(max(p, 1.0 / (len(deltas) + 1)), 1.0)

    return {
        'delta_observed': float(delta_obs),
        'cindex_a': float(obs_a),
        'cindex_b': float(obs_b),
        'delta_mean': float(deltas.mean()),
        'delta_lo': delta_lo,
        'delta_hi': delta_hi,
        'p_value': p,
        'n_resamples': int(len(deltas)),
    }
