"""
Calibration metrics for right-censored survival prediction.

Two outputs:

  integrated_brier_score(y_train, y_test, surv_probs_test, time_grid)
      Standard IBS via scikit-survival, normalised over the supplied
      time grid. Equivalent to the Graf et al. (1999) estimator.

  calibration_curve_at_horizon(y_test, surv_probs_at_t, n_bins)
      Binned reliability diagram for a single horizon t: bin patients
      by predicted survival probability and compare against observed
      Kaplan-Meier survival within the bin.

  cox_breslow_survival_for_neural(risk_train, time_train, event_train,
                                  risk_test, time_grid)
      Closed-form Breslow estimator of the baseline cumulative hazard
      using the training risk scores, then S(t|x) = exp(-H0(t) * exp(risk_x)).
      Lets neural Cox heads produce survival functions without retraining
      anything on disk.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def cox_breslow_survival_for_neural(
    risk_train: np.ndarray,
    time_train: np.ndarray,
    event_train: np.ndarray,
    risk_test: np.ndarray,
    time_grid: np.ndarray,
) -> np.ndarray:
    """
    Breslow estimator of S(t|x) for a neural Cox head.

    Parameters
    ----------
    risk_train : (n_train,)
        The linear predictor (log hazard ratio) on the training set.
        For neural heads this is risk_head(encoder(x_train)).squeeze(-1).
    time_train, event_train : (n_train,)
    risk_test  : (n_test,)
    time_grid  : (n_times,) ascending

    Returns
    -------
    surv : (n_times, n_test) survival probabilities at each grid time.
    """
    risk_train = np.asarray(risk_train, dtype=np.float64)
    time_train = np.asarray(time_train, dtype=np.float64)
    event_train = np.asarray(event_train, dtype=np.int64)
    risk_test = np.asarray(risk_test, dtype=np.float64)
    time_grid = np.asarray(time_grid, dtype=np.float64)

    exp_r = np.exp(risk_train)
    order = np.argsort(time_train)
    t_sorted = time_train[order]
    e_sorted = event_train[order]
    exp_sorted = exp_r[order]

    # At each event time, increment H0 by d_i / sum_{j in risk set}(exp(risk_j))
    event_times = np.unique(t_sorted[e_sorted == 1])
    h0 = np.zeros(len(event_times), dtype=np.float64)
    for i, t in enumerate(event_times):
        d = int(np.sum((t_sorted == t) & (e_sorted == 1)))
        denom = float(np.sum(exp_sorted[t_sorted >= t]))
        if denom > 0:
            h0[i] = d / denom
    # Cumulative baseline hazard at each query time = sum of jumps with event_time <= t
    H0_grid = np.array([h0[event_times <= t].sum() for t in time_grid], dtype=np.float64)

    # S(t|x) = exp(-H0(t) * exp(risk_x))
    surv = np.exp(-np.outer(H0_grid, np.exp(risk_test)))   # (n_times, n_test)
    return surv


def integrated_brier_score(
    times_train: np.ndarray, events_train: np.ndarray,
    times_test: np.ndarray, events_test: np.ndarray,
    surv_probs_test: np.ndarray,
    time_grid: np.ndarray,
) -> float:
    """
    Integrated Brier Score via scikit-survival.

    Parameters
    ----------
    surv_probs_test : (n_times, n_test) survival probabilities S(t_grid | x_i).
    """
    from sksurv.metrics import integrated_brier_score as sks_ibs
    from sksurv.util import Surv

    y_train = Surv.from_arrays(events_train.astype(bool), times_train)
    y_test = Surv.from_arrays(events_test.astype(bool), times_test)
    # sksurv expects (n_test, n_times)
    surv_t = np.asarray(surv_probs_test).T
    return float(sks_ibs(y_train, y_test, surv_t, time_grid))


def calibration_curve_at_horizon(
    times_test: np.ndarray, events_test: np.ndarray,
    surv_probs_at_t: np.ndarray, horizon: float, n_bins: int = 10,
) -> dict:
    """
    Binned reliability diagram at a single horizon t = horizon.

    Patients are grouped into n_bins by their predicted S(t|x). Within each
    bin the observed survival at t is estimated by Kaplan-Meier on the bin's
    subset and compared against the bin's mean predicted S(t|x).

    Returns a dict with arrays: bin_mid_pred, observed_surv, n_per_bin.
    """
    from lifelines import KaplanMeierFitter

    surv_probs_at_t = np.asarray(surv_probs_at_t, dtype=np.float64)
    times_test = np.asarray(times_test, dtype=np.float64)
    events_test = np.asarray(events_test, dtype=np.int64)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    mids, observed, counts = [], [], []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (surv_probs_at_t >= lo) & (surv_probs_at_t < hi)
        n = int(mask.sum())
        if n < 5:
            continue
        kmf = KaplanMeierFitter().fit(times_test[mask], events_test[mask])
        try:
            obs = float(kmf.predict(horizon))
        except Exception:
            obs = float('nan')
        mids.append(float(surv_probs_at_t[mask].mean()))
        observed.append(obs)
        counts.append(n)

    return {
        'horizon': float(horizon),
        'bin_mid_pred': np.array(mids),
        'observed_surv': np.array(observed),
        'n_per_bin': np.array(counts, dtype=np.int64),
    }


def cox_survival_at_grid_from_lifelines_obj(cph, X_test_scaled_df, time_grid):
    """
    Lifelines doesn't accept arbitrary query times directly; use
    predict_survival_function and resample onto time_grid via step-function
    (last-known carry-forward).
    """
    sf = cph.predict_survival_function(X_test_scaled_df)
    sf_times = np.asarray(sf.index.values, dtype=np.float64)
    sf_vals = sf.values  # (n_times_native, n_test)
    out = np.empty((len(time_grid), sf_vals.shape[1]), dtype=np.float64)
    for j, t in enumerate(time_grid):
        # last sf_time <= t
        idx = np.searchsorted(sf_times, t, side='right') - 1
        if idx < 0:
            out[j, :] = 1.0
        else:
            out[j, :] = sf_vals[idx, :]
    return out
