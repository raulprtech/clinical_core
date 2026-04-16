"""
Random Survival Forests — EXTERNAL non-compliant baseline.

This is NOT a CLINICAL-CORE component. It does not satisfy the TABULAR-IN
contract because it does not produce representational embeddings of fixed
dimensionality — RSF is a direct survival predictor (risk score per case).

Role in the experimental design:
    Serves as an ensemble-tree SOTA reference point for tabular survival
    analysis, complementing the contract-compliant encoders (cox_baseline,
    linear_compact, ft_transformer) and the other external DL baseline
    (TabPFN). Having both a tree-based and a neural-pretrained external SOTA
    bounds the upper performance envelope of tabular-only survival.

Reference:
    Ishwaran, Kogalur, Blackstone, Lauer. "Random survival forests."
    The Annals of Applied Statistics 2(3), 2008.

Comparability to compliant variants:
    • Same 19 preprocessed features as inputs.
    • Same KNN k=5 imputation (imputation_for_variants).
    • Same stratified K-fold partition and seed list.
    • Evaluated on same C-index metric.
    • Reported in summary.json under `external_baselines.rsf_external` with
      `contract_compliant: false` to preserve provenance.
"""
from __future__ import annotations

import time
from typing import Dict, Any

import numpy as np
import pandas as pd


class RSFExternalBaseline:
    """
    Wrapper around sksurv.ensemble.RandomSurvivalForest with a uniform
    interface to match how the experiment_runner invokes other baselines.

    Non-compliant by design: exposes .fit(X, y_struct) and .predict_risk(X)
    directly, no embedding layer.
    """

    contract_compliant: bool = False
    name: str = "rsf_external"

    def __init__(
        self,
        n_estimators: int = 100,
        min_samples_split: int = 10,
        min_samples_leaf: int = 15,
        max_features: str = 'sqrt',
        n_jobs: int = -1,
        random_state: int = 42,
    ):
        # Lazy import so environments without scikit-survival do not break
        # on module import — only on first use.
        try:
            from sksurv.ensemble import RandomSurvivalForest
        except ImportError as e:
            raise ImportError(
                "scikit-survival is required for RSFExternalBaseline. "
                "Install with: pip install scikit-survival"
            ) from e

        self.model = RandomSurvivalForest(
            n_estimators=n_estimators,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            n_jobs=n_jobs,
            random_state=random_state,
        )
        self._hyperparams = {
            'n_estimators': n_estimators,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'max_features': max_features,
            'random_state': random_state,
        }
        self._fit_time_s: float = 0.0
        self._inference_time_ms: float = 0.0

    @staticmethod
    def _build_structured_target(
        survival_days: np.ndarray,
        event: np.ndarray,
    ) -> np.ndarray:
        """
        Build the structured array expected by sksurv: dtype with fields
        (event: bool, time: float).
        """
        n = len(survival_days)
        y_struct = np.empty(n, dtype=[('event', bool), ('time', float)])
        y_struct['event'] = event.astype(bool)
        y_struct['time'] = survival_days.astype(float)
        return y_struct

    def fit(
        self,
        X: pd.DataFrame,
        survival_days: np.ndarray,
        event: np.ndarray,
    ) -> None:
        """Train the RSF. Timed so the summary.json reports training cost."""
        y_struct = self._build_structured_target(survival_days, event)
        t0 = time.time()
        self.model.fit(X.values.astype(np.float32), y_struct)
        self._fit_time_s = time.time() - t0

    def predict_risk(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return a per-case risk score. Higher score → higher risk of event.
        Used for C-index via `concordance_index(T, -risk, E)`.
        """
        t0 = time.time()
        risk = self.model.predict(X.values.astype(np.float32))
        self._inference_time_ms = (time.time() - t0) * 1000 / max(len(X), 1)
        return np.asarray(risk, dtype=np.float64).ravel()

    def summary(self) -> Dict[str, Any]:
        """Return a JSON-serializable summary for inclusion in summary.json."""
        return {
            'name': self.name,
            'contract_compliant': False,
            'model_type': 'ensemble_trees',
            'hyperparameters': self._hyperparams,
            'fit_time_seconds': round(self._fit_time_s, 4),
            'inference_time_ms_per_case': round(self._inference_time_ms, 4),
        }
