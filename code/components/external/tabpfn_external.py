"""
TabPFN v2 — EXTERNAL non-compliant baseline.

This is NOT a CLINICAL-CORE component. TabPFN v2 does not satisfy the
TABULAR-IN contract because it is an in-context foundation model for direct
prediction (no representational embedding layer in the contract-compliant
sense). Its rejection as a TABULAR-IN variant is documented in protocol v12
section 7.5.

Here we use TabPFN as a DIRECT survival predictor via the classification-based
survival formulation (Kim et al., 2026, "Tabular Foundation Models Can Do
Survival Analysis"). This is the most rigorous way to use TabPFN for survival
analysis: a single binary classification task at the median survival time of
the training cohort, where the positive class is "event before median time."

Why classification-based survival (and not, e.g., multi-task at many quantiles):
    • Simple, interpretable, works with any binary classifier.
    • TabPFN v2 is a classifier, not a regressor, so this is the natural fit.
    • C-index computation uses the classifier's positive-class probability as
      the risk score.
    • Kim et al. 2026 show this formulation is competitive with and often
      superior to purpose-built deep survival baselines.

Reference:
    Hollmann et al. "Accurate predictions on small data with a tabular
    foundation model." Nature (2025).
    Kim, Lai, Zhang. "Tabular Foundation Models Can Do Survival Analysis."
    Preprint, 2026.

Comparability to compliant variants:
    • Same 19 preprocessed features as inputs.
    • Same KNN k=5 imputation (imputation_for_variants).
    • Same stratified K-fold partition and seed list.
    • Evaluated on same C-index metric.
    • Reported in summary.json under `external_baselines.tabpfn_external`.
"""
from __future__ import annotations

import os
import time
import warnings
from typing import Dict, Any, Optional

from dotenv import load_dotenv
import numpy as np
import pandas as pd

# Attempt to load token from .env automatically
load_dotenv()


class TabPFNExternalBaseline:
    """
    Wrapper around TabPFNClassifier using classification-based survival.

    Non-compliant by design: no embedding output, no projection to 768-dim.
    The risk score is the TabPFN classifier's positive-class probability
    for the binary event "death before median training time."
    """

    contract_compliant: bool = False
    name: str = "tabpfn_external"

    def __init__(
        self,
        device: str = 'auto',
        n_estimators: int = 4,
        random_state: int = 42,
    ):
        # Lazy import so environments without tabpfn do not break on module
        # import — only on first use.
        try:
            from tabpfn import TabPFNClassifier
        except ImportError as e:
            raise ImportError(
                "tabpfn is required for TabPFNExternalBaseline. "
                "Install with: pip install tabpfn"
            ) from e

        # Device resolution
        if device == 'auto':
            try:
                import torch
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            except ImportError:
                device = 'cpu'
            if device == 'cpu':
                warnings.warn(
                    "TabPFN runs significantly slower on CPU; a GPU is "
                    "recommended for timely execution."
                )

        self.model = TabPFNClassifier(
            device=device,
            n_estimators=n_estimators,
            random_state=random_state,
            token=os.getenv("TABPFN_API_KEY")
        )
        self._hyperparams = {
            'device': device,
            'n_estimators': n_estimators,
            'random_state': random_state,
            'survival_formulation': 'classification_based_at_median',
        }
        self._median_time: Optional[float] = None
        self._fit_time_s: float = 0.0
        self._inference_time_ms: float = 0.0

    def fit(
        self,
        X: pd.DataFrame,
        survival_days: np.ndarray,
        event: np.ndarray,
    ) -> None:
        """
        Fit TabPFN on the binary task "death before median training time."

        Cases where event=0 (censored) AND survival_days < median are excluded
        from training because their label is unknowable: we cannot assert the
        patient died before median. Cases where event=0 AND survival_days
        >= median are labeled as negative (confirmed alive at median). Cases
        with event=1 are labeled based on whether death happened before
        median.

        This is the standard handling in Kim et al. 2026.
        """
        survival_days = np.asarray(survival_days, dtype=float)
        event = np.asarray(event, dtype=int)

        # Median of observed event times (or of all times if few events)
        events_mask = event == 1
        if events_mask.sum() >= 10:
            self._median_time = float(np.median(survival_days[events_mask]))
        else:
            self._median_time = float(np.median(survival_days))

        # Construct binary label with informative-censoring handling
        y_bin = (survival_days < self._median_time).astype(int)
        keep = ~((event == 0) & (survival_days < self._median_time))

        X_train = X.iloc[keep].values.astype(np.float32)
        y_train = y_bin[keep]

        t0 = time.time()
        self.model.fit(X_train, y_train)
        self._fit_time_s = time.time() - t0

    def predict_risk(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return the probability of the positive class (event before median).
        Higher → higher risk → used as risk score for C-index.
        """
        t0 = time.time()
        proba = self.model.predict_proba(X.values.astype(np.float32))
        self._inference_time_ms = (time.time() - t0) * 1000 / max(len(X), 1)
        # positive class is index 1 (event=True before median)
        # guard against edge cases where only one class appears in training
        if proba.shape[1] == 1:
            # All training labels were the same class; cannot discriminate
            return np.full(len(X), 0.5, dtype=np.float64)
        return np.asarray(proba[:, 1], dtype=np.float64)

    def summary(self) -> Dict[str, Any]:
        """Return a JSON-serializable summary for inclusion in summary.json."""
        return {
            'name': self.name,
            'contract_compliant': False,
            'model_type': 'foundation_model_tabular',
            'hyperparameters': self._hyperparams,
            'median_time_threshold_days': self._median_time,
            'fit_time_seconds': round(self._fit_time_s, 4),
            'inference_time_ms_per_case': round(self._inference_time_ms, 4),
        }