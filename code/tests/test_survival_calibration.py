import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from core.experiment_runner import (  # noqa: E402
    bootstrap_concordance_interval,
    ipcw_concordance,
    survival_ipcw_calibration,
)
from components.adapters.ingestion.tabular.utils.imputation_benchmark import (  # noqa: E402
    MeanMedianImputer,
    TabularPreprocessor,
)


class SurvivalCalibrationTests(unittest.TestCase):
    def test_ipcw_metrics_are_not_binary_event_ever_metrics(self):
        train = pd.DataFrame({"survival_days": [100, 300, 500, 900], "event": [1, 0, 1, 0]})
        test = pd.DataFrame({"survival_days": [100, 200, 800, 900], "event": [1, 0, 1, 0]})
        ece, brier = survival_ipcw_calibration(
            np.array([0.8, 0.7, 0.2, 0.1]), train, test, horizon=365,
        )
        self.assertTrue(np.isfinite(ece))
        self.assertTrue(np.isfinite(brier))
        self.assertLess(brier, 0.25)

    def test_bootstrap_concordance_interval_is_deterministic(self):
        times = np.arange(10, 210, 10, dtype=float)
        events = np.ones(20, dtype=int)
        risk = -times
        first = bootstrap_concordance_interval(
            times, events, risk, iterations=200, confidence_level=0.95, seed=7,
        )
        second = bootstrap_concordance_interval(
            times, events, risk, iterations=200, confidence_level=0.95, seed=7,
        )
        self.assertEqual(first, second)
        self.assertEqual(first[0], 1.0)
        self.assertEqual(first[1], 1.0)
        self.assertEqual(first[2], 200)

    def test_bootstrap_concordance_handles_no_observed_events(self):
        lower, upper, valid = bootstrap_concordance_interval(
            np.array([10.0, 20.0]),
            np.array([0, 0]),
            np.array([0.2, 0.1]),
            iterations=20,
        )
        self.assertTrue(np.isnan(lower))
        self.assertTrue(np.isnan(upper))
        self.assertEqual(valid, 0)

    def test_uno_ipcw_concordance_is_finite(self):
        train = pd.DataFrame({
            "survival_days": [100, 180, 260, 400, 600, 800],
            "event": [1, 0, 1, 1, 0, 1],
        })
        test = pd.DataFrame({
            "survival_days": [120, 220, 350, 500],
            "event": [1, 0, 1, 0],
        })
        risk = np.array([0.9, 0.6, 0.4, 0.1])
        result = ipcw_concordance(train, test, risk, tau=500)
        self.assertTrue(np.isfinite(result))
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)

    def test_declared_nominal_column_is_onehot_encoded(self):
        train = pd.DataFrame({"race": [0.0, 1.0, 2.0, 0.0], "age": [50.0, 60.0, 70.0, 55.0]})
        holdout = pd.DataFrame({"race": [3.0], "age": [65.0]})
        prep = TabularPreprocessor(onehot_columns=["race"])
        encoded_train, _, _ = prep.fit_transform(train, MeanMedianImputer())
        encoded_holdout, _, _ = prep.transform(holdout)
        self.assertNotIn("race", encoded_train.columns)
        self.assertTrue(any(col.startswith("race_") for col in encoded_train.columns))
        self.assertEqual(list(encoded_train.columns), list(encoded_holdout.columns))
        self.assertEqual(float(encoded_holdout.filter(like="race_").sum(axis=1).iloc[0]), 0.0)


if __name__ == "__main__":
    unittest.main()
