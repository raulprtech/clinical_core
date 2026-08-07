import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from core.experiment_runner import survival_ipcw_calibration  # noqa: E402
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
