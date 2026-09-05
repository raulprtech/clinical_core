import sys
import unittest
from pathlib import Path

import numpy as np


TOOLS_ROOT = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from evaluate_train_scaled_sequence_ensemble import (  # noqa: E402
    train_ecdf_percentile,
    train_zscore,
)


class TrainScaledSequenceEnsembleTests(unittest.TestCase):
    def test_ecdf_uses_reference_distribution(self):
        reference = np.array([0.0, 1.0, 2.0, 3.0])
        heldout = np.array([-1.0, 0.5, 2.0, 4.0])
        result = train_ecdf_percentile(reference, heldout)
        self.assertTrue(np.allclose(result, [0.0, 0.25, 0.75, 1.0]))

    def test_ecdf_is_invariant_to_shared_positive_affine_scale(self):
        reference = np.array([-2.0, 0.0, 4.0, 7.0])
        heldout = np.array([-3.0, 1.0, 9.0])
        first = train_ecdf_percentile(reference, heldout)
        second = train_ecdf_percentile(
            13.0 * reference - 8.0, 13.0 * heldout - 8.0
        )
        self.assertTrue(np.array_equal(first, second))

    def test_zscore_uses_train_mean_and_standard_deviation(self):
        reference = np.array([0.0, 2.0, 4.0])
        heldout = np.array([2.0, 5.0])
        result, mean, std = train_zscore(reference, heldout)
        self.assertAlmostEqual(mean, 2.0)
        self.assertAlmostEqual(std, float(reference.std(ddof=0)))
        self.assertTrue(np.allclose(result, (heldout - mean) / std))


if __name__ == "__main__":
    unittest.main()
