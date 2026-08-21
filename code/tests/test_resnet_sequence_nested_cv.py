import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


TOOLS_ROOT = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from evaluate_resnet_sequence_nested_cv import (  # noqa: E402
    clustered_patient_bootstrap,
    summarize_repeats,
)


def _predictions() -> pd.DataFrame:
    rows = []
    survival = np.array([5, 7, 9, 12, 14, 18, 21, 25], dtype=float)
    events = np.array([1, 0, 1, 1, 0, 1, 0, 1], dtype=int)
    for repeat in range(2):
        official = np.array([0.8, 0.6, 0.7, 0.5, 0.4, 0.3, 0.2, 0.1])
        mamba = np.array([0.9, 0.7, 0.8, 0.55, 0.45, 0.35, 0.25, 0.05])
        if repeat:
            official = official + np.linspace(-0.02, 0.02, len(official))
            mamba = mamba + np.linspace(0.01, -0.01, len(mamba))
        for index in range(len(survival)):
            rows.append({
                "repeat": repeat,
                "fold": index % 2,
                "case_id": f"CASE-{index:02d}",
                "survival_days": survival[index],
                "event": events[index],
                "risk_official_pca_cox": official[index],
                "risk_attention": mamba[index],
                "risk_mamba": mamba[index],
            })
    return pd.DataFrame(rows)


class NestedSequenceEvaluationTests(unittest.TestCase):
    def test_repeat_summary_pools_each_patient_once(self):
        summary = summarize_repeats(_predictions())
        self.assertEqual(summary["n_cases"].tolist(), [8, 8])
        self.assertEqual(summary["n_events"].tolist(), [5, 5])
        self.assertTrue(np.isfinite(summary["cindex_mamba"]).all())

    def test_clustered_bootstrap_resamples_patients_across_repeats(self):
        result = clustered_patient_bootstrap(
            _predictions(),
            (("mamba", "attention"),),
            n_iter=100,
            seed=91,
        )
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result.loc[0, "mean_delta_across_repeats"], 0.0)
        self.assertAlmostEqual(result.loc[0, "ci95_lo"], 0.0)
        self.assertAlmostEqual(result.loc[0, "ci95_hi"], 0.0)

    def test_repeat_summary_rejects_duplicate_heldout_patient(self):
        frame = _predictions()
        frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "duplicate held-out"):
            summarize_repeats(frame)


if __name__ == "__main__":
    unittest.main()
