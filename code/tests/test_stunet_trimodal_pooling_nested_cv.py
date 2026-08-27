import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


TOOLS_ROOT = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from evaluate_stunet_trimodal_pooling_nested_cv import repeat_metrics  # noqa: E402
from evaluate_trimodal_fusion import load_vision_csv  # noqa: E402


class STUNetTrimodalPoolingTests(unittest.TestCase):
    def test_load_vision_accepts_variable_z_dimension(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "moments.csv"
            frame = pd.DataFrame(
                {
                    "case_id": ["case-a", "case-b"],
                    **{
                        f"z{index:03d}": [float(index + 1), float(index + 2)]
                        for index in range(512)
                    },
                }
            )
            frame.to_csv(path, index=False)
            loaded = load_vision_csv(path)
        self.assertEqual(loaded.shape, (2, 512))
        self.assertEqual(loaded.index.tolist(), ["CASE-A", "CASE-B"])

    def test_repeat_metrics_pool_each_patient_once(self):
        rows = []
        for repeat in range(2):
            for index in range(10):
                base = 1.0 - index * 0.08 + repeat * 0.001
                rows.append(
                    {
                        "repeat": repeat,
                        "fold": index % 5,
                        "case_id": f"CASE-{index:02d}",
                        "survival_days": float(10 + index * 5),
                        "event": int(index % 3 != 0),
                        "risk_tabular": base,
                        "risk_text": base - 0.2,
                        "risk_vision_mean": base - 0.1,
                        "risk_vision_moments": base + 0.05,
                        "risk_fusion_mean": base + 0.01,
                        "risk_fusion_moments": base + 0.06,
                    }
                )
        metrics = repeat_metrics(pd.DataFrame(rows))
        self.assertEqual(metrics["n_cases"].tolist(), [10, 10])
        self.assertTrue(np.isfinite(metrics.filter(like="cindex_")).all().all())

    def test_repeat_metrics_reject_duplicate_patient(self):
        frame = pd.DataFrame(
            {
                "repeat": [0, 0],
                "case_id": ["A", "A"],
                "event": [1, 0],
                "survival_days": [1.0, 2.0],
            }
        )
        with self.assertRaisesRegex(ValueError, "duplicate OOF"):
            repeat_metrics(frame)


if __name__ == "__main__":
    unittest.main()
