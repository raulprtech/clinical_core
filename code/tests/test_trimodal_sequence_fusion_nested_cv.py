import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


TOOLS_ROOT = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from evaluate_trimodal_sequence_fusion_nested_cv import repeat_metrics  # noqa: E402


def _predictions() -> pd.DataFrame:
    rows = []
    for repeat in range(2):
        for index in range(8):
            base = 0.8 - index * 0.08 + repeat * 0.005
            rows.append({
                "repeat": repeat,
                "fold": index % 2,
                "case_id": f"CASE-{index:02d}",
                "survival_days": float(5 + index * 3),
                "event": int(index % 3 != 1),
                "risk_tabular": base,
                "risk_vision_resnet": base - 0.1,
                "risk_vision_mamba": base + 0.02,
                "risk_fusion_resnet": base + 0.01,
                "risk_fusion_mamba": base + 0.03,
            })
    return pd.DataFrame(rows)


class TrimodalNestedEvaluationTests(unittest.TestCase):
    def test_repeat_metrics_pool_each_patient_once(self):
        metrics = repeat_metrics(_predictions())
        self.assertEqual(metrics["n_cases"].tolist(), [8, 8])
        self.assertTrue(np.isfinite(metrics.filter(like="cindex_")).all().all())

    def test_repeat_metrics_reject_duplicate_patient(self):
        frame = _predictions()
        frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "duplicate OOF"):
            repeat_metrics(frame)


if __name__ == "__main__":
    unittest.main()
