import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


TOOLS_ROOT = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from evaluate_sequence_factorial_ablation import (  # noqa: E402
    cap_sequence_tokens,
    configuration_name,
    pooled_repeat_metrics,
)


class SequenceFactorialAblationTests(unittest.TestCase):
    def test_token_cap_is_uniform_and_keeps_endpoints(self):
        features = np.arange(10 * 3, dtype=np.float32).reshape(10, 3)
        positions = np.linspace(0, 1, 10, dtype=np.float32)
        capped = cap_sequence_tokens({"CASE": (features, positions)}, 4)["CASE"]
        self.assertEqual(capped[0].shape, (4, 3))
        self.assertAlmostEqual(float(capped[1][0]), 0.0)
        self.assertAlmostEqual(float(capped[1][-1]), 1.0)
        self.assertTrue(np.all(np.diff(capped[1]) > 0))

    def test_configuration_names_are_stable(self):
        self.assertEqual(
            configuration_name("mamba", 32, False), "mamba_t32_posoff"
        )
        self.assertEqual(
            configuration_name("attention", 64, True), "attention_t64_poson"
        )

    def test_pooled_metrics_report_ct_and_mr(self):
        name = "mamba_t32_posoff"
        rows = []
        for repeat in range(2):
            for index in range(8):
                rows.append({
                    "repeat": repeat,
                    "case_id": f"CASE-{index}",
                    "survival_days": 5.0 + index,
                    "event": int(index % 3 != 1),
                    "modality": "CT" if index < 5 else "MR",
                    f"risk_{name}": 1.0 - index / 10.0,
                })
        metrics = pooled_repeat_metrics(pd.DataFrame(rows), [name])
        self.assertEqual(metrics["n_ct"].tolist(), [5, 5])
        self.assertEqual(metrics["n_mr"].tolist(), [3, 3])
        self.assertTrue(np.isfinite(metrics["cindex_all"]).all())


if __name__ == "__main__":
    unittest.main()
