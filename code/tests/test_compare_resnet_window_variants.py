import sys
import unittest
from pathlib import Path

import pandas as pd

TOOLS_ROOT = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from compare_resnet_window_variants import (  # noqa: E402
    SOURCE_MODELS,
    paired_predictions,
)


def _frame(risk_offset=0.0):
    rows = []
    for case_id, event in (("A", 1), ("B", 0)):
        row = {
            "repeat": 0,
            "fold": 0,
            "case_id": case_id,
            "survival_days": 10.0 if case_id == "A" else 20.0,
            "event": event,
            "modality": "CT",
        }
        row.update({f"risk_{name}": risk_offset + event for name in SOURCE_MODELS})
        rows.append(row)
    return pd.DataFrame(rows)


class CompareResNetWindowVariantsTests(unittest.TestCase):
    def test_pairing_requires_and_preserves_exact_rows(self):
        paired = paired_predictions(_frame(), _frame(0.2))
        self.assertEqual(len(paired), 2)
        self.assertIn("risk_single_mamba", paired)
        self.assertIn("risk_multiwindow_mamba", paired)
        self.assertEqual(paired["event"].tolist(), [1, 0])

    def test_pairing_rejects_endpoint_mismatch(self):
        candidate = _frame(0.2)
        candidate.loc[0, "event"] = 0
        with self.assertRaisesRegex(ValueError, "event"):
            paired_predictions(_frame(), candidate)

    def test_pairing_rejects_missing_patient(self):
        with self.assertRaisesRegex(ValueError, "identical"):
            paired_predictions(_frame(), _frame(0.2).iloc[:1])


if __name__ == "__main__":
    unittest.main()
