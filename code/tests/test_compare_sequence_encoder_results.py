import sys
import unittest
from pathlib import Path

import pandas as pd

CODE_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = CODE_ROOT / "tools"
for candidate in (CODE_ROOT, TOOLS_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from compare_sequence_encoder_results import (  # noqa: E402
    SOURCE_MODELS,
    paired_predictions,
    safe_label,
)


def predictions(offset: float = 0.0) -> pd.DataFrame:
    rows = []
    for case_index, case_id in enumerate(("A", "B")):
        row = {
            "repeat": 0,
            "fold": case_index,
            "case_id": case_id,
            "survival_days": float(10 + case_index),
            "event": case_index,
            "modality": "CT" if case_index == 0 else "MR",
        }
        row.update({
            f"risk_{name}": case_index + offset + model_index / 10
            for model_index, name in enumerate(SOURCE_MODELS)
        })
        rows.append(row)
    return pd.DataFrame(rows)


class CompareSequenceEncoderResultsTests(unittest.TestCase):
    def test_labels_are_safe_and_stable(self):
        self.assertEqual(safe_label("ResNet50 RadImageNet"), "resnet50_radimagenet")
        with self.assertRaises(ValueError):
            safe_label("---")

    def test_pairing_preserves_outcomes_and_names_risks(self):
        result = paired_predictions(
            predictions(), predictions(1.0), "imagenet50", "radimagenet50"
        )
        self.assertEqual(result["case_id"].tolist(), ["A", "B"])
        self.assertIn("risk_imagenet50_mamba", result)
        self.assertIn("risk_radimagenet50_mamba", result)
        self.assertEqual(result.loc[0, "risk_radimagenet50_mamba"], 1.0)

    def test_pairing_rejects_outcome_disagreement(self):
        candidate = predictions(1.0)
        candidate.loc[0, "event"] = 1
        with self.assertRaisesRegex(ValueError, "disagree on event"):
            paired_predictions(predictions(), candidate, "left", "right")


if __name__ == "__main__":
    unittest.main()
