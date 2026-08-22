import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


TOOLS_ROOT = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from evaluate_fixed_sequence_ensemble import (  # noqa: E402
    add_fixed_ensembles,
    per_fold_metrics,
)


class FixedSequenceEnsembleTests(unittest.TestCase):
    def make_predictions(self):
        rows = []
        for repeat in range(2):
            for fold in range(2):
                for index in range(4):
                    rows.append({
                        "repeat": repeat,
                        "fold": fold,
                        "case_id": f"CASE-{fold}-{index}",
                        "survival_days": float(20 - index - 4 * fold),
                        "event": int(index != 1),
                        "modality": "CT" if index < 3 else "MR",
                        "risk_mamba_t64_posoff": float(index + fold),
                        "risk_attention_t32_posoff": float(10 - index + fold),
                    })
        return pd.DataFrame(rows)

    def test_rank_ensemble_is_invariant_to_positive_affine_scale(self):
        frame = self.make_predictions()
        first = add_fixed_ensembles(frame)["risk_ensemble_rank50"]
        frame["risk_mamba_t64_posoff"] = (
            17.0 * frame["risk_mamba_t64_posoff"] - 30.0
        )
        second = add_fixed_ensembles(frame)["risk_ensemble_rank50"]
        self.assertTrue(np.array_equal(first.to_numpy(), second.to_numpy()))

    def test_models_contribute_equal_rank_weight(self):
        output = add_fixed_ensembles(self.make_predictions())
        groups = output.groupby(["repeat", "fold"])
        expected = 0.5 * (
            groups["risk_mamba_t64_posoff"].rank(pct=True)
            + groups["risk_attention_t32_posoff"].rank(pct=True)
        )
        self.assertTrue(np.allclose(output["risk_ensemble_rank50"], expected))

    def test_fold_metrics_include_ct_and_mr(self):
        metrics = per_fold_metrics(add_fixed_ensembles(self.make_predictions()))
        self.assertEqual(set(metrics["n_ct"]), {3})
        self.assertEqual(set(metrics["n_mr"]), {1})


if __name__ == "__main__":
    unittest.main()
