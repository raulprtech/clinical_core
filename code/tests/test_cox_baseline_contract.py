import sys
import unittest
from pathlib import Path

import numpy as np
import torch


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from components.adapters.ingestion.tabular.models.cox_baseline import (  # noqa: E402
    VariantA_CoxBaseline,
)
from core.model_utils import verify_ingestion_contract  # noqa: E402


class CoxBaselineContractTests(unittest.TestCase):
    def test_projection_is_768d_unit_norm_including_zero_rows(self):
        model = VariantA_CoxBaseline(input_dim=3, output_dim=768)
        features = np.array(
            [[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        confidence = np.ones((2, 3), dtype=np.float32)
        embedding, confidence_tensor = model.encode(features, confidence)
        result = verify_ingestion_contract(
            embedding,
            confidence_tensor,
            expected_dim=768,
            verbose=False,
        )
        self.assertTrue(result["contract_satisfied"])
        torch.testing.assert_close(
            embedding.norm(dim=-1),
            torch.ones(2),
        )


if __name__ == "__main__":
    unittest.main()
