import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from components.adapters.ingestion.vision.models.turboconv import (  # noqa: E402
    FakeQuantConv3d,
    QuantizationSpec,
    fwht,
    inverse_rotate_channels,
    rotate_channels,
)
from tools.evaluate_stunet_turboconv import write_quantization_verdict  # noqa: E402


class TurboConvTests(unittest.TestCase):
    def test_fwht_is_orthonormal(self):
        torch.manual_seed(1)
        values = torch.randn(2, 8, 3)
        transformed = fwht(values, dim=1)
        restored = fwht(transformed, dim=1)
        torch.testing.assert_close(restored, values, rtol=1e-5, atol=1e-6)

    def test_randomized_rotation_has_exact_inverse(self):
        values = torch.randn(2, 8, 2, 2, 2)
        signs = torch.tensor([1, -1, 1, 1, -1, 1, -1, -1], dtype=torch.float32)
        restored = inverse_rotate_channels(
            rotate_channels(values, signs, dim=1), signs, dim=1
        )
        torch.testing.assert_close(restored, values, rtol=1e-5, atol=1e-6)

    def test_turboconv_is_equivalent_before_quantization(self):
        torch.manual_seed(3)
        source = nn.Conv3d(8, 5, kernel_size=3, padding=1, bias=True).eval()
        wrapper = FakeQuantConv3d(
            source,
            "conv",
            QuantizationSpec("turboconv", 8, 8),
            activation_scale=1.0,
            quantize=False,
        ).eval()
        inputs = torch.randn(2, 8, 7, 6, 5)
        torch.testing.assert_close(
            wrapper(inputs), source(inputs), rtol=2e-5, atol=2e-6
        )

    def test_equal_bit_width_changes_outputs(self):
        torch.manual_seed(7)
        source = nn.Conv3d(8, 4, kernel_size=3, padding=1).eval()
        inputs = torch.randn(1, 8, 5, 5, 5)
        ptq = FakeQuantConv3d(
            source,
            "conv",
            QuantizationSpec("ptq", 4, 4),
            activation_scale=0.2,
        )
        turbo = FakeQuantConv3d(
            source,
            "conv",
            QuantizationSpec("turboconv", 4, 4),
            activation_scale=0.2,
        )
        self.assertFalse(torch.equal(ptq(inputs), source(inputs)))
        self.assertFalse(torch.equal(turbo(inputs), source(inputs)))

    def test_verdict_prefers_turbo_only_after_retention_and_paired_gain(self):
        paired = pd.DataFrame(
            [
                {"case_id": case_id, "variant": variant,
                 "embedding_cosine": 0.995 if variant == "turboconv" else 0.994,
                 "logit_relative_l2": 0.04 if variant == "turboconv" else 0.06,
                 "mask_dice_kidney_union": 0.97}
                for case_id in ("a", "b") for variant in ("ptq", "turboconv")
            ]
        )
        summary = pd.DataFrame(
            [
                {"variant": variant, "embedding_cosine_median": 0.994,
                 "logit_cosine_median": 0.998,
                 "mask_dice_kidney_union_median": 0.97,
                 "mask_dice_kidney_union_p05": 0.96}
                for variant in ("ptq", "turboconv")
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            result = write_quantization_verdict(paired, summary, Path(directory))
        self.assertEqual(result["recommendation"], "turboconv_numerically_preferred")

    def test_verdict_marks_single_variant_analysis_incomplete(self):
        paired = pd.DataFrame(
            [{"case_id": "a", "variant": "ptq", "embedding_cosine": 0.995,
              "logit_relative_l2": 0.04, "mask_dice_kidney_union": 0.97}]
        )
        summary = pd.DataFrame(
            [{"variant": "ptq", "embedding_cosine_median": 0.995,
              "logit_cosine_median": 0.998,
              "mask_dice_kidney_union_median": 0.97,
              "mask_dice_kidney_union_p05": 0.96}]
        )
        with tempfile.TemporaryDirectory() as directory:
            result = write_quantization_verdict(paired, summary, Path(directory))
        self.assertFalse(result["analysis_complete"])
        self.assertEqual(result["missing_variants"], ["turboconv"])
        self.assertEqual(result["recommendation"], "insufficient_paired_variants")


if __name__ == "__main__":
    unittest.main()
