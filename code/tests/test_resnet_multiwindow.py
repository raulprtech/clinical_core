import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from components.adapters.ingestion.vision.models.resnet_multiwindow import (  # noqa: E402
    VisionResNet18_2p5DMultiWindow,
)
from components.adapters.ingestion.vision.models.resnet_multiview import (  # noqa: E402
    VisionResNet18_2p5D,
)


class _WindowSensitiveBackbone(nn.Module):
    def forward(self, batch):
        means = batch.mean(dim=(2, 3))
        spread = batch.amax(dim=(2, 3))[:, :1] - batch.amin(dim=(2, 3))[:, :1]
        return torch.cat((means, spread), dim=1).unsqueeze(-1).unsqueeze(-1)


def _common():
    return dict(
        use_imagenet_weights=False,
        image_size=12,
        window_low=-150,
        window_high=250,
        backbone=_WindowSensitiveBackbone(),
        feature_dim=4,
        min_slices=2,
        device="cpu",
    )


class ResNetMultiWindowTests(unittest.TestCase):
    def setUp(self):
        grid = np.indices((8, 9, 10)).sum(axis=0).astype(np.float32)
        self.volume = grid * 60.0 - 300.0

    def _loader(self, modality):
        return lambda _: (
            self.volume,
            {"modality": modality, "source_format": "test", "n_slices": 8},
        )

    def test_single_window_reproduces_legacy_features(self):
        legacy = VisionResNet18_2p5D(**_common())
        explicit = VisionResNet18_2p5DMultiWindow(
            ct_windows=((-150, 250),), **_common()
        )
        legacy.loader.load = self._loader("CT")
        explicit.loader.load = self._loader("CT")
        first, positions_first, _ = legacy.encode_axial_sequence("unused", max_tokens=4)
        second, positions_second, _ = explicit.encode_axial_sequence(
            "unused", max_tokens=4
        )
        self.assertTrue(torch.equal(positions_first, positions_second))
        self.assertTrue(torch.allclose(first, second, atol=1e-7))

    def test_ct_multiwindow_preserves_shape_and_changes_features(self):
        baseline = VisionResNet18_2p5D(**_common())
        multi = VisionResNet18_2p5DMultiWindow(**_common())
        baseline.loader.load = self._loader("CT")
        multi.loader.load = self._loader("CT")
        original, _, _ = baseline.encode_axial_sequence("unused", max_tokens=4)
        features, positions, metadata = multi.encode_axial_sequence(
            "unused", max_tokens=4
        )
        self.assertEqual(tuple(features.shape), (4, 4))
        self.assertEqual(tuple(positions.shape), (4,))
        self.assertTrue(torch.allclose(features.norm(dim=1), torch.ones(4)))
        self.assertFalse(torch.allclose(features, original))
        self.assertEqual(metadata["encoder_passes_per_token"], 3)
        self.assertEqual(metadata["window_feature_fusion"], "equal_mean_then_l2")

    def test_mr_is_unchanged(self):
        legacy = VisionResNet18_2p5D(**_common())
        multi = VisionResNet18_2p5DMultiWindow(**_common())
        legacy.loader.load = self._loader("MR")
        multi.loader.load = self._loader("MR")
        first, _, _ = legacy.encode_axial_sequence("unused", max_tokens=4)
        second, _, metadata = multi.encode_axial_sequence("unused", max_tokens=4)
        self.assertTrue(torch.allclose(first, second, atol=1e-7))
        self.assertEqual(metadata["encoder_passes_per_token"], 1)
        self.assertEqual(metadata["ct_windows_used"], [])

    def test_invalid_window_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "ordered"):
            VisionResNet18_2p5DMultiWindow(ct_windows=((100, 100),), **_common())


if __name__ == "__main__":
    unittest.main()
