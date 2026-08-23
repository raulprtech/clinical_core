import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

CODE_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = CODE_ROOT / "tools"
for candidate in (CODE_ROOT, TOOLS_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from components.adapters.ingestion.vision.models.resnet_multiview import (  # noqa: E402
    VisionResNet18_2p5D,
    VisionResNet50_2p5D,
)
from build_resnet50_sequence_embeddings import valid_case_cache  # noqa: E402
from core.registry import get_vision_conn  # noqa: E402


class TinyBackbone(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        mean = inputs.mean(dim=(2, 3), keepdim=True)
        repeats = 2048 // mean.shape[1] + 1
        return mean.repeat(1, repeats, 1, 1)[:, :2048]


class ResNet50SequenceContractTests(unittest.TestCase):
    def test_resnet18_sequence_keeps_native_512_dimensions(self):
        model = VisionResNet18_2p5D(use_imagenet_weights=False, output_dim=768)
        raw = torch.randn(2, 512)
        projected = model._sequence_contract_projection(raw)
        self.assertEqual(tuple(projected.shape), (2, 512))

    def test_fixed_projection_is_reproducible_and_normalized(self):
        common = dict(
            output_dim=512,
            use_imagenet_weights=False,
            backbone=TinyBackbone(),
            feature_dim=2048,
            image_size=16,
            min_slices=2,
            projection_seed=2026,
        )
        first = VisionResNet50_2p5D(**common)
        second = VisionResNet50_2p5D(**common)
        raw = torch.arange(3 * 2048, dtype=torch.float32).reshape(3, 2048)
        left = first._sequence_contract_projection(raw)
        right = second._sequence_contract_projection(raw)
        self.assertEqual(tuple(left.shape), (3, 512))
        self.assertTrue(torch.equal(left, right))
        self.assertTrue(torch.allclose(left.norm(dim=1), torch.ones(3), atol=1e-6))

    def test_registry_exposes_resnet50_2p5d(self):
        model = get_vision_conn(
            "vision_resnet50_2p5d",
            use_imagenet_weights=False,
            backbone=TinyBackbone(),
            feature_dim=2048,
            output_dim=512,
        )
        self.assertIsInstance(model, VisionResNet50_2p5D)

    def test_radimagenet_normalization_maps_unit_interval_to_minus_one_one(self):
        model = VisionResNet50_2p5D(
            output_dim=512,
            use_imagenet_weights=False,
            backbone=TinyBackbone(),
            feature_dim=2048,
            image_size=4,
            min_slices=2,
            input_mean=(0.5, 0.5, 0.5),
            input_std=(0.5, 0.5, 0.5),
        )
        volume = np.stack((np.zeros((4, 4)), np.ones((4, 4))), axis=0).astype(
            np.float32
        )
        images, _ = model.volume_to_axial_sequence(volume, "MR", max_tokens=2)
        self.assertGreaterEqual(float(images.min()), -1.0)
        self.assertLessEqual(float(images.max()), 1.0)

    def test_cache_validation_binds_encoder_identity(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "case.npz"
            np.savez_compressed(
                path,
                features=np.ones((2, 512), dtype=np.float16),
                positions=np.array([0.0, 1.0], dtype=np.float32),
                encoder_id=np.asarray("expected"),
            )
            self.assertTrue(valid_case_cache(path, 64, 512, "expected"))
            self.assertFalse(valid_case_cache(path, 64, 512, "different"))


if __name__ == "__main__":
    unittest.main()
