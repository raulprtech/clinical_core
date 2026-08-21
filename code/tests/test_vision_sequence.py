import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from components.adapters.ingestion.vision.models.resnet_multiview import (
    VisionResNet18_2p5D,
)
from components.adapters.ingestion.vision.models.sequence_pooling import (
    AttentionSequenceSurvival,
    MambaSequenceSurvival,
    cox_ph_loss,
)


class _SmallBackbone(nn.Module):
    def forward(self, batch):
        pooled = batch.mean(dim=(2, 3))
        return torch.cat((pooled, pooled[:, :1]), dim=1).unsqueeze(-1).unsqueeze(-1)


class VisionSequenceTests(unittest.TestCase):
    def test_axial_sequence_is_uniform_ordered_and_uses_neighbor_channels(self):
        volume = np.indices((10, 8, 9))[0].astype(np.float32)
        model = VisionResNet18_2p5D(
            use_imagenet_weights=False,
            image_size=12,
            window_low=0,
            window_high=9,
            backbone=_SmallBackbone(),
            feature_dim=4,
        )
        images, positions = model.volume_to_axial_sequence(volume, max_tokens=4)
        raw = images * model.imagenet_std + model.imagenet_mean
        self.assertEqual(tuple(images.shape), (4, 3, 12, 12))
        self.assertTrue(torch.all(positions[1:] > positions[:-1]))
        self.assertAlmostEqual(float(positions[0]), 0.0)
        self.assertAlmostEqual(float(positions[-1]), 1.0)
        self.assertFalse(torch.allclose(raw[1, 0], raw[1, 1]))

    def test_attention_ignores_padded_tokens(self):
        torch.manual_seed(3)
        model = AttentionSequenceSurvival(
            input_dim=8, model_dim=12, attention_dim=6, dropout=0
        ).eval()
        features = torch.randn(2, 5, 8)
        positions = torch.linspace(0, 1, 5).repeat(2, 1)
        mask = torch.tensor([
            [True, True, True, False, False],
            [True, True, True, True, True],
        ])
        first, weights = model(features, positions, mask, return_attention=True)
        features[0, 3:] = 1000
        second = model(features, positions, mask)
        self.assertTrue(torch.allclose(first, second, atol=1e-6))
        self.assertEqual(float(weights[0, 3:].sum().detach()), 0.0)

    def test_mamba_is_order_sensitive_beyond_attention_pooling(self):
        torch.manual_seed(7)
        attention = AttentionSequenceSurvival(
            input_dim=8, model_dim=12, attention_dim=6, dropout=0
        ).eval()
        mamba = MambaSequenceSurvival(
            input_dim=8,
            model_dim=12,
            attention_dim=6,
            state_dim=4,
            n_blocks=1,
            dropout=0,
        ).eval()
        features = torch.randn(2, 5, 8)
        positions = torch.linspace(0, 1, 5).repeat(2, 1)
        mask = torch.ones(2, 5, dtype=torch.bool)
        reverse = torch.arange(4, -1, -1)
        attention_first = attention(features, positions, mask)
        attention_reversed = attention(
            features[:, reverse], positions[:, reverse], mask[:, reverse]
        )
        mamba_first = mamba(features, positions, mask)
        mamba_reversed = mamba(
            features[:, reverse], positions[:, reverse], mask[:, reverse]
        )
        self.assertTrue(torch.allclose(attention_first, attention_reversed, atol=1e-6))
        self.assertFalse(torch.allclose(mamba_first, mamba_reversed, atol=1e-5))

    def test_cox_loss_is_finite_and_differentiable(self):
        risk = torch.tensor([0.2, -0.1, 0.5, 0.0], requires_grad=True)
        loss = cox_ph_loss(
            risk,
            torch.tensor([10.0, 8.0, 6.0, 4.0]),
            torch.tensor([1.0, 0.0, 1.0, 1.0]),
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(risk.grad).all())


if __name__ == "__main__":
    unittest.main()
