"""Outcome-independent CT multi-window extension of frozen ResNet18 2.5D."""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .resnet_multiview import VisionResNet18_2p5D, VolumePath


DEFAULT_CT_WINDOWS: Tuple[Tuple[float, float], ...] = (
    (-150.0, 250.0),
    (-73.0, 304.0),
    (-200.0, 500.0),
)


class VisionResNet18_2p5DMultiWindow(VisionResNet18_2p5D):
    """Fuse frozen features from fixed CT windows without changing token size.

    Each axial 2.5D token retains the neighboring-slice RGB convention. CT is
    encoded once per fixed HU window; per-window 512D features are normalized,
    averaged with equal weights, and normalized again. MR follows the legacy
    percentile preprocessing in a single pass.
    """

    name = "vision_resnet18_2p5d_multiwindow"

    def __init__(
        self,
        ct_windows: Sequence[Sequence[float]] = DEFAULT_CT_WINDOWS,
        **kwargs: object,
    ):
        super().__init__(**kwargs)
        parsed = tuple((float(item[0]), float(item[1])) for item in ct_windows)
        if not parsed or any(high <= low for low, high in parsed):
            raise ValueError("ct_windows must contain ordered (low, high) pairs")
        self.ct_windows = parsed

    @staticmethod
    def _fixed_ct_window(plane: np.ndarray, bounds: Tuple[float, float]) -> np.ndarray:
        low, high = bounds
        plane = np.asarray(plane, dtype=np.float32)
        scaled = (np.clip(plane, low, high) - low) / (high - low)
        return scaled.astype(np.float32)

    def _volume_to_windowed_axial_sequence(
        self,
        volume: np.ndarray,
        modality: str,
        max_tokens: Optional[int],
        ct_window: Optional[Tuple[float, float]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        volume = np.asarray(volume, dtype=np.float32)
        indices = self._uniform_indices(int(volume.shape[0]), max_tokens)
        images = []
        for center in indices:
            channels = []
            for offset in self.slice_offsets:
                index = max(0, min(volume.shape[0] - 1, int(center) + offset))
                if modality.upper() == "CT":
                    if ct_window is None:
                        raise ValueError("CT multi-window pass requires explicit bounds")
                    channel = self._fixed_ct_window(volume[index], ct_window)
                else:
                    channel = self._window(volume[index], modality)
                channels.append(channel)
            image = torch.from_numpy(np.stack(channels, axis=0)).float()
            image = F.interpolate(
                image.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            images.append(image)
        batch = torch.stack(images)
        batch = (batch - self.imagenet_mean.cpu()) / self.imagenet_std.cpu()
        denominator = max(int(volume.shape[0]) - 1, 1)
        positions = torch.tensor(indices / denominator, dtype=torch.float32)
        return batch, positions

    def encode_axial_sequence(
        self,
        volume_path: VolumePath,
        max_tokens: Optional[int] = 64,
        inference_batch_size: int = 32,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, object]]:
        if inference_batch_size < 1:
            raise ValueError("inference_batch_size must be positive")
        volume, metadata = self.loader.load(volume_path)
        modality = str(metadata["modality"])
        windows: Tuple[Optional[Tuple[float, float]], ...] = (
            tuple(self.ct_windows) if modality.upper() == "CT" else (None,)
        )
        device = self._resolve_device()
        backbone = self._get_backbone().to(device).eval()
        per_window_features = []
        positions = None
        with torch.inference_mode():
            for bounds in windows:
                inputs, current_positions = self._volume_to_windowed_axial_sequence(
                    volume, modality, max_tokens, bounds
                )
                if positions is None:
                    positions = current_positions
                elif not torch.equal(positions, current_positions):
                    raise RuntimeError("Window passes produced inconsistent positions")
                chunks = []
                for start in range(0, len(inputs), int(inference_batch_size)):
                    encoded = backbone(
                        inputs[start : start + inference_batch_size].to(device)
                    )
                    chunks.append(encoded.flatten(1).cpu())
                window_features = torch.cat(chunks, dim=0)
                if window_features.shape[1] != self.feature_dim:
                    raise ValueError(
                        f"ResNet18 returned {window_features.shape[1]} features; "
                        f"expected {self.feature_dim}"
                    )
                per_window_features.append(
                    F.normalize(window_features, p=2, dim=1)
                )
        if positions is None:
            raise RuntimeError("No axial positions were generated")
        features = F.normalize(
            torch.stack(per_window_features).mean(dim=0), p=2, dim=1
        )
        sequence_metadata = dict(metadata)
        sequence_metadata.update({
            "original_slices": int(volume.shape[0]),
            "sequence_tokens": int(features.shape[0]),
            "feature_dim": int(features.shape[1]),
            "ct_windows_used": (
                [list(bounds) for bounds in self.ct_windows]
                if modality.upper() == "CT"
                else []
            ),
            "window_feature_fusion": "equal_mean_then_l2",
            "encoder_passes_per_token": len(windows),
        })
        return features, positions, sequence_metadata
