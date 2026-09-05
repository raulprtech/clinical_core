"""Frozen multi-view ResNet encoders derived from the VISION-L0 notebooks.

The module deliberately contains no survival fitting.  It turns one DICOM
series (or a NIfTI volume) into the 768-dimensional, L2-normalized VISION-IN
embedding consumed by the multimodal pipeline.  This keeps PCA/Cox fitting in
the evaluation split and prevents outcome leakage into the shared embedding.
"""

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


VolumePath = Union[str, Path]
VIEW_ORDER = ("axial", "coronal", "sagittal")


class MedicalVolumeLoader:
    """Load a volume in canonical ``(z, y, x)`` array order."""

    def __init__(self, min_slices: int = 16, default_modality: str = "CT"):
        self.min_slices = int(min_slices)
        self.default_modality = str(default_modality).upper()

    def load(self, path: VolumePath) -> Tuple[np.ndarray, Dict[str, object]]:
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(f"Vision input does not exist: {source}")
        if source.is_dir() or source.suffix.lower() == ".dcm":
            return self._load_dicom(source if source.is_dir() else source.parent)
        if source.suffix.lower() == ".nii" or source.name.lower().endswith(".nii.gz"):
            return self._load_nifti(source)
        raise ValueError(
            f"Unsupported vision input {source}; expected a DICOM directory or NIfTI volume"
        )

    def _load_dicom(self, series_dir: Path) -> Tuple[np.ndarray, Dict[str, object]]:
        try:
            import pydicom
        except ImportError as exc:
            raise ImportError("DICOM input requires the 'pydicom' package") from exc

        files = sorted(series_dir.rglob("*.dcm"))
        if not files:
            files = sorted(p for p in series_dir.rglob("*") if p.is_file())
        slices = []
        modality = self.default_modality
        for file_path in files:
            try:
                ds = pydicom.dcmread(str(file_path), force=True)
                if not hasattr(ds, "PixelData"):
                    continue
                pixels = ds.pixel_array.astype(np.float32)
                pixels = pixels * float(getattr(ds, "RescaleSlope", 1.0))
                pixels = pixels + float(getattr(ds, "RescaleIntercept", 0.0))
                if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
                    pixels = pixels.max() - pixels
                position = getattr(ds, "ImagePositionPatient", None)
                key = float(position[2]) if position is not None else float(
                    getattr(ds, "InstanceNumber", 0)
                )
                modality = str(getattr(ds, "Modality", modality)).upper()
                slices.append((key, pixels))
            except Exception:
                continue
        readable_slices = len(slices)
        shape_groups: Dict[Tuple[int, ...], list] = {}
        for item in slices:
            shape_groups.setdefault(tuple(item[1].shape), []).append(item)
        if shape_groups:
            # Some TCIA series contain a few scouts/reconstructions with a
            # different matrix. Keep the dominant coherent stack.
            slices = max(
                shape_groups.values(),
                key=lambda group: (len(group), int(np.prod(group[0][1].shape))),
            )
        if len(slices) < self.min_slices:
            raise ValueError(
                f"DICOM series has {len(slices)} shape-consistent readable slices; "
                f"minimum is {self.min_slices}"
            )
        slices.sort(key=lambda item: item[0])
        volume = np.stack([item[1] for item in slices], axis=0)
        return self._validate(volume), {
            "source_format": "dicom",
            "modality": modality,
            "n_slices": len(slices),
            "dropped_mixed_shape_slices": readable_slices - len(slices),
        }

    def _load_nifti(self, path: Path) -> Tuple[np.ndarray, Dict[str, object]]:
        try:
            import nibabel as nib
        except ImportError as exc:
            raise ImportError("NIfTI input requires the 'nibabel' package") from exc
        image = nib.as_closest_canonical(nib.load(str(path)))
        data = np.asarray(image.get_fdata(dtype=np.float32)).squeeze()
        if data.ndim != 3:
            raise ValueError(f"Expected a 3D NIfTI volume, got shape {data.shape}")
        # nibabel canonical order is (x, y, z); notebooks operate on (z, y, x).
        volume = np.transpose(data, (2, 1, 0))
        return self._validate(volume), {
            "source_format": "nifti",
            "modality": self.default_modality,
            "n_slices": int(volume.shape[0]),
        }

    @staticmethod
    def _validate(volume: np.ndarray) -> np.ndarray:
        volume = np.asarray(volume, dtype=np.float32)
        if volume.ndim != 3 or min(volume.shape) < 2:
            raise ValueError(f"Expected a non-degenerate 3D volume, got {volume.shape}")
        if not np.isfinite(volume).any():
            raise ValueError("Volume contains no finite voxels")
        return np.nan_to_num(volume, copy=False)


class FrozenResNetMultiView(nn.Module):
    """Common implementation for ResNet18/50 2D and ResNet18 2.5D."""

    output_dim = 768

    def __init__(
        self,
        backbone_name: str,
        context: str,
        output_dim: int = 768,
        use_imagenet_weights: bool = True,
        image_size: int = 224,
        window_low: float = -150.0,
        window_high: float = 250.0,
        slice_offsets: Sequence[int] = (-1, 0, 1),
        aggregation: str = "mean3",
        projection_seed: int = 2026,
        min_slices: int = 16,
        default_modality: str = "CT",
        device: str = "auto",
        weights_dir: Optional[VolumePath] = None,
        backbone: Optional[nn.Module] = None,
        feature_dim: Optional[int] = None,
        input_mean: Sequence[float] = (0.485, 0.456, 0.406),
        input_std: Sequence[float] = (0.229, 0.224, 0.225),
        **_: object,
    ):
        super().__init__()
        if backbone_name not in {"resnet18", "resnet50"}:
            raise ValueError("backbone_name must be 'resnet18' or 'resnet50'")
        if context not in {"2d", "2p5d"}:
            raise ValueError("context must be '2d' or '2p5d'")
        if aggregation != "mean3":
            raise ValueError("Only the notebook's leakage-safe 'mean3' aggregation is supported")
        if context == "2p5d" and len(tuple(slice_offsets)) != 3:
            raise ValueError("2.5D encoding requires exactly three slice offsets")

        self.backbone_name = backbone_name
        self.context = context
        self.output_dim = int(output_dim)
        self.use_imagenet_weights = bool(use_imagenet_weights)
        self.image_size = int(image_size)
        self.window_low = float(window_low)
        self.window_high = float(window_high)
        self.slice_offsets = tuple(int(v) for v in slice_offsets)
        self.aggregation = aggregation
        self.projection_seed = int(projection_seed)
        self.loader = MedicalVolumeLoader(min_slices, default_modality)
        self._device_spec = device
        self.weights_dir = Path(weights_dir) if weights_dir else None
        self._backbone = backbone
        self.feature_dim = int(feature_dim or (512 if backbone_name == "resnet18" else 2048))
        input_mean = tuple(float(value) for value in input_mean)
        input_std = tuple(float(value) for value in input_std)
        if len(input_mean) != 3 or len(input_std) != 3 or min(input_std) <= 0:
            raise ValueError("input_mean/input_std must contain three values and positive stds")
        self.register_buffer(
            "imagenet_mean", torch.tensor(input_mean).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "imagenet_std", torch.tensor(input_std).view(1, 3, 1, 1)
        )
        if self.feature_dim > self.output_dim:
            # Match VISION-L0.5 exactly: NumPy PCG64, data-oblivious Gaussian
            # projection with scale 1/sqrt(output_dim).
            rng = np.random.default_rng(self.projection_seed)
            projection = torch.from_numpy(rng.normal(
                loc=0.0,
                scale=1.0 / np.sqrt(self.output_dim),
                size=(self.feature_dim, self.output_dim),
            ).astype(np.float32))
            self.register_buffer("fixed_projection", projection)
        else:
            self.register_buffer("fixed_projection", torch.empty(0))

    @property
    def name(self) -> str:
        return f"vision_{self.backbone_name}_{self.context}"

    def _resolve_device(self) -> torch.device:
        if self._device_spec == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self._device_spec)

    def _get_backbone(self) -> nn.Module:
        if self._backbone is not None:
            return self._backbone
        from torchvision import models

        if self.weights_dir is not None:
            self.weights_dir.mkdir(parents=True, exist_ok=True)
            torch.hub.set_dir(str(self.weights_dir))

        if self.backbone_name == "resnet18":
            weights = (
                models.ResNet18_Weights.IMAGENET1K_V1
                if self.use_imagenet_weights else None
            )
            model = models.resnet18(weights=weights)
        else:
            weights = (
                models.ResNet50_Weights.IMAGENET1K_V2
                if self.use_imagenet_weights else None
            )
            model = models.resnet50(weights=weights)
        self._backbone = nn.Sequential(*list(model.children())[:-1]).eval()
        for parameter in self._backbone.parameters():
            parameter.requires_grad = False
        return self._backbone

    @staticmethod
    def _plane(volume: np.ndarray, view: str, index: int) -> np.ndarray:
        if view == "axial":
            return volume[index, :, :]
        if view == "coronal":
            return volume[:, index, :]
        if view == "sagittal":
            return volume[:, :, index]
        raise ValueError(f"Unknown view: {view}")

    def _window(self, plane: np.ndarray, modality: str) -> np.ndarray:
        plane = np.asarray(plane, dtype=np.float32)
        if modality.upper() == "CT":
            low, high = self.window_low, self.window_high
        else:
            low, high = np.percentile(plane, [1, 99])
            if high <= low:
                low, high = float(plane.min()), float(plane.max())
        scaled = (np.clip(plane, low, high) - low) / max(float(high - low), 1e-6)
        return scaled.astype(np.float32)

    def volume_to_views(self, volume: np.ndarray, modality: str = "CT") -> torch.Tensor:
        """Return the three notebook-compatible views as ``[3, 3, H, W]``."""
        centers = dict(zip(VIEW_ORDER, (s // 2 for s in volume.shape)))
        images = []
        for view in VIEW_ORDER:
            center = centers[view]
            if self.context == "2d":
                channel = self._window(self._plane(volume, view, center), modality)
                image = np.stack([channel, channel, channel], axis=0)
            else:
                axis_size = volume.shape[VIEW_ORDER.index(view)]
                channels = []
                for offset in self.slice_offsets:
                    index = max(0, min(axis_size - 1, center + offset))
                    channels.append(self._window(self._plane(volume, view, index), modality))
                image = np.stack(channels, axis=0)
            resized = F.interpolate(
                torch.from_numpy(image).float().unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            images.append(resized)
        batch = torch.stack(images)
        return (batch - self.imagenet_mean.cpu()) / self.imagenet_std.cpu()

    @staticmethod
    def _uniform_indices(length: int, max_tokens: Optional[int]) -> np.ndarray:
        """Return ordered indices without repeating slices.

        Sequence models consume a bounded number of tokens so the cache and
        training budget do not depend on scanner slice thickness. Endpoints
        are retained and the sampling rule is entirely outcome-independent.
        """
        if length < 1:
            raise ValueError("A volume sequence must contain at least one slice")
        if max_tokens is None or int(max_tokens) >= length:
            return np.arange(length, dtype=np.int64)
        if int(max_tokens) < 2:
            raise ValueError("max_tokens must be at least 2")
        return np.linspace(0, length - 1, num=int(max_tokens), dtype=np.int64)

    def volume_to_axial_sequence(
        self,
        volume: np.ndarray,
        modality: str = "CT",
        max_tokens: Optional[int] = 64,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create ordered 2.5D axial windows and relative positions.

        Returns images [T, 3, H, W] and positions in [0, 1]. Unlike
        volume_to_views, this supports patient-level sequence pooling rather
        than the historical three-central-view baseline.
        """
        if self.context != "2p5d":
            raise ValueError("Axial sequences require a 2.5D encoder")
        volume = np.asarray(volume, dtype=np.float32)
        indices = self._uniform_indices(int(volume.shape[0]), max_tokens)
        images = []
        for center in indices:
            channels = []
            for offset in self.slice_offsets:
                index = max(0, min(volume.shape[0] - 1, int(center) + offset))
                channels.append(self._window(volume[index], modality))
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

    def _contract_projection(self, features: torch.Tensor) -> torch.Tensor:
        if features.numel() != self.feature_dim:
            raise ValueError(
                f"{self.backbone_name} returned {features.numel()} features; "
                f"expected {self.feature_dim}"
            )
        if self.feature_dim < self.output_dim:
            embedding = F.pad(features, (0, self.output_dim - self.feature_dim))
        elif self.feature_dim > self.output_dim:
            embedding = features @ self.fixed_projection.to(features.device)
        else:
            embedding = features
        return F.normalize(embedding, p=2, dim=0).cpu()

    def _sequence_contract_projection(self, features: torch.Tensor) -> torch.Tensor:
        """Project wide tokens while preserving narrower native token features."""
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError(
                f"{self.backbone_name} returned shape {tuple(features.shape)}; "
                f"expected [tokens, {self.feature_dim}]"
            )
        # Historical ResNet18 sequence caches retain their native 512D output
        # even though patient-level embeddings are padded to 768D. ResNet50 is
        # wider than the requested contract and therefore receives the fixed
        # data-oblivious projection.
        if self.feature_dim > self.output_dim:
            projected = features @ self.fixed_projection.to(features.device)
        else:
            projected = features
        return F.normalize(projected, p=2, dim=1).cpu()

    def encode(self, volume_path: VolumePath) -> Tuple[torch.Tensor, float]:
        volume, metadata = self.loader.load(volume_path)
        inputs = self.volume_to_views(volume, str(metadata["modality"]))
        device = self._resolve_device()
        backbone = self._get_backbone().to(device).eval()
        with torch.inference_mode():
            view_features = backbone(inputs.to(device)).flatten(1)
            patient_features = view_features.mean(dim=0)
            embedding = self._contract_projection(patient_features)
        return embedding, 1.0

    def encode_axial_sequence(
        self,
        volume_path: VolumePath,
        max_tokens: Optional[int] = 64,
        inference_batch_size: int = 32,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, object]]:
        """Encode ordered axial 2.5D windows with the frozen backbone.

        Features wider than ``output_dim`` use the same fixed,
        outcome-independent projection as the patient-level contract. Native
        features narrower than that contract are preserved without padding.
        No outcomes are read or used by this operation.
        """
        if self.context != "2p5d":
            raise ValueError("Sequence encoding requires a 2.5D encoder")
        if inference_batch_size < 1:
            raise ValueError("inference_batch_size must be positive")
        volume, metadata = self.loader.load(volume_path)
        inputs, positions = self.volume_to_axial_sequence(
            volume, str(metadata["modality"]), max_tokens=max_tokens
        )
        device = self._resolve_device()
        backbone = self._get_backbone().to(device).eval()
        feature_chunks = []
        with torch.inference_mode():
            for start in range(0, len(inputs), int(inference_batch_size)):
                chunk = backbone(inputs[start : start + inference_batch_size].to(device))
                feature_chunks.append(chunk.flatten(1).cpu())
        features = torch.cat(feature_chunks, dim=0)
        if features.shape[1] != self.feature_dim:
            raise ValueError(
                f"{self.backbone_name} returned {features.shape[1]} features; "
                f"expected {self.feature_dim}"
            )
        features = self._sequence_contract_projection(features)
        sequence_metadata = dict(metadata)
        sequence_metadata.update({
            "original_slices": int(volume.shape[0]),
            "sequence_tokens": int(features.shape[0]),
            "feature_dim": int(features.shape[1]),
        })
        return features, positions, sequence_metadata

    def encode_batch(self, paths: Sequence[VolumePath]) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = [self.encode(path) for path in paths]
        return (
            torch.stack([item[0] for item in outputs]),
            torch.tensor([item[1] for item in outputs], dtype=torch.float32),
        )


class VisionResNet18_2D(FrozenResNetMultiView):
    name = "vision_resnet18_2d"

    def __init__(self, **kwargs: object):
        super().__init__(backbone_name="resnet18", context="2d", **kwargs)


class VisionResNet50_2D(FrozenResNetMultiView):
    name = "vision_resnet50_2d"

    def __init__(self, **kwargs: object):
        super().__init__(backbone_name="resnet50", context="2d", **kwargs)


class VisionResNet50_2p5D(FrozenResNetMultiView):
    name = "vision_resnet50_2p5d"

    def __init__(self, **kwargs: object):
        super().__init__(backbone_name="resnet50", context="2p5d", **kwargs)


class VisionResNet18_2p5D(FrozenResNetMultiView):
    name = "vision_resnet18_2p5d"

    def __init__(self, **kwargs: object):
        super().__init__(backbone_name="resnet18", context="2p5d", **kwargs)


def load_precomputed_embeddings(
    csv_path: VolumePath, output_dim: int = 768
) -> Dict[str, Tuple[torch.Tensor, float]]:
    """Load the contractual CSV exported by any of the three notebooks."""
    import pandas as pd

    path = Path(csv_path)
    frame = pd.read_csv(path)
    if "case_id" not in frame:
        raise ValueError(f"Embedding CSV has no case_id column: {path}")
    columns = [f"z{i:03d}" for i in range(output_dim)]
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(
            f"Embedding CSV is missing {len(missing)} contractual z-columns: {path}"
        )
    result = {}
    for _, row in frame.iterrows():
        case_id = str(row["case_id"]).strip().upper()
        values = torch.tensor(row[columns].to_numpy(dtype=np.float32))
        if not torch.isfinite(values).all() or float(values.norm()) == 0.0:
            continue
        values = F.normalize(values, p=2, dim=0)
        confidence = float(row.get("vision_confidence", 1.0))
        if not np.isfinite(confidence):
            confidence = 0.0
        result[case_id] = (values, min(1.0, max(0.0, confidence)))
    return result
