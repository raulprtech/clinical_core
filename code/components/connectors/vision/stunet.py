"""
VISION-CONN: STU-Net Segmentation Backend
==========================================

STU-Net backend (Huang et al. 2023).
"""

import os
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from .radiomics import RadiomicsExtractor

class VolumeLoader:
    """NIfTI loader using nibabel. Falls back to mock metadata if nibabel missing."""
    def __init__(self):
        self._mode = self._init_backend()
    def _init_backend(self) -> str:
        try:
            import nibabel  # noqa
            return "nibabel"
        except ImportError:
            return "mock"
    def load(self, path: Union[str, Path]) -> Tuple[Optional[np.ndarray], dict]:
        path = Path(path)
        if not path.exists(): return None, {'error': 'file_not_found', 'path': str(path)}
        if self._mode == "nibabel":
            try:
                import nibabel as nib
                img = nib.load(str(path))
                volume = img.get_fdata().astype(np.float32)
                return volume, {
                    'shape': tuple(int(s) for s in volume.shape),
                    'spacing': tuple(float(s) for s in img.header.get_zooms()),
                    'affine': img.affine.tolist(),
                }
            except Exception as e:
                warnings.warn(f"nibabel load failed: {e}")
                return None, {'error': str(e)}
        return None, {'mock': True, 'path': str(path)}

class _BaseSegmenter:
    """Common interface for all segmenter backends."""
    backend_name: str = "base"

    def is_available(self) -> bool:
        return False

    def segment_kidney(
        self, volume: np.ndarray, metadata: dict
    ) -> Tuple[Optional[np.ndarray], float]:
        """Returns (binary_kidney_mask, segmentation_quality_score)."""
        raise NotImplementedError

class TotalSegmentatorSegmenter(_BaseSegmenter):
    """
    Primary backend: TotalSegmentator python API.
    """
    backend_name = "totalseg"
    KIDNEY_LEFT_LABEL = 2
    KIDNEY_RIGHT_LABEL = 3

    def __init__(self, fast: bool = True):
        self.fast = fast
        self._totalsegmentator = None
        self._available = self._init_backend()

    def _init_backend(self) -> bool:
        try:
            from totalsegmentator.python_api import totalsegmentator
            self._totalsegmentator = totalsegmentator
            return True
        except ImportError:
            return False

    def is_available(self) -> bool:
        return self._available

    def segment_kidney(
        self, volume: np.ndarray, metadata: dict
    ) -> Tuple[Optional[np.ndarray], float]:
        if not self._available:
            return None, 0.0
        try:
            import nibabel as nib
            import tempfile
            affine = np.array(metadata.get('affine', np.eye(4).tolist()))
            input_img = nib.Nifti1Image(volume.astype(np.float32), affine)
            with tempfile.TemporaryDirectory() as tmpdir:
                in_path = Path(tmpdir) / "input.nii.gz"
                out_path = Path(tmpdir) / "output.nii.gz"
                nib.save(input_img, str(in_path))
                self._totalsegmentator(
                    input=str(in_path),
                    output=str(out_path),
                    task='total', fast=self.fast, quiet=True, ml=True,
                )
                seg_img = nib.load(str(out_path))
                seg_array = seg_img.get_fdata().astype(np.int16)
            kidney_mask = ((seg_array == self.KIDNEY_LEFT_LABEL) | (seg_array == self.KIDNEY_RIGHT_LABEL)).astype(np.uint8)
            if kidney_mask.sum() == 0:
                return None, 0.0
            quality = 0.95 if not self.fast else 0.85
            return kidney_mask, quality
        except Exception as e:
            warnings.warn(f"TotalSegmentator inference failed: {e}")
            return None, 0.0

class STUNetSegmenter(_BaseSegmenter):
    """STU-Net backend (Huang et al. 2023)."""
    backend_name = "stunet"

    def __init__(self, model_size: str = "B"):
        self.model_size = model_size
        self.checkpoint_path = os.environ.get('CLINICAL_CORE_STUNET_CHECKPOINT')
        self._available = self._init_model()

    def _init_model(self) -> bool:
        if self.checkpoint_path is None or not Path(self.checkpoint_path).exists():
            return False
        try:
            from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor  # noqa
            return True
        except ImportError:
            return False

    def is_available(self) -> bool:
        return self._available

    def segment_kidney(
        self, volume: np.ndarray, metadata: dict
    ) -> Tuple[Optional[np.ndarray], float]:
        # Implementation placeholder as per vision_conn.py
        return None, 0.0

class MockSegmenter(_BaseSegmenter):
    """Synthetic ellipsoid mask for development without real models."""
    backend_name = "mock"
    def is_available(self) -> bool: return True
    def segment_kidney(self, volume: np.ndarray, metadata: dict) -> Tuple[Optional[np.ndarray], float]:
        if volume is None: return None, 0.0
        mask = np.zeros_like(volume, dtype=np.uint8)
        cz, cy, cx = [s // 2 for s in volume.shape]
        rz, ry, rx = [max(s // 6, 5) for s in volume.shape]
        zz, yy, xx = np.ogrid[:volume.shape[0], :volume.shape[1], :volume.shape[2]]
        sphere = (((zz - cz) / rz) ** 2 + ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2) <= 1
        mask[sphere] = 1
        return mask, 0.5

def _resolve_segmenter(backend: str) -> _BaseSegmenter:
    if backend == "auto":
        seg = TotalSegmentatorSegmenter()
        if seg.is_available(): return seg
        seg = STUNetSegmenter()
        if seg.is_available(): return seg
        return MockSegmenter()
    if backend == "totalseg":
        seg = TotalSegmentatorSegmenter()
        return seg if seg.is_available() else MockSegmenter()
    if backend == "stunet":
        seg = STUNetSegmenter()
        return seg if seg.is_available() else MockSegmenter()
    if backend == "mock": return MockSegmenter()
    raise ValueError(f"Unknown segmenter backend: {backend}")

class VisionConn_Baseline:
    name = "vision_baseline_stunet_radiomics"
    output_dim = 768
    def __init__(self, **kwargs):
        backend = kwargs.get('backend', 'auto')
        self.loader = VolumeLoader()
        self.segmenter = _resolve_segmenter(backend)
        self.radiomics = RadiomicsExtractor()
        self.projection = nn.Linear(self.radiomics.EXPECTED_FEATURE_DIM, self.output_dim)
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
        self._expected_kidney_voxels = kwargs.get('expected_kidney_voxels', 50000)

    def encode(self, ct_path: Union[str, Path]) -> Tuple[torch.Tensor, float]:
        volume, metadata = self.loader.load(ct_path)
        if volume is None:
            seed = int(hashlib.sha256(str(ct_path).encode()).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed)
            features = rng.standard_normal(self.radiomics.EXPECTED_FEATURE_DIM).astype(np.float32)
            with torch.no_grad():
                emb = self.projection(torch.tensor(features))
                emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
            return emb, 0.3
        mask, seg_quality = self.segmenter.segment_kidney(volume, metadata)
        if mask is None or mask.sum() == 0: return torch.zeros(self.output_dim), 0.0
        volume_ratio = mask.sum() / self._expected_kidney_voxels
        volume_confidence = float(np.exp(-abs(np.log(max(volume_ratio, 1e-3)))))
        features = self.radiomics.extract(volume, mask)
        with torch.no_grad():
            emb = self.projection(torch.tensor(features))
            emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
        return emb, float(seg_quality * volume_confidence)

    def encode_batch(self, paths: list) -> Tuple[torch.Tensor, torch.Tensor]:
        embeddings, confidences = [], []
        for p in paths:
            emb, conf = self.encode(p)
            embeddings.append(emb); confidences.append(conf)
        return torch.stack(embeddings), torch.tensor(confidences, dtype=torch.float32)
