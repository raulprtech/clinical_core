"""
VISION-CONN: Radiomics Feature Extraction
==========================================

PyRadiomics wrapper and visual feature extraction.
"""

import numpy as np
import hashlib
import warnings

class RadiomicsExtractor:
    """PyRadiomics wrapper. Mock fallback computes simple statistics."""

    EXPECTED_FEATURE_DIM = 100

    def __init__(self):
        self._extractor = None
        self._mode = self._init_backend()

    def _init_backend(self) -> str:
        try:
            from radiomics import featureextractor
            self._extractor = featureextractor.RadiomicsFeatureExtractor()
            return "real"
        except ImportError:
            return "mock"

    def extract(self, volume: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if self._mode == "real":
            try:
                import SimpleITK as sitk
                vol_sitk = sitk.GetImageFromArray(volume.astype(np.float32))
                mask_sitk = sitk.GetImageFromArray(mask.astype(np.uint8))
                features = self._extractor.execute(vol_sitk, mask_sitk)
                values = [
                    float(v) for k, v in features.items()
                    if not k.startswith('diagnostics_')
                    and isinstance(v, (int, float, np.number))
                ]
                arr = np.array(values, dtype=np.float32)
                if len(arr) < self.EXPECTED_FEATURE_DIM:
                    arr = np.pad(arr, (0, self.EXPECTED_FEATURE_DIM - len(arr)))
                return arr[:self.EXPECTED_FEATURE_DIM]
            except Exception as e:
                warnings.warn(f"PyRadiomics failed: {e}, using mock features")

        # MOCK / FALLBACK
        roi = volume[mask > 0]
        if len(roi) == 0:
            return np.zeros(self.EXPECTED_FEATURE_DIM, dtype=np.float32)

        first_order = [
            float(roi.mean()), float(roi.std()),
            float(roi.min()), float(roi.max()),
            float(np.percentile(roi, 10)), float(np.percentile(roi, 25)),
            float(np.percentile(roi, 50)), float(np.percentile(roi, 75)),
            float(np.percentile(roi, 90)),
            float(roi.var()), float(np.median(np.abs(roi - np.median(roi)))),
        ]
        shape = [
            float(mask.sum()),
            float(mask.sum() / max(volume.size, 1)),
            float(mask.shape[0]), float(mask.shape[1]), float(mask.shape[2]),
        ]
        base = np.array(first_order + shape, dtype=np.float32)

        seed = int(hashlib.md5(volume.tobytes()[:1024]).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        padding = rng.standard_normal(
            self.EXPECTED_FEATURE_DIM - len(base)
        ).astype(np.float32) * 0.01
        return np.concatenate([base, padding])
