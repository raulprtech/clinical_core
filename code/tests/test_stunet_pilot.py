import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import SimpleITK as sitk


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from tools.build_stunet_embeddings import (  # noqa: E402
    PatchFeature,
    STUNetRuntime,
    dicom_geometry_metrics,
    evenly_spaced_rows,
    existing_nifti_metrics,
)


class STUNetPilotTests(unittest.TestCase):
    def test_evenly_spaced_selection_includes_endpoints(self):
        frame = pd.DataFrame({"value": range(11)})
        selected = evenly_spaced_rows(frame, 3)
        self.assertEqual(selected["value"].tolist(), [0, 5, 10])

    def test_kidney_roi_is_reproducible_and_clipped(self):
        segmentation = np.zeros((20, 30, 40), dtype=np.uint8)
        segmentation[2:5, 10:14, 35:39] = 38
        roi = STUNetRuntime._kidney_roi(segmentation, margin_voxels=3)
        self.assertEqual(
            [(part.start, part.stop) for part in roi],
            [(0, 8), (7, 17), (32, 40)],
        )

    def test_existing_nifti_metrics_reads_header_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "case.nii.gz"
            image = sitk.Image([2, 3, 4], sitk.sitkInt16)
            image.SetSpacing([0.7, 0.8, 1.5])
            sitk.WriteImage(image, str(path))
            row = SimpleNamespace(
                SeriesInstanceUID="1.2.3",
                ImageCount_num=4,
                geometry_qc="pass",
            )
            metrics = existing_nifti_metrics(path, row)

        self.assertEqual(metrics["dicom_series_uid"], "1.2.3")
        self.assertEqual(metrics["input_size_xyz"], [2, 3, 4])
        np.testing.assert_allclose(
            metrics["input_spacing_xyz_mm"], [0.7, 0.8, 1.5], atol=1e-6
        )
        self.assertEqual(metrics["nifti_storage_dtype"], "16-bit signed integer")
        self.assertTrue(metrics["input_reused_from_existing_nifti"])

    def test_volumetric_moments_capture_dispersion_and_keep_legacy_api(self):
        runtime = STUNetRuntime.__new__(STUNetRuntime)
        runtime.patch_size = np.asarray([4, 4, 4])
        runtime.network = SimpleNamespace(
            _get_gaussian=lambda _patch_size: np.ones((4, 4, 4), dtype=np.float32)
        )
        segmentation = np.full((4, 4, 4), 38, dtype=np.uint8)
        features = np.zeros((256, 2, 2, 2), dtype=np.float32)
        features[0] = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
        features[1] = 2.0
        patch_features = [PatchFeature((0, 0, 0), features)]

        variants, metrics = runtime._pool_embedding_variants(
            segmentation, patch_features, margin_voxels=0
        )
        legacy, legacy_metrics = runtime._pool_embedding(
            segmentation, patch_features, margin_voxels=0
        )

        self.assertEqual(variants["mean_768"].shape, (768,))
        self.assertEqual(variants["renal_moments_512"].shape, (512,))
        self.assertAlmostEqual(np.linalg.norm(variants["mean_768"]), 1.0, places=6)
        self.assertAlmostEqual(
            np.linalg.norm(variants["renal_moments_512"]), 1.0, places=6
        )
        self.assertGreater(variants["renal_moments_512"][256], 0.0)
        np.testing.assert_allclose(legacy, variants["mean_768"], atol=1e-7)
        self.assertEqual(metrics["embedding_dims"]["renal_moments_512"], 512)
        self.assertEqual(legacy_metrics["embedding_dim"], 768)

    @staticmethod
    def _dataset_at(z_position):
        return SimpleNamespace(
            ImageOrientationPatient=[1, 0, 0, 0, 1, 0],
            ImagePositionPatient=[0, 0, z_position],
        )

    @patch("tools.build_stunet_embeddings.pydicom.dcmread")
    def test_geometry_qc_accepts_uniform_positions(self, dcmread):
        dcmread.side_effect = [self._dataset_at(z) for z in (0, 5, 10, 15)]
        metrics = dicom_geometry_metrics(["a", "b", "c", "d"])
        self.assertEqual(metrics["geometry_qc"], "pass")
        self.assertEqual(metrics["slice_gap_ratio"], 1.0)

    @patch("tools.build_stunet_embeddings.pydicom.dcmread")
    def test_geometry_qc_rejects_missing_or_duplicate_slices(self, dcmread):
        dcmread.side_effect = [self._dataset_at(z) for z in (0, 5, 10, 20, 20)]
        metrics = dicom_geometry_metrics(["a", "b", "c", "d", "e"])
        self.assertEqual(metrics["geometry_qc"], "fail")
        self.assertEqual(metrics["duplicate_slice_positions"], 1)
        self.assertEqual(metrics["slice_gap_ratio"], 2.0)


if __name__ == "__main__":
    unittest.main()
