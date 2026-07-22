import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from tools.build_stunet_embeddings import (  # noqa: E402
    STUNetRuntime,
    dicom_geometry_metrics,
    evenly_spaced_rows,
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
