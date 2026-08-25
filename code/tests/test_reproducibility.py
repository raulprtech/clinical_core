import io
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from core.reproducibility import resolve_runtime_paths, strict_json_dump  # noqa: E402


class ReproducibilityTests(unittest.TestCase):
    def test_relative_paths_resolve_from_manifest_without_mutation(self):
        source = {
            "data": {"xml_dir": "../../data/xml", "feature_config": "../schema.yaml"},
            "output": {"base_dir": "../../results"},
        }
        with tempfile.TemporaryDirectory() as temporary:
            config_path = Path(temporary) / "code" / "experiments" / "run.yaml"
            resolved = resolve_runtime_paths(source, config_path)
            self.assertEqual(Path(resolved["data"]["xml_dir"]), Path(temporary) / "data/xml")
            self.assertEqual(source["data"]["xml_dir"], "../../data/xml")

    def test_modality_paths_expand_from_environment(self):
        source = {
            "data": {
                "xml_dir": "${CLINICAL_CORE_DATA_ROOT}/raw/xml",
                "feature_config": "../schema.yaml",
            },
            "output": {"base_dir": "${CLINICAL_CORE_OUTPUT_ROOT}/text"},
            "phase_2_text_only_nested_cv": {
                "text_embeddings_cache": "${CLINICAL_CORE_DATA_ROOT}/text.npz"
            },
            "phase_5_multimodal": {
                "text_embeddings_npz": "${CLINICAL_CORE_DATA_ROOT}/text.npz",
                "vision_embeddings_csv": "${CLINICAL_CORE_DATA_ROOT}/vision.csv",
                "text_dir": None,
                "vision_dir": "${CLINICAL_CORE_DATA_ROOT}/images",
                "vision_params": {
                    "weights_dir": "${CLINICAL_CORE_DATA_ROOT}/weights"
                },
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            resolved = resolve_runtime_paths(
                source,
                root / "configs" / "run.yaml",
                {
                    "CLINICAL_CORE_DATA_ROOT": str(root / "data"),
                    "CLINICAL_CORE_OUTPUT_ROOT": str(root / "outputs"),
                },
            )
        self.assertEqual(
            Path(resolved["phase_5_multimodal"]["vision_embeddings_csv"]),
            root / "data" / "vision.csv",
        )
        self.assertEqual(
            Path(resolved["phase_5_multimodal"]["vision_params"]["weights_dir"]),
            root / "data" / "weights",
        )
        self.assertIn("${CLINICAL_CORE_DATA_ROOT}", source["data"]["xml_dir"])

    def test_strict_json_replaces_all_non_finite_numbers_with_null(self):
        output = io.StringIO()
        strict_json_dump(
            {"nan": float("nan"), "inf": np.float64(math.inf), "values": [1.0]},
            output,
        )
        text = output.getvalue()
        self.assertNotIn("NaN", text)
        self.assertNotIn("Infinity", text)
        self.assertEqual(json.loads(text), {"nan": None, "inf": None, "values": [1.0]})


if __name__ == "__main__":
    unittest.main()
