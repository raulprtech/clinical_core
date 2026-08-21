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
