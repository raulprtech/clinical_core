import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def load_tool(name):
    path = REPO_ROOT / "code/tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


compare_tool = load_tool("compare_luad_reproduction")
restore_tool = load_tool("restore_luad_source")


class LuadReproductionToolTests(unittest.TestCase):
    def test_frozen_source_inventory_is_valid_and_identifier_free(self):
        inventory = restore_tool.load_inventory(restore_tool.DEFAULT_INVENTORY)
        self.assertEqual(inventory["file_count"], 522)
        rendered = json.dumps(inventory)
        self.assertNotIn("file_name", rendered)
        self.assertNotIn("case_id", rendered)

    def test_comparison_accepts_canonical_local_run(self):
        manifest = json.loads(compare_tool.DEFAULT_MANIFEST.read_text())
        summary = REPO_ROOT / manifest["canonical_local_run"]["path"] / "summary.json"
        result = compare_tool.compare(
            summary,
            compare_tool.DEFAULT_CANONICAL,
            compare_tool.DEFAULT_MANIFEST,
        )
        self.assertTrue(result["passed"], result["failed"])

    def test_comparison_rejects_metric_drift(self):
        manifest = json.loads(compare_tool.DEFAULT_MANIFEST.read_text())
        source = REPO_ROOT / manifest["canonical_local_run"]["path"] / "summary.json"
        payload = json.loads(source.read_text())
        payload["phases"]["phase_2_holdout"][0]["mean"] += 0.01
        with tempfile.TemporaryDirectory() as temporary:
            changed = Path(temporary) / "summary.json"
            changed.write_text(json.dumps(payload))
            result = compare_tool.compare(
                changed,
                compare_tool.DEFAULT_CANONICAL,
                compare_tool.DEFAULT_MANIFEST,
            )
        self.assertFalse(result["passed"])
        self.assertIn("holdout.stage_model.cindex", result["failed"])


if __name__ == "__main__":
    unittest.main()
