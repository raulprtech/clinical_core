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

    def _write_canonical_run(self, directory):
        canonical = json.loads(compare_tool.DEFAULT_CANONICAL.read_text())
        payload = {
            "n_cases": canonical["cohort"]["parsed_cases"],
            "n_cases_survival": canonical["cohort"]["survival_cases"],
            "n_events_survival": canonical["cohort"]["survival_events"],
            "errors": canonical["errors"],
            "phases": {
                "phase_2_holdout": [
                    {"protocol": row["protocol"], "mean": row["cindex"]}
                    for row in canonical["holdout"]
                ],
                "phase_2_repeated_cv": [
                    {
                        "protocol": row["protocol"],
                        "mean": row["cindex_mean"],
                        "count": row["folds"],
                    }
                    for row in canonical["repeated_cross_validation"]
                ],
                "phase_2_temporal_validation": [
                    {
                        "protocol": row["protocol"],
                        "cindex": row["cindex"],
                        "cindex_ipcw": row["ipcw_cindex"],
                    }
                    for row in canonical["internal_temporal_transport"]
                ],
            },
        }
        summary = Path(directory) / "summary.json"
        summary.write_text(json.dumps(payload))
        return summary

    def test_comparison_accepts_canonical_aggregate(self):
        with tempfile.TemporaryDirectory() as temporary:
            summary = self._write_canonical_run(temporary)
            result = compare_tool.compare(
                summary,
                compare_tool.DEFAULT_CANONICAL,
                compare_tool.DEFAULT_MANIFEST,
            )
        self.assertTrue(result["passed"], result["failed"])

    def test_comparison_rejects_metric_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = self._write_canonical_run(temporary)
            payload = json.loads(source.read_text())
            payload["phases"]["phase_2_holdout"][0]["mean"] += 0.01
            source.write_text(json.dumps(payload))
            result = compare_tool.compare(
                source,
                compare_tool.DEFAULT_CANONICAL,
                compare_tool.DEFAULT_MANIFEST,
            )
        self.assertFalse(result["passed"])
        self.assertIn("holdout.stage_model.cindex", result["failed"])



if __name__ == "__main__":
    unittest.main()
