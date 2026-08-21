import importlib.util
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "code/tools/verify_luad_publication_package.py"
SPEC = importlib.util.spec_from_file_location("verify_luad_publication_package", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class PublicationPackageTests(unittest.TestCase):
    def test_tracked_package_is_strict_aggregate_only_and_hash_verified(self):
        result = MODULE.verify_manifest()
        self.assertGreaterEqual(len(result["tracked_files"]), 4)

    def test_current_data_and_canonical_run_are_verifiable(self):
        result = MODULE.verify_manifest(require_data=True, require_run=True)
        self.assertEqual(result["data"], "verified")
        self.assertEqual(result["run"], "verified")


if __name__ == "__main__":
    unittest.main()
