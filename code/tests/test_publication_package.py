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

    def test_available_local_evidence_is_verifiable(self):
        manifest = MODULE._strict_load(MODULE.DEFAULT_MANIFEST)
        source_path = REPO_ROOT / manifest["source"]["local_manifest"]
        restore_path = source_path.with_name("RESTORE.json")
        run_path = REPO_ROOT / manifest["canonical_local_run"]["path"]
        result = MODULE.verify_manifest(
            require_data=source_path.exists() or restore_path.exists(),
            require_run=run_path.exists(),
        )
        if source_path.exists() or restore_path.exists():
            self.assertIn(result["data"], {"verified", "verified_restoration"})
        if run_path.exists():
            self.assertEqual(result["run"], "verified")


if __name__ == "__main__":
    unittest.main()
