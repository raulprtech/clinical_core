import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code"))

from core.frozen_cox_bundle import (
    assert_privacy_safe,
    canonical_bytes,
    fit_frozen_stage_model,
    predict_partial_hazard,
    write_bundle,
)


class FrozenCoxBundleTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        size = 80
        index = [f"subject-{number:03d}" for number in range(size)]
        self.features = pd.DataFrame({
            "age": rng.normal(65, 8, size),
            "gender": rng.integers(0, 2, size).astype(float),
            "race": rng.integers(0, 4, size).astype(float),
            "ethnicity": rng.integers(0, 2, size).astype(float),
            "pathologic_stage": rng.integers(1, 5, size).astype(float),
            "pathologic_T": rng.integers(1, 5, size).astype(float),
            "pathologic_N": rng.integers(0, 4, size).astype(float),
            "pathologic_M": rng.integers(0, 2, size).astype(float),
            "pack_years_smoked": rng.normal(35, 15, size),
        }, index=index)
        self.features.loc[index[::9], "pack_years_smoked"] = np.nan
        linear = 0.7 * self.features["pathologic_stage"] + 0.01 * self.features["age"]
        time = rng.exponential(1200 / np.exp(linear - linear.mean())) + 1
        event = rng.binomial(1, 0.65, size)
        self.targets = pd.DataFrame(
            {"survival_days": time, "event": event}, index=index
        )

    def test_reload_reproduces_and_export_is_byte_stable(self):
        provenance = {"fixture": "synthetic"}
        first, _ = fit_frozen_stage_model(
            self.features.sample(frac=1, random_state=1), self.targets, provenance=provenance
        )
        second, _ = fit_frozen_stage_model(
            self.features.sample(frac=1, random_state=2), self.targets, provenance=provenance
        )
        self.assertEqual(canonical_bytes(first), canonical_bytes(second))
        raw = self.features.drop(columns=["pathologic_T", "pathologic_N", "pathologic_M"])
        risk = predict_partial_hazard(first, raw)
        self.assertEqual(risk.shape, (len(raw),))
        self.assertTrue(np.isfinite(risk).all())

    def test_receipt_and_privacy_guard(self):
        bundle, _ = fit_frozen_stage_model(
            self.features, self.targets, provenance={"fixture": "synthetic"}
        )
        assert_privacy_safe(bundle)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "model.json"
            digest = write_bundle(bundle, output)
            self.assertEqual(len(digest), 64)
            self.assertTrue(output.with_suffix(".json.sha256").exists())
            json.loads(output.read_text())
        bundle["case_id"] = "forbidden"
        with self.assertRaises(ValueError):
            assert_privacy_safe(bundle)


    def test_committed_bundle_receipt_and_contract(self):
        model_path = REPO_ROOT / "publication/luad_frozen_stage_model_v1/model.json"
        receipt_path = model_path.with_suffix(".json.sha256")
        payload = model_path.read_bytes()
        expected = receipt_path.read_text().split()[0]
        self.assertEqual(hashlib.sha256(payload).hexdigest(), expected)
        bundle = json.loads(payload)
        self.assertEqual(bundle["format"], "clinical-core-portable-cox-v1")
        self.assertEqual(bundle["model"]["protocol"], "stage_model")
        self.assertEqual(bundle["training_summary"]["cases"], 507)
        self.assertFalse(bundle["privacy"]["contains_patient_identifiers"])
        assert_privacy_safe(bundle)


if __name__ == "__main__":
    unittest.main()
