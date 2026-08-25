import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RECEIPT = (
    ROOT
    / "publication"
    / "kirc_modality_profiles_v1"
    / "aggregate_validation.json"
)


class KircModalityValidationReceiptTests(unittest.TestCase):
    def setUp(self):
        self.payload = json.loads(RECEIPT.read_text(encoding="utf-8"))

    def test_receipt_has_all_seven_profiles_and_verified_code(self):
        self.assertEqual(
            self.payload["schema"],
            "clinical-core.kirc-modality-validation/v1",
        )
        modalities = {
            tuple(result["modalities"]) for result in self.payload["results"]
        }
        self.assertEqual(
            modalities,
            {
                ("tabular",),
                ("text",),
                ("vision",),
                ("tabular", "text"),
                ("tabular", "vision"),
                ("text", "vision"),
                ("tabular", "text", "vision"),
            },
        )
        implementation = self.payload["implementation"]
        actual = hashlib.sha256(
            (ROOT / implementation["path"]).read_bytes()
        ).hexdigest()
        self.assertEqual(actual, implementation["sha256"])
        self.assertEqual(
            self.payload["determinism_audit"]["post_repair"],
            "passed: 7 of 7 aggregate CSV pairs are byte-identical",
        )

    def test_receipt_is_aggregate_only(self):
        privacy = self.payload["privacy"]
        self.assertFalse(privacy["contains_patient_rows"])
        self.assertFalse(privacy["contains_case_identifiers"])
        self.assertFalse(privacy["contains_individual_predictions"])

        forbidden = {
            "case_id",
            "patient_id",
            "raw_features",
            "raw_targets",
            "individual_predictions",
        }

        def inspect(value):
            if isinstance(value, dict):
                self.assertTrue(forbidden.isdisjoint(value))
                for child in value.values():
                    inspect(child)
            elif isinstance(value, list):
                for child in value:
                    inspect(child)

        inspect(self.payload)


if __name__ == "__main__":
    unittest.main()
