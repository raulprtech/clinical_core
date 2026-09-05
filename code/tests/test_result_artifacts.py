import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from check_result_artifacts import individual_payload


class ResultArtifactTests(unittest.TestCase):
    def test_rejects_individual_header(self):
        self.assertTrue(individual_payload("a.csv", "case_id,risk\nopaque,0.2\n"))

    def test_rejects_nested_individual_json(self):
        self.assertTrue(individual_payload("a.json", '{"cohort":{"patient_ids":[]}}'))

    def test_accepts_aggregate_counts_and_metrics(self):
        self.assertFalse(individual_payload("a.json", '{"patients":75,"events":20}'))
        self.assertFalse(individual_payload("a.csv", "fold,cindex\n0,0.7\n"))
