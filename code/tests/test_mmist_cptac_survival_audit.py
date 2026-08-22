import sys
import unittest
from pathlib import Path

import pandas as pd

TOOLS_ROOT = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from audit_mmist_cptac_survival import build_audit, derive_os, infer_12_month_status  # noqa: E402


class MmistCptacSurvivalAuditTests(unittest.TestCase):
    def test_dead_case_uses_days_to_death(self):
        result = derive_os({
            "demographic": {"vital_status": "Dead", "days_to_death": 420},
            "follow_ups": [{"days_to_follow_up": 365}],
        })
        self.assertEqual(result["survival_days"], 420)
        self.assertEqual(result["event"], 1)

    def test_alive_case_uses_latest_available_followup(self):
        result = derive_os({
            "demographic": {"vital_status": "Alive"},
            "diagnoses": [{"days_to_last_follow_up": 300}],
            "follow_ups": [{"days_to_follow_up": 200}, {"days_to_follow_up": 500}],
        })
        self.assertEqual(result["survival_days"], 500)
        self.assertEqual(result["event"], 0)

    def test_short_censoring_has_unknown_12_month_status(self):
        self.assertIsNone(infer_12_month_status(200, 0))
        self.assertEqual(infer_12_month_status(200, 1), 0)
        self.assertEqual(infer_12_month_status(500, 0), 1)
        self.assertEqual(infer_12_month_status(500, 1), 1)

    def test_audit_keeps_only_cptac_and_counts_modalities(self):
        clinical = pd.DataFrame({
            "case_id": ["C3L-00001", "C3N-00002", "TCGA-AA-0001"],
            "vital_status_12": [1, 0, 1],
        })
        ct = pd.DataFrame({"case_id": ["C3L-00001", "C3L-00001"]})
        mr = pd.DataFrame({"case_id": ["C3N-00002"]})
        payload = {"data": {"hits": [
            {
                "submitter_id": "C3L-00001",
                "demographic": {"vital_status": "Alive"},
                "follow_ups": [{"days_to_follow_up": 600}],
            },
            {
                "submitter_id": "C3N-00002",
                "demographic": {"vital_status": "Dead", "days_to_death": 100},
            },
        ]}}
        audit, summary = build_audit(clinical, ct, mr, payload)
        self.assertEqual(len(audit), 2)
        self.assertEqual(summary["cohorts"]["ct"]["mapped_cases"], 1)
        self.assertEqual(summary["cohorts"]["mr"]["events"], 1)
        self.assertEqual(summary["endpoint_crosscheck"]["agreements"], 2)


if __name__ == "__main__":
    unittest.main()
