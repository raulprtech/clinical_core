import sys
import unittest
from pathlib import Path

import pandas as pd

TOOLS_ROOT = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from audit_tcia_cptac_inventory import build_inventory_summary  # noqa: E402


class TciaCptacInventoryAuditTests(unittest.TestCase):
    def test_additional_tcia_case_extends_ct_without_event(self):
        clinical = pd.DataFrame({
            "case_id": ["C3L-00001", "C3N-00002", "TCGA-AA-0001"],
            "vital_status_12": [1, 1, 1],
        })
        ct = pd.DataFrame({"case_id": ["C3L-00001"]})
        mr = pd.DataFrame({"case_id": []})
        series = [
            {"PatientID": "C3L-00001", "Modality": "CT"},
            {"PatientID": "C3N-00002", "Modality": "CT"},
            {"PatientID": "C3N-00002", "Modality": "RTSTRUCT"},
            {"PatientID": "C3L-99999", "Modality": "MR"},
        ]
        gdc = {"data": {"hits": [
            {
                "submitter_id": "C3L-00001",
                "demographic": {"vital_status": "Dead", "days_to_death": 500},
            },
            {
                "submitter_id": "C3N-00002",
                "demographic": {"vital_status": "Alive"},
                "follow_ups": [{"days_to_follow_up": 700}],
            },
        ]}}
        _, summary = build_inventory_summary(clinical, ct, mr, series, gdc)
        crosscheck = summary["eligibility_crosscheck"]
        self.assertEqual(crosscheck["additional_within_mmist_clinical"], 1)
        self.assertEqual(crosscheck["additional_events"], 0)
        self.assertEqual(summary["extended_known_ccrcc_ct"]["events"], 1)
        self.assertEqual(summary["extended_known_ccrcc_ct"]["valid_os"], 2)


if __name__ == "__main__":
    unittest.main()
