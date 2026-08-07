import sys
import tempfile
import unittest
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = REPO_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from components.adapters.ingestion.tabular.utils.extractor import TCGAExtractor  # noqa: E402

MAPPING = CODE_ROOT / "components/adapters/ingestion/tabular/configs/tabular_mapping_tcga_luad_postop_os_v1.yaml"
EXPERIMENT = CODE_ROOT / "experiments/experiment_config_nigma_luad_postop_os_v1.yaml"


class LuadAdapterContractTests(unittest.TestCase):
    def setUp(self):
        self.extractor = TCGAExtractor(str(MAPPING))
        self.mapping = yaml.safe_load(MAPPING.read_text(encoding="utf-8"))
        self.experiment = yaml.safe_load(EXPERIMENT.read_text(encoding="utf-8"))

    def test_contract_is_luad_only_postoperative_and_non_executable(self):
        contract = self.mapping["contract"]
        self.assertEqual(contract["project_id"], "TCGA-LUAD")
        self.assertEqual(contract["excluded_project_ids"], ["TCGA-LUSC"])
        self.assertEqual(contract["prediction_time"], "post_surgery")
        self.assertEqual(
            self.experiment["disease_contract"]["source_status"],
            "reference_registered_data_not_present_locally",
        )
        self.assertFalse(self.experiment["phase_2_holdout"]["enabled"])

    def test_contract_excludes_unaudited_renal_features(self):
        features = set(self.mapping["features"])
        portable = {"age", "gender", "race", "ethnicity", "pathologic_stage", "pathologic_T", "pathologic_N", "pathologic_M"}
        excluded = {"hemoglobin", "ldh", "serum_calcium", "platelet_count", "white_cell_count", "tumor_status"}
        self.assertTrue(portable <= features)
        self.assertTrue(excluded.isdisjoint(features))

    def test_current_gdc_follow_up_alias_resolves_censoring(self):
        with tempfile.TemporaryDirectory() as directory:
            xml = Path(directory) / "case.xml"
            xml.write_text(
                "<patient>"
                "<bcr_patient_barcode>TCGA-LU-0001</bcr_patient_barcode>"
                "<vital_status>Alive</vital_status>"
                "<days_to_follow_up>400</days_to_follow_up>"
                "<ajcc_pathologic_stage>Stage IIA</ajcc_pathologic_stage>"
                "<number_pack_years_smoked>30.5</number_pack_years_smoked>"
                "<cigarettes_per_day>12</cigarettes_per_day>"
                "</patient>",
                encoding="utf-8",
            )
            features, targets = self.extractor.extract_cohort(directory)
        self.assertEqual(targets.loc["TCGA-LU-0001", "event"], 0)
        self.assertEqual(targets.loc["TCGA-LU-0001", "survival_days"], 400)
        self.assertEqual(features.loc["TCGA-LU-0001", "pathologic_stage"], 2)
        self.assertEqual(features.loc["TCGA-LU-0001", "pack_years_smoked"], 30.5)

    def test_death_time_has_priority_over_follow_up(self):
        raw = {
            "target__vital_status": "Dead",
            "target__source__days_to_death": "250",
            "target__source__days_to_follow_up": "400",
        }
        self.assertEqual(self.extractor._resolve_survival(raw), (250.0, 1))

    def test_stage_fallback_prefers_most_specific_label(self):
        config = self.mapping["features"]["pathologic_stage"]
        self.assertEqual(self.extractor._apply_mapping("Stage II NOS", config), 2)


if __name__ == "__main__":
    unittest.main()
