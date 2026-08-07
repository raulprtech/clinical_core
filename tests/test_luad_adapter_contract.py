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

MAPPING = CODE_ROOT / "components/adapters/ingestion/tabular/configs/tabular_mapping_tcga_luad_baseline_os_v2.yaml"
EXPERIMENT = CODE_ROOT / "experiments/experiment_config_nigma_luad_baseline_os_v2.yaml"


class LuadAdapterContractTests(unittest.TestCase):
    def setUp(self):
        self.extractor = TCGAExtractor(str(MAPPING))
        self.mapping = yaml.safe_load(MAPPING.read_text(encoding="utf-8"))
        self.experiment = yaml.safe_load(EXPERIMENT.read_text(encoding="utf-8"))

    def test_contract_is_luad_only_retrospective_baseline_and_sealed(self):
        contract = self.mapping["contract"]
        self.assertEqual(contract["project_id"], "TCGA-LUAD")
        self.assertEqual(contract["excluded_project_ids"], ["TCGA-LUSC"])
        self.assertEqual(contract["prediction_time"], "retrospective_pathology_baseline")
        self.assertEqual(contract["time_origin"], "initial_pathologic_diagnosis")
        self.assertIn("not_postoperative", contract["claim_scope"])
        self.assertEqual(
            self.experiment["disease_contract"]["source_status"],
            "local_verified_522_patient_xml",
        )
        holdout = self.experiment["phase_2_holdout"]
        self.assertTrue(holdout["enabled"])
        self.assertEqual(holdout["holdout_fraction"], 0.20)
        self.assertEqual(holdout["seeds"], [42])
        self.assertEqual(holdout["variants"], ["cox_baseline"])
        self.assertFalse(holdout["save_artifacts"])
        self.assertEqual(holdout["onehot_features"], ["race"])
        self.assertTrue(holdout["onehot_drop_first"])
        self.assertEqual(
            holdout["protocols"],
            [
                {
                    "name": "stage_model",
                    "drop_features": ["pathologic_T", "pathologic_N", "pathologic_M"],
                },
                {"name": "tnm_model", "drop_features": ["pathologic_stage"]},
                {"name": "stage_tnm_model", "drop_features": []},
            ],
        )
        self.assertEqual(holdout["calibration_horizon_days"], 730)
        self.assertEqual(holdout["bootstrap_iterations"], 1000)
        self.assertEqual(holdout["bootstrap_confidence_level"], 0.95)
        self.assertEqual(holdout["ipcw_tau_days"], 730)
        self.assertFalse(self.experiment["output"]["save_raw_extraction"])
        repeated = self.experiment["phase_2_repeated_cv"]
        self.assertEqual(repeated["seeds"], [42, 101, 202])
        self.assertEqual(repeated["n_folds"], 5)
        self.assertEqual(repeated["protocols"], holdout["protocols"])
        self.assertTrue(repeated["onehot_drop_first"])
        self.assertFalse(repeated["save_artifacts"])
        temporal = self.experiment["phase_2_temporal_validation"]
        self.assertTrue(temporal["enabled"])
        self.assertEqual(temporal["cutoff_year"], 2007)
        self.assertEqual(temporal["partition_field"], "diagnosis_year")
        self.assertEqual(temporal["protocols"], holdout["protocols"])
        self.assertFalse(temporal["save_artifacts"])
        self.assertEqual(
            self.mapping["targets"]["diagnosis_year"]["role"],
            "partition_only",
        )
        self.assertNotIn("diagnosis_year", self.mapping["features"])
        enabled = [
            name for name, phase in self.experiment.items()
            if name.startswith("phase_") and isinstance(phase, dict)
            and phase.get("enabled") is True
        ]
        self.assertEqual(
            enabled,
            [
                "phase_2_holdout",
                "phase_2_repeated_cv",
                "phase_2_temporal_validation",
            ],
        )

    def test_contract_excludes_unaudited_renal_features(self):
        features = set(self.mapping["features"])
        portable = {"age", "gender", "race", "ethnicity", "pathologic_stage", "pathologic_T", "pathologic_N", "pathologic_M"}
        excluded = {"hemoglobin", "ldh", "serum_calcium", "platelet_count", "white_cell_count", "tumor_status", "cigarettes_per_day"}
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
                "<year_of_initial_pathologic_diagnosis>2007</year_of_initial_pathologic_diagnosis>"
                "</patient>",
                encoding="utf-8",
            )
            features, targets = self.extractor.extract_cohort(directory)
        self.assertEqual(targets.loc["TCGA-LU-0001", "event"], 0)
        self.assertEqual(targets.loc["TCGA-LU-0001", "survival_days"], 400)
        self.assertEqual(features.loc["TCGA-LU-0001", "pathologic_stage"], 2)
        self.assertEqual(features.loc["TCGA-LU-0001", "pack_years_smoked"], 30.5)
        self.assertEqual(targets.loc["TCGA-LU-0001", "diagnosis_year"], 2007)
        self.assertNotIn("diagnosis_year", features.columns)

    def test_death_time_has_priority_over_follow_up(self):
        raw = {
            "target__vital_status": "Dead",
            "target__source__days_to_death": "250",
            "target__source__days_to_follow_up": "400",
        }
        self.assertEqual(self.extractor._resolve_survival(raw), (250.0, 1))

    def test_latest_longitudinal_follow_up_is_used(self):
        with tempfile.TemporaryDirectory() as directory:
            xml = Path(directory) / "case.xml"
            xml.write_text(
                "<patient>"
                "<bcr_patient_barcode>TCGA-LU-0002</bcr_patient_barcode>"
                "<vital_status>Alive</vital_status>"
                "<days_to_last_followup>100</days_to_last_followup>"
                "<follow_up><days_to_last_followup>650</days_to_last_followup></follow_up>"
                "</patient>",
                encoding="utf-8",
            )
            _, targets = self.extractor.extract_cohort(directory)
        self.assertEqual(targets.loc["TCGA-LU-0002", "survival_days"], 650)

    def test_race_onehot_is_required_by_experiment(self):
        self.assertEqual(
            self.experiment["phase_2_holdout"]["onehot_features"],
            ["race"],
        )

    def test_stage_fallback_prefers_most_specific_label(self):
        config = self.mapping["features"]["pathologic_stage"]
        self.assertEqual(self.extractor._apply_mapping("Stage II NOS", config), 2)


if __name__ == "__main__":
    unittest.main()
