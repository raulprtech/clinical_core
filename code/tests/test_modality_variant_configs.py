import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from core.experiment_runner import phase_5_multimodal  # noqa: E402
from core.reproducibility import resolve_runtime_paths  # noqa: E402


CONFIGS = {
    "experiment_config_kirc_tabular_only_portable_v1.yaml": {
        "modalities": ["tabular"],
        "required_precomputed": [],
    },
    "experiment_config_kirc_text_only_portable_v1.yaml": {
        "modalities": ["text"],
        "required_precomputed": ["text"],
    },
    "experiment_config_kirc_vision_only_portable_v1.yaml": {
        "modalities": ["vision"],
        "required_precomputed": ["vision"],
    },
    "experiment_config_kirc_tabular_text_portable_v1.yaml": {
        "modalities": ["tabular", "text"],
        "required_precomputed": ["text"],
    },
    "experiment_config_kirc_tabular_vision_portable_v1.yaml": {
        "modalities": ["tabular", "vision"],
        "required_precomputed": ["vision"],
    },
    "experiment_config_kirc_text_vision_portable_v1.yaml": {
        "modalities": ["text", "vision"],
        "required_precomputed": ["text", "vision"],
    },
    "experiment_config_kirc_trimodal_portable_v1.yaml": {
        "modalities": ["tabular", "text", "vision"],
        "required_precomputed": ["text", "vision"],
    },
}


class ModalityVariantConfigTests(unittest.TestCase):
    def test_configs_are_exact_portable_and_fail_fast(self):
        for filename, profile in CONFIGS.items():
            path = ROOT / "code" / "experiments" / filename
            source = path.read_text(encoding="utf-8")
            self.assertNotIn("/home/", source)
            manifest = yaml.safe_load(source)
            phase = manifest["phase_5_multimodal"]
            modalities = profile["modalities"]
            required_precomputed = profile["required_precomputed"]
            self.assertEqual(phase["modalities"], modalities)
            self.assertEqual(
                phase["required_precomputed_modalities"],
                required_precomputed,
            )
            self.assertEqual(phase["ablations"], [modalities])
            self.assertEqual(
                manifest["clinical_context"]["moment"], "post_surgery"
            )
            self.assertFalse(manifest["output"]["save_raw_extraction"])

            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                resolved = resolve_runtime_paths(
                    manifest,
                    path,
                    {
                        "CLINICAL_CORE_DATA_ROOT": str(root / "data"),
                        "CLINICAL_CORE_OUTPUT_ROOT": str(root / "output"),
                    },
                )
                if required_precomputed:
                    with self.assertRaisesRegex(
                        FileNotFoundError, "Required precomputed"
                    ):
                        phase_5_multimodal(
                            pd.DataFrame(),
                            pd.DataFrame(),
                            resolved,
                            root / "run",
                        )
                else:
                    self.assertIsNone(
                        resolved["phase_5_multimodal"]["text_embeddings_npz"]
                    )
                    self.assertIsNone(
                        resolved["phase_5_multimodal"]["vision_embeddings_csv"]
                    )


if __name__ == "__main__":
    unittest.main()
