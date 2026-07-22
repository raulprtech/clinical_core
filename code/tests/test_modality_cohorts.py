import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from core.experiment_runner import apply_cohort_filter, phase_2_text_only_nested_cv
from core.main import EmbeddingCache, select_cases_for_modalities


class ModalityCohortTests(unittest.TestCase):
    def setUp(self):
        self.targets = pd.DataFrame(
            {
                'survival_days': [10.0, 20.0, 30.0, float('nan')],
                'event': [1, 0, 1, 0],
            },
            index=['tab-only', 'text-only', 'both', 'invalid-survival'],
        )
        self.cache = EmbeddingCache()
        self.cache.put('tab-only', 'tabular', torch.ones(2), 1.0)
        self.cache.put('text-only', 'text', torch.ones(2), 1.0)
        self.cache.put('both', 'tabular', torch.ones(2), 1.0)
        self.cache.put('both', 'text', torch.ones(2), 1.0)
        self.cache.put('invalid-survival', 'text', torch.ones(2), 1.0)

    def test_each_subset_uses_only_its_required_modalities(self):
        self.assertEqual(
            select_cases_for_modalities(self.cache, self.targets, ['tabular']),
            ['tab-only', 'both'],
        )
        self.assertEqual(
            select_cases_for_modalities(self.cache, self.targets, ['text']),
            ['text-only', 'both'],
        )
        self.assertEqual(
            select_cases_for_modalities(
                self.cache, self.targets, ['tabular', 'text']
            ),
            ['both'],
        )

    def test_pathology_text_is_rejected_pre_surgery(self):
        config = {
            'clinical_context': {'moment': 'pre_surgery'},
            'phase_2_text_only_nested_cv': {'enabled': True},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(ValueError, 'Pre-surgery'):
                phase_2_text_only_nested_cv(
                    self.targets, config, Path(tmpdir)
                )

    def test_per_modality_policy_does_not_shrink_global_cohort(self):
        features = pd.DataFrame({'x': [1, 2, 3]}, index=['a', 'b', 'c'])
        targets = pd.DataFrame(
            {'survival_days': [10, 20, 30], 'event': [1, 0, 1]},
            index=features.index,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            manifest_path = tmpdir / 'modalities.csv'
            pd.DataFrame(
                {
                    'case_id': ['a', 'b', 'c'],
                    'has_wsi': [True, False, True],
                    'has_mrna': [False, True, True],
                }
            ).to_csv(manifest_path, index=False)
            filtered_features, _, audit = apply_cohort_filter(
                features,
                targets,
                {
                    'enabled': True,
                    'modality_policy': 'per_modality',
                    'modality_manifest_path': str(manifest_path),
                    'require_modalities': ['wsi', 'mrna'],
                },
                tmpdir,
            )
        self.assertEqual(list(filtered_features.index), ['a', 'b', 'c'])
        self.assertEqual(audit['n_final'], 3)
        self.assertEqual(audit['modality_counts'], {'wsi': 2, 'mrna': 2})


if __name__ == '__main__':
    unittest.main()
