import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from components.adapters.ingestion.vision.models.resnet_multiview import (
    VisionResNet18_2D,
    VisionResNet18_2p5D,
    VisionResNet50_2D,
    load_precomputed_embeddings,
)
from components.adapters.ingestion.text.models.clinicalbert import (
    load_precomputed_text_embeddings,
)
from core.model_utils import verify_ingestion_contract
from core.main import discover_modality_files
from core.registry import get_vision_conn, list_components


class _FourFeatureBackbone(nn.Module):
    def forward(self, batch):
        means = batch.mean(dim=(2, 3))
        return torch.cat((means, means[:, :1]), dim=1).unsqueeze(-1).unsqueeze(-1)


class VisionResNetTests(unittest.TestCase):
    def test_registry_exposes_all_notebook_variants(self):
        names = list_components()['vision_conn']
        self.assertIn('vision_resnet18_2d', names)
        self.assertIn('vision_resnet50_2d', names)
        self.assertIn('vision_resnet18_2p5d', names)

    def test_2d_repeats_center_slice_and_2p5d_uses_neighbors(self):
        volume = np.indices((7, 8, 9))[0].astype(np.float32)
        common = dict(
            use_imagenet_weights=False, image_size=16,
            window_low=0, window_high=6,
            backbone=_FourFeatureBackbone(), feature_dim=4,
        )
        model_2d = VisionResNet18_2D(**common)
        model_2p5d = VisionResNet18_2p5D(**common)
        two_d = model_2d.volume_to_views(volume)
        two_p5d = model_2p5d.volume_to_views(volume)
        raw_2d = two_d * model_2d.imagenet_std + model_2d.imagenet_mean
        raw_2p5d = two_p5d * model_2p5d.imagenet_std + model_2p5d.imagenet_mean
        self.assertTrue(torch.allclose(raw_2d[0, 0], raw_2d[0, 1]))
        self.assertFalse(torch.allclose(raw_2p5d[0, 0], raw_2p5d[0, 1]))

    def test_resnet18_padding_and_resnet50_projection_satisfy_contract(self):
        feature18 = torch.arange(1, 513, dtype=torch.float32)
        model18 = VisionResNet18_2D(use_imagenet_weights=False)
        emb18 = model18._contract_projection(feature18)
        self.assertTrue(verify_ingestion_contract(
            emb18.unsqueeze(0), torch.ones(1, 1), verbose=False
        )['contract_satisfied'])

        feature50 = torch.arange(1, 2049, dtype=torch.float32)
        first = VisionResNet50_2D(use_imagenet_weights=False)
        second = VisionResNet50_2D(use_imagenet_weights=False)
        emb50 = first._contract_projection(feature50)
        self.assertTrue(torch.allclose(emb50, second._contract_projection(feature50)))
        self.assertEqual(tuple(emb50.shape), (768,))

    def test_notebook_embedding_csv_is_loaded_and_normalized(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'vision_embeddings.csv'
            row = {'case_id': 'tcga-aa-0001', 'vision_confidence': 0.8}
            row.update({f'z{i:03d}': float(i + 1) for i in range(768)})
            pd.DataFrame([row]).to_csv(path, index=False)
            loaded = load_precomputed_embeddings(path)
        embedding, confidence = loaded['TCGA-AA-0001']
        self.assertAlmostEqual(float(embedding.norm()), 1.0, places=5)
        self.assertAlmostEqual(confidence, 0.8)

    def test_precomputed_text_npz_is_loaded_and_missing_rows_are_skipped(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'text_embeddings.npz'
            embeddings = np.vstack((np.ones(768), np.zeros(768))).astype(np.float32)
            np.savez(
                path,
                embeddings=embeddings,
                confidence=np.array([0.9, 0.0], dtype=np.float32),
                case_ids=np.array(['tcga-aa-0001', 'tcga-aa-0002'], dtype=object),
            )
            loaded = load_precomputed_text_embeddings(path)
        self.assertEqual(set(loaded), {'TCGA-AA-0001'})
        self.assertAlmostEqual(float(loaded['TCGA-AA-0001'][0].norm()), 1.0, places=5)

    def test_registry_forwards_connector_parameters(self):
        model = get_vision_conn(
            'vision_resnet18_2p5d', use_imagenet_weights=False,
            image_size=32, slice_offsets=[-2, 0, 2],
        )
        self.assertEqual(model.image_size, 32)
        self.assertEqual(model.slice_offsets, (-2, 0, 2))

    def test_discovery_resolves_tcia_series_with_extensionless_dicoms(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            series = root / 'TCGA-AA-0001' / '1.2.3' / 'nested'
            series.mkdir(parents=True)
            for index in range(3):
                (series / str(index)).write_bytes(b'DICOM')
            manifest = discover_modality_files(
                {'text_dir': None, 'vision_dir': str(root)},
                ['TCGA-AA-0001'],
            )
        self.assertEqual(
            Path(manifest.loc['TCGA-AA-0001', 'vision_path']).name,
            '1.2.3',
        )


if __name__ == '__main__':
    unittest.main()
