import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
import copy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from build_renal_2p5d_program_cache import aligned_arrays, renal_box, plane_features, radiomics_2d
from evaluate_renal_2p5d_program import comparable_scores
from evaluate_resnet_sequence_models import safe_cindex
from evaluate_renal_resnet_adaptation import two_pass_backward, cox_ph_loss


class RenalProgramTests(unittest.TestCase):
    def test_two_pass_gradient_matches_full_cohort_with_tied_times(self):
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 1)
            def forward(self, x):
                return self.linear(x).squeeze()
        torch.manual_seed(5)
        first = Tiny()
        second = copy.deepcopy(first)
        xs = torch.randn(6, 3)
        times = torch.tensor([1., 2., 2., 3., 4., 4.])
        events = torch.tensor([1., 1., 0., 1., 0., 1.])
        risk = torch.stack([first(x) for x in xs])
        cox_ph_loss(risk, times, events).backward()
        two_pass_backward(second, list(xs), times, events, torch.device('cpu'))
        for p, q in zip(first.parameters(), second.parameters()):
            torch.testing.assert_close(p.grad, q.grad, rtol=1e-5, atol=1e-6)

    def test_breslow_is_invariant_to_order_of_ties(self):
        risk = torch.tensor([.2, .5, -.8])
        times = torch.tensor([1., 1., 2.])
        events = torch.tensor([1., 0., 1.])
        order = torch.tensor([1, 0, 2])
        torch.testing.assert_close(cox_ph_loss(risk, times, events),
                                   cox_ph_loss(risk[order], times[order], events[order]))
    def test_pair_matrix_handles_censoring_and_ties(self):
        rng = np.random.default_rng(4)
        for _ in range(30):
            times = rng.integers(1, 6, 20)
            events = rng.integers(0, 2, 20)
            risks = rng.integers(0, 4, 20)
            denominator, numerator = comparable_scores(times, events, risks)
            self.assertAlmostEqual(numerator.sum() / denominator.sum(),
                                   safe_cindex(times, risks, events))

    def test_geometry_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            p, q = Path(folder) / 'i.nii.gz', Path(folder) / 'm.nii.gz'
            image = sitk.GetImageFromArray(np.zeros((4, 5, 6), np.int16))
            mask = sitk.GetImageFromArray(np.zeros((4, 5, 6), np.uint8))
            mask.SetOrigin((1., 0., 0.))
            sitk.WriteImage(image, str(p))
            sitk.WriteImage(mask, str(q))
            with self.assertRaisesRegex(ValueError, 'Origin'):
                aligned_arrays(p, q)

    def test_margin_uses_spacing_and_clips_bounds(self):
        labels = np.zeros((30, 40, 50), np.uint8)
        labels[10:12, 15:17, 20:22] = 38
        box = renal_box(labels, [5., 2., 1.])
        self.assertEqual([(s.start, s.stop) for s in box], [(8, 14), (10, 22), (10, 32)])

    def test_texture_excludes_background_and_area_has_physical_units(self):
        mask = np.zeros((10, 10), bool)
        mask[2:8, 2:8] = True
        first = np.full((10, 10), 50., np.float32)
        second = first.copy()
        second[~mask] = 250.
        a, b = [plane_features(x, mask, np.array([2., 3.])) for x in (first, second)]
        np.testing.assert_allclose(a, b)
        self.assertEqual(a[7], 36 * 6)
        self.assertEqual(a[10], 0.)
        self.assertEqual(a[11], 1.)
        self.assertEqual(a[12], 1.)

    def test_bilateral_summary_is_label_invariant(self):
        labels = np.zeros((3, 10, 10), np.uint8)
        labels[1, 1:4, 1:4] = 38
        labels[2, 5:9, 5:9] = 39
        volume = np.arange(300).reshape(3, 10, 10).astype(np.float32)
        swapped = np.where(labels == 38, 39, np.where(labels == 39, 38, 0))
        np.testing.assert_allclose(radiomics_2d(volume, labels, np.ones(3)),
                                   radiomics_2d(volume, swapped, np.ones(3)))


if __name__ == '__main__':
    unittest.main()
