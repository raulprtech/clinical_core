import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import copy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from build_renal_2p5d_program_cache import aligned_arrays, renal_box, plane_features, radiomics_2d
from evaluate_renal_2p5d_program import comparable_scores, cox_nested
from evaluate_resnet_sequence_models import safe_cindex
from evaluate_renal_resnet_adaptation import two_pass_backward, cox_ph_loss, new_model, train_epoch
from build_fullfield_adaptation_cache import selected_centers
from build_renal_letterbox_cache import square_pad
from verify_renal_2p5d_program import verify_reused_risks
from evaluate_stunet_trimodal_pooling_nested_cv import fit_outer_modality


class RenalProgramTests(unittest.TestCase):
    def test_reused_risk_audit_allows_roundoff_but_rejects_changed_ties(self):
        left=pd.DataFrame({'case_id':['a','b','c'],'model':['x']*3,'repeat':[0]*3,'fold':[0]*3,'risk':[1.,2.,3.]})
        right=left.copy()
        right['risk']=np.nextafter(right.risk.to_numpy(),np.inf)
        audit=verify_reused_risks(left,right,3)
        self.assertTrue(audit['within_fold_ranks_and_ties_unchanged'])
        self.assertFalse(audit['exact'])
        left['risk']=[1.,1.,2.]
        right=left.copy()
        right.loc[1,'risk']=np.nextafter(1.,np.inf)
        with self.assertRaises(AssertionError):
            verify_reused_risks(left,right,3)

    def test_letterbox_preserves_pixels_and_only_adds_zero_padding(self):
        for height,width in [(4,8),(8,4),(5,8),(7,7)]:
            image=torch.arange(3*height*width,dtype=torch.float32).reshape(3,height,width)+1
            padded=square_pad(image)
            side=max(height,width)
            top,left=(side-height)//2,(side-width)//2
            self.assertEqual(padded.shape,(3,side,side))
            torch.testing.assert_close(padded[:,top:top+height,left:left+width],image,rtol=0,atol=0)
            self.assertEqual(int(torch.count_nonzero(padded)),image.numel())

    @staticmethod
    def synthetic_cohort():
        rng = np.random.default_rng(47)
        values = rng.normal(size=(48,12))
        times = rng.uniform(1,100,48)
        events = np.tile([0,1],24)
        train, heldout = np.arange(36), np.arange(36,48)
        return values, times, events, train, heldout

    def test_cox_selection_ignores_heldout_outcomes(self):
        values, times, events, train, heldout = self.synthetic_cohort()
        first, a = cox_nested(values,times,events,train,heldout,4049)
        times[heldout] = times[heldout][::-1]*10
        events[heldout] = 1-events[heldout]
        second, b = cox_nested(values,times,events,train,heldout,4049)
        np.testing.assert_array_equal(first,second)
        self.assertEqual(a,b)

    def test_fusion_modality_ignores_heldout_outcomes(self):
        values, times, events, train, heldout = self.synthetic_cohort()
        first = fit_outer_modality(values,times,events,train,heldout,'vision',4049,[4],[1.],3)
        times[heldout] = times[heldout][::-1]*10
        events[heldout] = 1-events[heldout]
        second = fit_outer_modality(values,times,events,train,heldout,'vision',4049,[4],[1.],3)
        np.testing.assert_array_equal(first[0],second[0])
        np.testing.assert_array_equal(first[1],second[1])
        self.assertEqual(first[2],second[2])

    def test_adaptation_keeps_batchnorm_frozen(self):
        torch.manual_seed(47)
        block = torch.nn.Sequential(torch.nn.Conv2d(3,512,1),torch.nn.BatchNorm2d(512),torch.nn.ReLU())
        xs = [torch.randn(2,3,2,2) for _ in range(4)]
        for tune in (False,True):
            model, optimizer = new_model(block,tune,47,torch.device('cpu'))
            before = {k:v.clone() for k,v in model.layer4.state_dict().items()}
            head = model.head.weight.detach().clone()
            train_epoch(model,optimizer,xs,np.array([1.,2.,3.,4.]),np.array([1,0,1,0]),torch.device('cpu'))
            after = model.layer4.state_dict()
            for k in before:
                if not tune or k.startswith('1.'):
                    torch.testing.assert_close(before[k],after[k],rtol=0,atol=0)
            self.assertFalse(torch.equal(head,model.head.weight))
            if tune:
                self.assertFalse(torch.equal(before['0.weight'],after['0.weight']))

    def test_fullfield_centers_match_existing_image_subsampling(self):
        for n in (1, 5, 16, 32, 64):
            centers = np.arange(n)*2+3
            expected = centers[np.linspace(0,n-1,min(16,n),dtype=int)]
            np.testing.assert_array_equal(selected_centers(centers),expected)
        with self.assertRaises(ValueError):
            selected_centers([])
        with self.assertRaises(ValueError):
            selected_centers([[1,2]])

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
