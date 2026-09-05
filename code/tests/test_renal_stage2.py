import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from evaluate_stage2_joint_adaptation import JointMamba, select_epoch
from evaluate_renal_resnet_adaptation import two_pass_backward, cox_ph_loss


class Stage2Tests(unittest.TestCase):
    def test_joint_mamba_two_pass_gradient_matches_monolithic(self):
        torch.set_num_threads(1)
        torch.manual_seed(47)
        block=torch.nn.Sequential(torch.nn.Conv2d(3,512,1),torch.nn.BatchNorm2d(512),torch.nn.ReLU())
        first=JointMamba(block,True).eval()
        second=copy.deepcopy(first)
        xs=[torch.randn(2,3,2,2) for _ in range(3)]
        times,events=np.array([1.,2.,2.]),np.array([1.,1.,0.])
        risks=torch.stack([first(x) for x in xs])
        cox_ph_loss(risks,torch.tensor(times),torch.tensor(events)).backward()
        two_pass_backward(second,xs,times,events,torch.device('cpu'))
        for a,b in zip(first.parameters(),second.parameters()):
            if a.requires_grad:
                torch.testing.assert_close(a.grad,b.grad,rtol=1e-4,atol=1e-6)
        self.assertTrue(all(m.p==0 for m in first.modules() if isinstance(m,torch.nn.Dropout)))

    def test_early_stop_respects_minimum_and_records_best_not_last(self):
        with patch('evaluate_stage2_joint_adaptation.initialize',return_value=(object(),object())), \
             patch('evaluate_stage2_joint_adaptation.train_epoch',return_value=3.) as train, \
             patch('evaluate_stage2_joint_adaptation.predict',return_value=np.array([0.])), \
             patch('evaluate_stage2_joint_adaptation.safe_cindex',side_effect=[.5]+[.6]*19):
            result=select_epoch(None,False,'linear',[0,1,2,999],np.arange(4)+1,np.ones(4),
                                np.array([0,1]),np.array([2]),47,'cpu',100,20,15)
        self.assertEqual(result['best_epoch'],2)
        self.assertEqual(result['epochs_run'],20)
        self.assertEqual(result['stop_reason'],'patience')
        self.assertEqual(train.call_count,20)
        self.assertTrue(all(call.args[2]==[0,1] for call in train.call_args_list))


if __name__=='__main__': unittest.main()
