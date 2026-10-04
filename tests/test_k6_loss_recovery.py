import unittest

import torch

from models.hivt_loss_recovery import HiVTLossRecovery
from models.hivt_nuscenes import HiVTNuScenesVehicle
from tests.test_hivt_stage_two import graph


class RecoveryLossTest(unittest.TestCase):
    def test_architecture_and_original_loss_unchanged(self):
        torch.manual_seed(2022);old=HiVTNuScenesVehicle()
        torch.manual_seed(2022);new=HiVTLossRecovery()
        self.assertEqual(old.state_dict().keys(),new.state_dict().keys())
        for key,value in old.state_dict().items():torch.testing.assert_close(value,new.state_dict()[key])
        data=graph();data.y[:]=3.
        raw=torch.ones(6,2,12,4)
        output={"raw_prediction":raw,"mode_logits":torch.randn(2,6),"rotation":torch.eye(2).repeat(2,1,1)}
        a=old.loss(output,data);b=new.recovery_loss(output,data,"original_nll")
        for key in a:torch.testing.assert_close(a[key],b[key])

    def test_fixed_location_keeps_mode_classification_and_target_mask(self):
        model=HiVTLossRecovery();data=graph();data.y[:]=3.;data.target_mask[1]=False
        raw=torch.ones(6,2,12,4);raw[0,...,:2]=2.;raw.requires_grad_()
        logits=torch.zeros(2,6,requires_grad=True)
        output={"raw_prediction":raw,"mode_logits":logits,"rotation":torch.eye(2).repeat(2,1,1)}
        values=model.recovery_loss(output,data,"fixed_scale")
        self.assertAlmostEqual(float(values["regression_loss"]),1.,places=6)
        torch.testing.assert_close(values["classification_loss"],model.loss(output,data)["classification_loss"])
        values["loss"].backward()
        self.assertGreater(torch.count_nonzero(logits.grad[0]).item(),0)
        self.assertEqual(torch.count_nonzero(logits.grad[1]).item(),0)
        self.assertEqual(torch.count_nonzero(raw.grad[...,2:]).item(),0)
        self.assertEqual(torch.count_nonzero(raw.grad[:,1]).item(),0)
        self.assertGreater(torch.count_nonzero(raw.grad[0,0,:,:2]).item(),0)

    def test_bounded_scale_clamps_only_uncertainty(self):
        model=HiVTLossRecovery();data=graph();data.y[:]=3.
        raw=torch.ones(6,2,12,4);raw[...,2:]=10.;raw.requires_grad_()
        output={"raw_prediction":raw,"mode_logits":torch.zeros(2,6),"rotation":torch.eye(2).repeat(2,1,1)}
        values=model.recovery_loss(output,data,"bounded_scale",2.)
        self.assertAlmostEqual(float(values["regression_loss"]),torch.log(torch.tensor(4.)).item()+1.,places=6)
        values["loss"].backward()
        self.assertEqual(torch.count_nonzero(raw.grad[...,2:]).item(),0)
        self.assertGreater(torch.count_nonzero(raw.grad[0,...,:2]).item(),0)
        self.assertTrue(torch.equal(raw.detach()[...,2:],torch.full_like(raw.detach()[...,2:],10.)))


class TinyAcceptanceTest(unittest.TestCase):
    def test_good_overall_cannot_hide_failed_moving_subgroup(self):
        from scripts.k6_baseline import tiny_gate, baseline_config
        reference={"Original HiVT free-scale failed run":{"metrics":{"full_horizon":{
            "overall":{"count":236,"minADE":5.57,"minFDE":11.27},
            "vehicle.moving":{"count":63,"minADE":19.11,"minFDE":38.53}}}}}
        result={"metrics":{"full_horizon":{"overall":{"count":236,"minADE":.7,"minFDE":1.7},
                 "vehicle.moving":{"count":63,"minADE":19.,"minFDE":38.}}}}
        gate=baseline_config()["tiny"]["gate"]
        self.assertFalse(tiny_gate(result,reference,gate))
        result["metrics"]["full_horizon"]["vehicle.moving"].update(minADE=2.3,minFDE=5.8)
        self.assertTrue(tiny_gate(result,reference,gate))
        result["metrics"]["full_horizon"]["vehicle.moving"]["count"]=62
        self.assertFalse(tiny_gate(result,reference,gate))


if __name__=="__main__":unittest.main()
