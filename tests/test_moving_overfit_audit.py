"""Check diagnostic supervision and uncertainty intervention independently."""
import unittest

import torch

from models.hivt_nuscenes import HiVTNuScenesVehicle
from scripts.moving_overfit_audit import diagnostic_loss, selected_graphs
from tests.test_hivt_stage_two import graph


class MovingAuditTest(unittest.TestCase):
    def test_context_preserved_and_only_selected_prediction_supervised(self):
        source = graph()
        source.history_mask = torch.ones(2,5,dtype=torch.bool)
        source.instance_tokens = ["moving", "context"]
        selected = selected_graphs([source], [{"graph_index":0,"node":0,"instance_token":"moving"}])[0]
        self.assertEqual(selected.target_mask.tolist(), [True,False])
        self.assertEqual(source.target_mask.tolist(), [True,True])
        self.assertEqual(selected.num_nodes,2)
        torch.testing.assert_close(selected.lane_vectors,source.lane_vectors)
        torch.testing.assert_close(selected.edge_index,source.edge_index)
        model = HiVTNuScenesVehicle()
        source.y[:] = 3.
        selected.y[:] = 3.
        raw = torch.ones(6,2,12,4,requires_grad=True)
        logits = torch.zeros(2,6,requires_grad=True)
        output={"raw_prediction":raw,"mode_logits":logits,"rotation":torch.eye(2).repeat(2,1,1)}
        values = model.loss(output,selected)
        # Unit scale, |3-1|=2, exact official coordinate-wise Laplace NLL.
        self.assertAlmostEqual(float(values["regression_loss"]),2.+torch.log(torch.tensor(2.)).item(),places=6)
        self.assertAlmostEqual(float(values["classification_loss"]),torch.log(torch.tensor(6.)).item(),places=6)
        values["loss"].backward()
        self.assertEqual(torch.count_nonzero(raw.grad[:,1]).item(),0)
        self.assertGreater(torch.count_nonzero(raw.grad[:,0]).item(),0)
        self.assertEqual(torch.count_nonzero(logits.grad[1]).item(),0)

    def test_fixed_scale_location_gradient_independent_of_scale_prediction(self):
        data=graph(); data.target_mask[1]=False; data.y[:]=3.
        class Model:
            num_modes=1
        gradients=[]
        losses=[]
        for scale in (1.,100.):
            raw=torch.ones(1,2,12,4)
            raw[...,2:]=scale; raw.requires_grad_()
            output={"raw_prediction":raw,"rotation":torch.eye(2).repeat(2,1,1)}
            values=diagnostic_loss(Model(),output,data,True)
            values["loss"].backward()
            gradients.append(raw.grad.clone()); losses.append(float(values["loss"]))
            self.assertEqual(torch.count_nonzero(raw.grad[...,2:]).item(),0)
            self.assertEqual(torch.count_nonzero(raw.grad[:,1]).item(),0)
        self.assertEqual(losses[0],losses[1])
        torch.testing.assert_close(gradients[0],gradients[1])
        self.assertAlmostEqual(float(gradients[0][0,0,0,0]),-1./24.,places=7)

    def test_official_nll_location_gradient_has_inverse_scale_factor(self):
        data=graph(); data.target_mask[1]=False; data.y[:]=3.
        model=HiVTNuScenesVehicle(num_modes=1)
        gradient=[]
        for scale in (1.,10.):
            raw=torch.ones(1,2,12,4); raw[...,2:]=scale; raw.requires_grad_()
            output={"raw_prediction":raw,"rotation":torch.eye(2).repeat(2,1,1),"mode_logits":torch.zeros(2,1)}
            model.loss(output,data)["loss"].backward()
            gradient.append(float(raw.grad[0,0,0,0]))
        self.assertAlmostEqual(gradient[0]/gradient[1],10.,places=5)


if __name__ == "__main__":
    unittest.main()
