"""Nonzero residual probes verify real broadcast and direct pi independence."""
from pathlib import Path
import sys
import copy
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'00_manifest'))
from stage5a_common import model_new, atomic_json, verify_previous
import torch


@torch.no_grad()
def main():
    torch.set_num_threads(1);verify_previous()
    model=model_new('cpu').eval();decoder=copy.deepcopy(model.decoder)
    # Constant, distinct expert outputs give an independent analytical oracle.
    decoder.experts[0][-1].bias.fill_(1.)
    decoder.experts[1][-1].bias.fill_(2.)
    hidden=torch.arange(6*3*64,dtype=torch.float32).reshape(6,3,64)/1000
    routing=torch.tensor([[.1,.9],[.75,.25],[.5,.5]])
    residual,experts=decoder.residual(hidden,routing)
    expected=torch.tensor([1.9,1.25,1.5])[None,:,None].expand(6,3,64)
    torch.testing.assert_close(residual,expected,rtol=0,atol=2e-7)
    assert experts[:,:,0].eq(1).all() and experts[:,:,1].eq(2).all()
    local=torch.arange(3*64,dtype=torch.float32).reshape(3,64)/100
    global_embed=hidden
    condition=torch.tensor([[1.,0.,0.,0.,0.,0.],[0.,1.,0.,.1,.5,.8],[0.,0.,1.,1.,2.,3.]])
    baseline_prediction,baseline_pi=model.decoder(local,global_embed,condition)
    changed_prediction,changed_pi=decoder(local,global_embed,condition)
    assert torch.equal(baseline_pi,changed_pi)
    assert (baseline_prediction-changed_prediction).abs().max()>0
    # This is a discarded numerical test fixture, never an optimized model.
    result={'status':'PASS','nonzero_experts_actor_broadcast_max_abs_diff':float((residual-expected).abs().max()),
        'pi_change_with_nonzero_residual':float((baseline_pi-changed_pi).abs().max()),
        'prediction_change_with_nonzero_residual':float((baseline_prediction-changed_prediction).abs().max()),
        'real_residual_function_tested':True,'same_actor_routing_all6_modes':True,
        'backward':False,'optimizer_updates':0,'fixture_discarded':True}
    atomic_json(ROOT/'00_manifest/stage5a_decoder_branch_audit.json',result)
    print('DECODER_BRANCH_AUDIT_PASS',result)


if __name__=='__main__':main()
