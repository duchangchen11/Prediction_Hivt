"""Nonneutral trained-tiny audit of target direction, broadcast and no leakage."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'00_manifest'))
from stage4f_common import model_new,SceneDataset,read_json,atomic_json,sha256,model_input
import torch
from torch_geometric.data import Batch


def main():
    checkpoint=ROOT/'07_checkpoints/stage4f_tiny_last.pt'
    model=model_new('cpu');model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['state_dict']);model.eval()
    ds=SceneDataset('train');graphs=[ds[r['index']] for r in read_json(ROOT/'00_manifest/stage4f_tiny_selection.json')['windows']];ds.clear()
    data=Batch.from_data_list(graphs);original=data.edge_index.clone();gi=model.global_interactor
    gi.record_necessity_gate=True;gi.record_relation_bias=True;layer_inputs=[]
    hooks=[layer.register_forward_pre_hook(lambda module,args:layer_inputs.append(args[3].detach().clone())) for layer in gi.global_interactor_layers]
    with torch.no_grad():out=model(model_input(data))
    for h in hooks:h.remove()
    gate=gi.necessity_gate_observation['gate'];relation=gi.relation_bias_observation
    source,target=relation['edge_index'];assert float(gate.max()-gate.min())>1e-4
    assert float((gate[source]-gate[target]).abs().max())>1e-4
    expected=gate[target,None,None]*relation['bias'];assert torch.equal(expected,relation['gated_bias'])
    assert len(layer_inputs)==3 and all(torch.equal(layer_inputs[l],expected[:,l,:]) for l in range(3))
    assert relation['bias'].abs().mean()>0 and torch.equal(data.edge_index,original)
    changed=data.clone();changed.positions[:,5:]+=999;changed.y-=555;changed.future_mask=~changed.future_mask;changed.padding_mask[:,5:]=~changed.padding_mask[:,5:]
    with torch.no_grad():out2=model(model_input(changed))
    second=gi.necessity_gate_observation;assert torch.equal(gate,second['gate'])
    diffs={k:float((out[k]-out2[k]).abs().max()) for k in ('raw_prediction','mode_logits','mode_prob')};assert all(v<1e-6 for v in diffs.values())
    atomic_json(ROOT/'04_evaluation/stage4f_target_gate_broadcast_audit.json',{'status':'PASS','backend':'CPU single thread',
        'own_tiny_checkpoint_sha256':sha256(checkpoint),'relation_bias_nonzero':True,'gate_nonconstant':True,
        'source_target_gate_difference_exists':True,'all3_layer_inputs_bitwise_equal_to_target_gate_times_bias':True,
        'same_target_scalar_all_incoming_edges_all8heads_all3layers':True,'future_counterfactual_gate_bitwise_equal':True,
        'future_counterfactual_outputs_max_abs_diff':diffs,'edge_index_unchanged':True,'optimizer_updates':0,
        'formal_training_process_unmodified':True})
    print('TARGET_GATE_BROADCAST_AND_TRAINED_NO_LEAKAGE_PASS',flush=True)


if __name__=='__main__':torch.set_num_threads(1);main()
