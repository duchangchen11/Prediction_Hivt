"""Static parameters, empty-map tests and no-grad neutral/forward memory audits."""
from pathlib import Path
import sys,gc,hashlib
from unittest.mock import patch
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/d) for d in ('00_manifest','01_selector','03_model_audit')]
from stage8a0c_common import *
from stage8a0c_selector import SparseSemanticIndex
from stage8a0c_model import SparseGraphReranker,MapAggregator
from stage8a_model import GraphReranker as HistoricalPrototype

def digest(t):return hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest()

@torch.no_grad()
def main():
    torch.set_num_threads(2);torch.manual_seed(2022);torch.use_deterministic_algorithms(True)
    maps=SparseSemanticIndex();location=next(iter(maps.regions))
    far=np.stack((np.linspace(1e6,1e6+5,12),np.full(12,1e6)),axis=-1)
    empty=maps.select_sparse_semantic_entities(0,far,location)
    assert not any(len(v) for v in empty.values())
    aggregator=MapAggregator().eval().requires_grad_(False)
    for module in aggregator.modules():
        if isinstance(module,torch.nn.Linear):torch.nn.init.constant_(module.bias,10.)
    h=torch.randn(2,6,64);raw=torch.randn(2,6,15)
    with patch.object(torch.Tensor,'softmax',side_effect=AssertionError('empty map must not call softmax')):
        message=aggregator(h,raw,torch.empty(2,6,0,18),torch.empty(2,6,0,11),torch.empty(2,6,0,dtype=torch.bool))
        padded=aggregator(h,raw,torch.randn(2,6,8,18),torch.randn(2,6,8,11),torch.zeros(2,6,8,dtype=torch.bool))
    assert torch.equal(message,torch.zeros_like(h)) and torch.equal(padded,message)
    norm=torch.nn.LayerNorm(64);mi=torch.randn_like(h);joint=norm(h+mi+message)
    assert torch.isfinite(joint).all() and torch.equal(joint,norm(h+mi))
    mask=torch.zeros(2,6,8,dtype=torch.bool);mask[1,0,0]=True
    mixed=aggregator(h,raw,torch.randn(2,6,8,18),torch.randn(2,6,8,11),mask)
    assert torch.equal(mixed[~mask.any(-1)],torch.zeros_like(mixed[~mask.any(-1)]))
    assert torch.isfinite(mixed).all()
    atomic_json(ROOT/'03_model_audit/stage8a0c_empty_map_audit.json',{'status':'PASS','synthetic_candidate_returns_no_entities':True,
        'zero_length_edge_tensor_exact_zero64':True,'padded_empty_rows_exact_zero64':True,'nonzero_encoder_bias_cannot_affect_empty_map':True,
        'softmax_called_on_empty_rows':False,'mixed_empty_nonempty_rows_finite':True,'LayerNorm_hnode_plus_mint_plus_zero':'PASS',
        'NULL_node':False,'nearest_fallback':False,'empty_samples_deleted':False})
    with np.load(ROOT/'02_graph_cache/stage8a0c_neutral_arguments.npz') as z:
        arguments=tuple(torch.from_numpy(z[f'arg{k}'].copy()) for k in range(7));prob=torch.from_numpy(z['original_probability'].copy())
    assert all(len(a)==128 for a in arguments) and prob.shape==(128,6)
    models={v:SparseGraphReranker(v).eval().requires_grad_(False) for v in ('G1','G2','G3')}
    historical=HistoricalPrototype('G1').eval().requires_grad_(False)
    assert models['G1'].state_dict().keys()==historical.state_dict().keys()
    for key,value in historical.state_dict().items():assert torch.equal(value,models['G1'].state_dict()[key])
    del historical
    shared=[];rows=[];neutral_rows=[];memory_rows=[]
    for variant,model in models.items():
        params=sum(p.numel() for p in model.parameters());assert params<150000
        common={k:digest(v) for k,v in model.state_dict().items() if k.startswith(('node_encoder.','norm.','head.'))}
        shared.append(common);assert not model.head[-1].weight.any() and not model.head[-1].bias.any()
        out=model(*arguments)
        logit_diff=float((out['mode_logits']-arguments[-1]).abs().max());prob_diff=float((out['mode_prob']-prob).abs().max())
        assert logit_diff==0 and prob_diff<1e-6 and not out['delta_logits'].any()
        assert not any(p.grad is not None for p in model.parameters())
        neutral_rows.append({'Variant':variant,'Device':'CPU','RealUniqueTargets':128,'LogitMaxDiff':logit_diff,'ProbabilityMaxDiff':prob_diff})
        # Also verify a whole graph with zero map slots (not merely padding).
        empty_args=list(arguments);empty_args[3]=torch.empty(128,6,0,18);empty_args[4]=torch.empty(128,6,0,11);empty_args[5]=torch.empty(128,6,0,dtype=torch.bool)
        zero_out=model(*empty_args);assert not zero_out['map_message'].any() and torch.equal(zero_out['mode_logits'],arguments[-1])
        estimated=None;peak=None
        if torch.cuda.is_available():
            gpu=SparseGraphReranker(variant).eval().requires_grad_(False).cuda();inputs=tuple(a.cuda() for a in arguments)
            gpu(*inputs);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            resident=torch.cuda.memory_allocated();output=gpu(*inputs);torch.cuda.synchronize();peak=torch.cuda.max_memory_allocated()/2**20
            gd=float((output['mode_logits']-inputs[-1]).abs().max());pd=float((output['mode_prob'].cpu()-prob).abs().max())
            assert gd==0 and pd<1e-6 and not any(p.grad is not None for p in gpu.parameters())
            neutral_rows.append({'Variant':variant,'Device':'CUDA','RealUniqueTargets':128,'LogitMaxDiff':gd,'ProbabilityMaxDiff':pd})
            raw_bytes=sum(a.numel()*a.element_size() for a in inputs)
            # Fixed planning margin for no-grad inference; training is explicitly unmeasured.
            estimated=peak+32.
            memory_rows.append({'Variant':variant,'MeasuredForwardPeakMiB':peak,'ResidentBeforeForwardMiB':resident/2**20,
                'InputBytesBatch128':raw_bytes,'EstimatedInferencePeakMiB':estimated,'MarginMiB':32.,'TrainingMemoryMeasured':False})
            del gpu,inputs,output;gc.collect();torch.cuda.empty_cache()
        else:
            historical_peak=max(r['PeakCUDAAllocatedMiB'] for r in read_json(STAGE8/'03_model_audit/stage8a_memory_audit.json')['rows'])
            estimated=historical_peak+32.
            memory_rows.append({'Variant':variant,'MeasuredForwardPeakMiB':None,'ResidentBeforeForwardMiB':None,
                'InputBytesBatch128':sum(a.numel()*a.element_size() for a in arguments),
                'EstimatedInferencePeakMiB':estimated,'MarginMiB':32.,'TrainingMemoryMeasured':False})
        rows.append({'Variant':variant,'Parameters':params,'InputDims':'Node15; Interaction17; MapNode18; MapEdge11',
            'HiddenDim':64,'MaxMapEdges':0 if variant=='G1' else 48,'MaxInteractionEdges':0 if variant=='G2' else 288,
            'NeutralOutputMaxDiff':logit_diff,'EstimatedMemoryMiB':estimated})
    assert all(s==shared[0] for s in shared)
    write_csv(ROOT/'06_tables/stage8a0c_model_spec.csv',rows);write_csv(ROOT/'06_tables/stage8a0c_neutral_output.csv',neutral_rows)
    write_csv(ROOT/'06_tables/stage8a0c_memory.csv',memory_rows)
    feasible=max(r['EstimatedMemoryMiB'] for r in rows)<=1024.
    atomic_json(ROOT/'03_model_audit/stage8a0c_model_audit.json',{'status':'PASS','variants':rows,'shared_node_norm_head_bitwise_equal':True,
        'G1_bitwise_identical_to_Stage8A0_static_prototype':True,'NeutralLogitMaxDiff':max(r['LogitMaxDiff'] for r in neutral_rows),
        'NeutralProbabilityMaxDiff':max(r['ProbabilityMaxDiff'] for r in neutral_rows),'final_head_exact_zero':True,
        'real_unique_actors':128,'optimizer_steps':0,'backward_calls':0,'new_checkpoints':0,'training_executed':False,
        'parameters_require_grad_during_audit':False})
    atomic_json(ROOT/'05_memory/stage8a0c_memory_audit.json',{'MemoryFeasible':'YES' if feasible else 'NO','rows':memory_rows,
        'EstimatedInferenceMiB':max(r['EstimatedMemoryMiB'] for r in rows),'limit_MiB':1024,
        'dimension_exact_input_bytes_per_target':sum(a.numel()*a.element_size() for a in arguments)//128,
        'batch128_input_bytes':sum(a.numel()*a.element_size() for a in arguments),'method':'allowed static no-grad forward peak plus fixed32MiB planning margin; historical fallback if CUDA unavailable',
        'training_memory':'unmeasured; requires Stage8A-1 tiny audit under separate authorization','includes_predictor':False})
    print('MODEL EMPTY/NEUTRAL/MEMORY PASS',rows,flush=True)

if __name__=='__main__':main()
