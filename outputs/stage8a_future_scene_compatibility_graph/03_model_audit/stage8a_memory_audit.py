"""Untrained prototype forward memory; no optimizer, backward or trained output claim."""
from pathlib import Path
import sys,gc
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'00_manifest'))
from stage8a_common import *
from stage8a_graph import *
from stage8a_map import MapIndex
from stage8a_model import *
from stage6a_features import observable_features,normalize
from stage6a_head import ReliabilityHead

@torch.no_grad()
def main():
    verify_frozen();maps=MapIndex();ds=SceneDataset('train')
    record=read_json(ROOT/'01_cache_audit/stage8a_candidate_identity.json')['selected_source_batches'][0]
    w=torch.load(STAGE6/record['path'],map_location='cpu',weights_only=False)['windows'][0];g=ds[w['dataset_index']]
    obs=observable_window(w,g.map_location,g.origin.numpy(),float(g.ego_yaw));node=node_features(obs);idx,keep=neighbors(obs)
    selected,mf=maps.build(obs);target=torch.tensor(selected[:8]);edge=interaction_edges(obs,idx,keep,target)
    arguments=pack_targets(node,idx,keep,edge,selected,mf,target)
    repeats=(128+len(target)-1)//len(target)
    arguments=tuple(a.repeat((repeats,)+(1,)*(a.ndim-1))[:128].contiguous() for a in arguments)
    assert all(len(a)==128 for a in arguments)
    rfeature,flags,_,_=observable_features(*[w[k] for k in ('history','history_padding','agent_type','ego_prediction','mode_logits','mode_prob')])
    rfeature=normalize(rfeature,flags,read_json(STAGE6/'02_features/stage6a_normalization.json'))[target]
    rfeature=rfeature.repeat(repeats,1,1)[:128].contiguous()
    geometry_before=tensor_sha(w['ego_prediction']);rows=[]
    for variant in ('R2','G1','G2','G3'):
        if variant=='R2':
            head=ReliabilityHead();head.load_state_dict(torch.load(R2,map_location='cpu',weights_only=False)['state_dict'],strict=True)
            inputs=(rfeature,arguments[-1]);head=head.cuda().eval()
        else:head=GraphReranker(variant).cuda().eval();inputs=arguments
        inputs=tuple(a.cuda() for a in inputs)
        head(*inputs);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        resident=torch.cuda.memory_allocated();output=head(*inputs);torch.cuda.synchronize()
        assert all(torch.isfinite(a).all() for a in output.values()) and not any(p.grad is not None for p in head.parameters())
        peak=torch.cuda.max_memory_allocated();raw_bytes=sum(a.numel()*a.element_size() for a in inputs)
        rows.append({'Model':variant,'Parameters':sum(p.numel() for p in head.parameters()),'TargetsPerForward':128,
            'InputTensorBytes':raw_bytes,'InputKiBPerTarget':raw_bytes/128/1024,'PeakCUDAAllocatedMiB':peak/1024**2,
            'ForwardIncrementalPeakMiB':(peak-resident)/1024**2,'TrainingExecuted':False,
            'Weights':'frozen trained R2' if variant=='R2' else 'zero-final-layer untrained prototype'})
        del inputs,output,head;gc.collect();torch.cuda.empty_cache()
    assert tensor_sha(w['ego_prediction'])==geometry_before
    write_csv(ROOT/'06_tables/stage8a_efficiency.csv',rows)
    atomic_json(ROOT/'03_model_audit/stage8a_memory_audit.json',{'status':'PASS','device':torch.cuda.get_device_name(),
        'rows':rows,'maximum_mode_nodes_per_target':54,'maximum_mode_mode_edges_per_target':288,'maximum_mode_map_edges_per_target':48,
        'dense_padded_float32_input_bytes_per_target':rows[-1]['InputTensorBytes']//128,
        'inference_only':True,'autograd_training_activation_memory_measured':False,'no_latency_claim':True,
        'predictor_latency_not_included':True,'graph_retrieval_construction_not_included':True,
        'shared_input_candidates_unchanged':True,'new_checkpoint_count':0,'optimizer_steps':0})
    ds.clear();print('STAGE8A_MEMORY_PASS',rows,flush=True)

if __name__=='__main__':torch.set_num_threads(4);main()
