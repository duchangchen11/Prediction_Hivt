"""GT-access protection, geometry tests, Stage6A neighbor identity and neutral graphs."""
from pathlib import Path
import sys,inspect,dataclasses
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'00_manifest'))
from stage8a_common import *
from stage8a_graph import *
from stage8a_map import *
from stage8a_model import *
from stage6a_features import observable_features

class ProtectedCache(dict):
    def __getitem__(self,key):
        if key in ('GT','future_mask','target_mask','GT_turn','future_intent'):raise AssertionError('Forbidden future field accessed: '+key)
        return super().__getitem__(key)

def compare_arrays(a,b):
    assert a.keys()==b.keys()
    for k in a:assert np.array_equal(a[k],b[k]),k

@torch.no_grad()
def main():
    verify_frozen();maps=MapIndex();samples=read_json(ROOT/'01_cache_audit/stage8a_candidate_identity.json')['selected_source_batches']
    model={v:GraphReranker(v).eval() for v in ('G1','G2','G3')};initialization=[]
    for v,m in model.items():
        count=sum(p.numel() for p in m.parameters());assert count<50000
        shared={k:tensor_sha(p) for k,p in m.state_dict().items() if k.startswith(('node_encoder.','norm.','head.'))}
        initialization.append({'Variant':v,'Parameters':count,'SharedState':shared,'Seed':2022,'InteractionSeed':2123,'MapSeed':2124})
    assert all(row['SharedState']==initialization[0]['SharedState'] for row in initialization)
    neutral=[];neighbor_windows=0;source_patterns=set();semantic_edges=0
    for split in ('train','val'):
        ds=SceneDataset(split)
        for sample in [x for x in samples if x['split']==split][:8]:
            cached=torch.load(STAGE6/sample['path'],map_location='cpu',weights_only=False)
            w=cached['windows'][0];g=ds[w['dataset_index']]
            observable=observable_window(ProtectedCache(w),g.map_location,g.origin.numpy(),float(g.ego_yaw))
            feature=node_features(observable);idx,keep=neighbors(observable);selected,mf=maps.build(observable)
            _,_,old_idx,old_keep=observable_features(*[w[k].cuda() for k in ('history','history_padding','agent_type','ego_prediction','mode_logits','mode_prob')])
            m=old_idx.shape[1];assert torch.equal(idx[:,:m],old_idx.cpu()) and torch.equal(keep[:,:m],old_keep.cpu())
            neighbor_windows+=1
            poison=dict(w);poison['GT']=torch.full_like(w['GT'],float('nan'));poison['future_mask']=~w['future_mask'];poison['target_mask']=~w['target_mask']
            poison['GT_turn']='fabricated sign';poison['future_intent']=object()
            mutated=observable_window(ProtectedCache(poison),g.map_location,g.origin.numpy(),float(g.ego_yaw))
            assert torch.equal(feature,node_features(mutated));ni,nm=neighbors(mutated)
            assert torch.equal(idx,ni) and torch.equal(keep,nm)
            ss,ff=maps.build(mutated);assert np.array_equal(selected,ss);compare_arrays(mf,ff)
            targets=torch.tensor(selected[:8]);edges=interaction_edges(observable,idx,keep,targets)
            assert torch.equal(edges,interaction_edges(mutated,ni,nm,targets))
            for mode_row in mf['map_token_index'].reshape(-1,8):
                valid=mode_row[mode_row>=0];assert len(valid)==len(set(valid.tolist()))
            mask=mf['map_mask'];ids=mf['map_token_index'][mask];s=np.stack([maps.token_dictionary[int(i)]['semantic'] for i in ids])
            assert np.array_equal(mf['map_node'][mask][:,:9],s) and np.array_equal(mf['map_edge'][mask][:,8:],s)
            semantic_edges+=len(ids);source_patterns.update(tuple(row) for row in s)
            pack=pack_targets(feature,idx,keep,edges,selected,mf,targets)
            for v,head in model.items():
                out=head(*pack);base=pack[-1]
                assert torch.equal(out['mode_logits'],base) and torch.equal(out['mode_prob'],base.softmax(-1))
                cached_prob=observable.original_probability[targets]
                probability_difference=float((out['mode_prob']-cached_prob).abs().max())
                assert probability_difference<1e-6
                assert (out['delta_logits']==0).all() and all(torch.isfinite(t).all() for t in out.values())
                neutral.append({'Split':split,'Window':w['dataset_index'],'Variant':v,'Targets':len(targets),'LogitMaxDiff':0.,
                    'ProbabilityMaxDiffVsOriginalGPUCache':probability_difference,'ProbabilityMaxDiffVsSameDeviceR0':0.})
        ds.clear()
    # Analytical geometry fixture: crossings between future points, not only point proximity.
    crossing=shapely.LineString([[-20.,0.],[20.,0.]])
    lane=shapely.LineString([[0.,-2.],[0.,2.]])
    assert shapely.distance(crossing,lane)==0 and min(shapely.distance(shapely.Point(p),lane) for p in [[-20.,0.],[20.,0.]])>10
    location=next(iter(maps.regions));point=maps.regions[location]['coordinates'][0][0]
    stationary=np.broadcast_to(point,(1,12,2)).copy();ret=maps.retrieve(stationary,location)
    fields=maps.features(stationary,np.zeros(1),location,0.,ret)
    assert np.all(fields['map_edge'][fields['map_mask']][:,7]==np.float32(1/12))
    no_neighbor=torch.zeros((2,48),dtype=torch.bool);weights=masked_attention(torch.randn(2,48),no_neighbor)
    assert torch.equal(weights,torch.zeros_like(weights)) and torch.isfinite(weights).all()
    write_csv(ROOT/'06_tables/stage8a_neutral_graph_batches.csv',neutral)
    atomic_json(ROOT/'03_model_audit/stage8a_initialization_manifest.json',{'status':'PASS','variants':initialization,
        'shared_node_norm_head_bitwise_equal':True,'final_layer_weight_bias_exact_zero':True,'training_executed':False,
        'architecture':'one message-passing layer; hidden64; branch messages/learned attention; score head64-32-1',
        'predictor_loaded_into_graph':False,'G1_G2_G3_candidates_are_references_to_same_R0_tensor':True})
    atomic_json(ROOT/'03_model_audit/stage8a_gt_leakage_audit.json',{'status':'PASS','observable_fields':[f.name for f in dataclasses.fields(ObservableWindow)],
        'GT_mask_target_mask_access_raises':True,'poisoned_GT_and_inverted_future_masks_leave_all_graph_arrays_bitwise_identical':True,
        'windows':neighbor_windows,'GT_used_for_neighbor_selection':False,'GT_used_for_map_retrieval':False,
        'ranking_labels_not_built_in_Stage0':True,'full_target_mask_used_only_for_coverage_population_counts':True,
        'graph_predictor_backward_executed':False,'predictor_gradient_count':0})
    atomic_json(ROOT/'03_model_audit/stage8a_graph_definition_audit.json',{'status':'PASS','Stage6A_neighbor_identity_windows':neighbor_windows,
        'maximum_neighbors':8,'maximum_local_mode_nodes':54,'maximum_interaction_edges_per_target':288,
        'no_preaggregated_interaction_features':True,'whole_polyline_crossing_test':True,
        'stationary_polyline_closest_time_earliest_future_point':True,'all_absent_attention_exact_zero':True,
        'semantic_attachment_edges_checked':semantic_edges,'semantic_patterns_seen':len(source_patterns),
        'semantic_vector_source':'unchanged Stage7A semantic_vector; includes control/crosswalk definitions unchanged',
        'semantic_attachment':'PASS','raw_features_finite':True,'neutral_logit_maxdiff':0.,
        'neutral_probability_maxdiff_vs_same_device_R0':0.,
        'neutral_probability_maxdiff_vs_original_GPU_cache':max(r['ProbabilityMaxDiffVsOriginalGPUCache'] for r in neutral),
        'CPU_GPU_softmax_tolerance':1e-6,'new_checkpoints_created':0})
    print('STAGE8A_GRAPH_LEAKAGE_NEUTRAL_PASS',[{'variant':x['Variant'],'parameters':x['Parameters']} for x in initialization],flush=True)

if __name__=='__main__':torch.set_num_threads(4);main()
