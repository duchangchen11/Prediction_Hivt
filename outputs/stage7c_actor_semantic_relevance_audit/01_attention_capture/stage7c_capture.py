"""One paired VAL pass: unmodified predictors, attention archives, offline GT analysis."""
from pathlib import Path
import sys,collections,time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'00_manifest'),str(ROOT/'02_relevance_analysis')]
from stage7c_common import *
from stage7c_hooks import LaneCapture
from stage7c_geometry import (segments,future_distances,point_segment_distances,
    turn_membership,TurnAmbiguity,numerical_distance_checks)
from torch_geometric.data import Batch

def attention_metrics(lanes,alpha,dist_all,sem,turning,heading):
    mean=alpha.astype(np.float64).mean(axis=1)
    dense=np.zeros(len(dist_all),dtype=np.float64);dense[lanes]=mean
    nearest=int(np.argmin(dist_all)) if len(dist_all) else -1
    a=dense[nearest] if nearest>=0 else np.nan
    def rank(weight):
        return float((dense>weight).sum()+1+((dense==weight).sum()-1)/2)
    rank_nearest=rank(a) if nearest>=0 else np.nan
    tied=np.flatnonzero(dist_all<=dist_all.min()+1e-8) if len(dist_all) else np.array([],int)
    best_tie_weight=float(dense[tied].max()) if len(tied) else np.nan
    nearest_in_edges=bool(nearest in lanes)
    order=np.lexsort((lanes,-mean))
    d=dist_all[lanes]
    ss=sem[lanes]
    incoming_min=int(np.argmin(d)) if len(d) else -1
    incoming_rank=float((mean>mean[incoming_min]).sum()+1+((mean==mean[incoming_min]).sum()-1)/2) if len(d) else np.nan
    entropy=float(np.mean(-np.sum(np.where(alpha>0,alpha*np.log(np.maximum(alpha,1e-30)),0),axis=0))) if len(alpha) else np.nan
    mass=lambda mask:float(mean[mask].sum())
    correct_col=1 if heading>0 else 3;opposite_col=3 if heading>0 else 1
    connector=np.flatnonzero(ss[:,0]==1)
    if len(connector):
        top=connector[np.lexsort((lanes[connector],-mean[connector]))[0]]
        t=int(np.argmax(ss[top,1:5]))+1
        turn_type={1:'left',2:'straight',3:'right',4:'unknown'}[t]
    else:turn_type='NO_CONNECTOR'
    metrics={'GTRelevantMass2m':mass(d<=2),'GTRelevantMass4m':mass(d<=4),
        'GTNearestAttentionWeight':float(a),'GTNearestAttentionRank':rank_nearest,
        'IncomingGTNearestAttentionWeight':float(mean[incoming_min]) if len(d) else np.nan,
        'IncomingGTNearestAttentionRank':incoming_rank,
        'GTNearestIncoming':int(nearest_in_edges),'GTNearestLaneIndex':nearest,
        'GTNearestDistance':float(dist_all[nearest]) if nearest>=0 else np.nan,
        'GTNearestTieCount':len(tied),'BestCoNearestAttentionRank':rank(best_tie_weight) if len(tied) else np.nan,
        'BestCoNearestAttentionWeight':best_tie_weight,
        'Top1LaneGTDistance':float(d[order[:1]].min()) if len(d) else np.nan,
        'Top3MinGTDistance':float(d[order[:3]].min()) if len(d) else np.nan,
        'Top5MinGTDistance':float(d[order[:5]].min()) if len(d) else np.nan,
        'Top1LaneIndex':int(lanes[order[0]]) if len(d) else -1,
        'Top3LaneIndices':','.join(str(int(i)) for i in lanes[order[:3]]),
        'Top5LaneIndices':','.join(str(int(i)) for i in lanes[order[:5]]),
        'AttentionEntropy':entropy,'StraightMass':mass(ss[:,2]==1),
        'UnknownTurnMass':mass(ss[:,4]==1),'ConnectorPresent':int(bool(len(connector))),
        'CorrectTurnMass':mass(ss[:,correct_col]==1) if turning else np.nan,
        'OppositeTurnMass':mass(ss[:,opposite_col]==1) if turning else np.nan,
        'TopConnectorTurnType':turn_type,
        'TopConnectorMatchRate':int(turn_type==('left' if heading>0 else 'right')) if turning else np.nan}
    return metrics

@torch.no_grad()
def main():
    torch.set_num_threads(4)
    assert read_json(ROOT/'00_manifest/stage7c_attention_capture_integrity.json')['status']=='PASS'
    geometry_checks=numerical_distance_checks()
    ds=Stage7ASemanticDataset('val');assert len(ds)==3603 and len(ds.scene_indices)==150
    centerlines=np.load(STAGE7A/'02_semantic_cache/stage7a_centerlines.npz')
    ambiguity=TurnAmbiguity(ds.metadata,centerlines)
    side=pd.read_csv(STAGE7A/'01_data_audit/stage7a_formal_actor_val_groups.csv').set_index(IDS)
    models={n:load_model(n) for n in ['Stage3B','Stage7A']}
    digests={n:state_digest(m.state_dict()) for n,m in models.items()}
    hooks={n:LaneCapture(m) for n,m in models.items()}
    mlp=models['Stage7A'].local_encoder.al_encoder.semantic_mlp
    # All possible binary patterns supply a fixed lookup; frequency comes from VAL.
    patterns=np.array([[(k>>i)&1 for i in range(9)] for k in range(512)],dtype=np.float32)
    residual=mlp(torch.from_numpy(patterns).cuda()).cpu().numpy()
    r0=residual[0];norm_r=np.linalg.norm(residual,axis=1)
    norm_delta=np.linalg.norm(residual-r0,axis=1);ratio=norm_delta/(norm_r+1e-8)
    al=models['Stage7A'].local_encoder.al_encoder
    wv=al.lin_v.weight.detach().cpu().numpy();wk=al.lin_k.weight.detach().cpu().numpy()
    v_delta_norm=np.linalg.norm((residual-r0)@wv.T,axis=1)
    v_residual_norm=np.linalg.norm(residual@wv.T,axis=1)
    k_residual_norm=np.linalg.norm(residual@wk.T,axis=1)
    constant_v_norm=float(np.linalg.norm(r0@wv.T))
    pattern_counts=collections.Counter();rows=[];archives=[];sum_max=0.;edges_total=0
    t=time.monotonic()
    for batchno,start in enumerate(range(0,len(ds),16)):
        graphs=[ds[i] for i in range(start,min(start+16,len(ds)))]
        data=Batch.from_data_list(graphs).cuda()
        errors={};captured={}
        for name,model in models.items():
            out=model(input_for(data,name))
            assert all(torch.isfinite(out[k]).all() for k in ['raw_prediction','mode_logits','mode_prob'])
            _,er=errors_with_top1(model,out,data)
            errors[name]={k:er[k].cpu().numpy() for k in ['minFDE_K','Top1FDE6']}
            captured[name]=hooks[name].data.copy()
        ei=captured['Stage3B']['edge_index'];assert np.array_equal(ei,captured['Stage7A']['edge_index'])
        assert np.array_equal(captured['Stage3B']['edge_attr'],captured['Stage7A']['edge_attr'])
        ptr=np.cumsum([0]+[g.num_nodes for g in graphs]);lp=np.cumsum([0]+[len(g.lane_tokens) for g in graphs])
        actor_graph=np.repeat(np.arange(len(graphs)),np.diff(ptr));lane_graph=np.repeat(np.arange(len(graphs)),np.diff(lp))
        actor_node=np.arange(ptr[-1])-ptr[actor_graph];lane_local=np.arange(lp[-1])-lp[lane_graph]
        sem=np.concatenate([g.lane_semantic.numpy() for g in graphs])
        positions=np.concatenate([g.positions[:,4].numpy() for g in graphs])
        lane_positions=np.concatenate([g.lane_positions.numpy() for g in graphs])
        lane_vectors=np.concatenate([g.lane_vectors.numpy() for g in graphs])
        assert np.array_equal(actor_graph[ei[1]],lane_graph[ei[0]])
        dist_actor=point_segment_distances(positions[ei[1]],lane_positions[ei[0]],lane_vectors[ei[0]])
        edge_gt=np.full(ei.shape[1],np.nan,dtype=np.float32)
        incoming=[np.flatnonzero(ei[1]==i) for i in range(ptr[-1])]
        for name,cap in captured.items():
            alpha=cap['alpha'];assert alpha.shape==(ei.shape[1],8) and (alpha>=0).all()
            for ix in incoming:
                if len(ix):
                    err=float(np.abs(alpha[ix].sum(axis=0,dtype=np.float64)-1).max())
                    sum_max=max(sum_max,err);assert err<1e-5
        for j,g in enumerate(graphs):
            s=g.lane_semantic.numpy();codes=(s.astype(np.int32)*(1<<np.arange(9))).sum(axis=1)
            pattern_counts.update(codes.tolist())
            ambiguity_rows=ambiguity.graph(g)
            sg=segments(g.lane_positions.numpy(),g.lane_vectors.numpy())
            for node in np.flatnonzero(g.full_horizon_mask.numpy()):
                key=(g.scene_token,g.sample_token,g.instance_tokens[node]);old=side.loc[key]
                turning,heading=turn_membership(g.positions[node,4:].numpy(),int(g.agent_type[node])==0)
                assert turning==int(old.TurningVehicle_GT)
                fp=GT_fingerprint(g.positions[node,5:],g.future_mask[node],g.agent_type[node])
                assert fp['GT_trajectory_sha256']==old.GT_trajectory_sha256
                ix=incoming[ptr[j]+node];lanes=ei[0,ix]-lp[j]
                distances=future_distances(g.positions[node,5:].numpy(),sg)
                edge_gt[ix]=distances[lanes]
                row={'scene_name':g.scene_name,'scene_token':g.scene_token,'sample_token':g.sample_token,
                    'instance_token':g.instance_tokens[node],'node_in_graph':int(node),
                    'agent_type':CLASSES[int(g.agent_type[node])],'motion_state':g.t0_motion_state[node],
                    'GT_endpoint_displacement_m':float(torch.linalg.vector_norm(g.y[node,-1])),
                    'TurningVehicle_GT':turning,'GT_heading_change_deg':heading,
                    'LeftConnectorCount20':int(ambiguity_rows[node,0]),'StraightConnectorCount20':int(ambiguity_rows[node,1]),
                    'RightConnectorCount20':int(ambiguity_rows[node,2]),'TurnOptionCount20':int(ambiguity_rows[node,3]),
                    'IncomingLaneCount':len(ix),'GraphLaneCount':len(s),**fp}
                for name in models:
                    cap=captured[name];a=cap['alpha'][ix]
                    row.update({name+'_'+k:v for k,v in attention_metrics(lanes,a,distances,s,turning,heading).items()})
                    row[name+'_forward_minFDE']=float(errors[name]['minFDE_K'][ptr[j]+node])
                    row[name+'_forward_Top1FDE']=float(errors[name]['Top1FDE6'][ptr[j]+node])
                    for source,target in [('geometry_norm','MeanGeometryNorm'),('key_norm','MeanKeyNorm'),('value_norm','MeanValueNorm')]:
                        row[name+'_'+target]=float(cap[source][ix].mean()) if len(ix) else np.nan
                a=captured['Stage7A']['alpha'][ix].astype(np.float64).mean(axis=1);c=codes[lanes]
                row['SemanticEffect']=float(np.sum(a*norm_delta[c]))
                row['GTRelevantSemanticEffect']=float(np.sum(a[distances[lanes]<=2]*norm_delta[c[distances[lanes]<=2]]))
                row['AttentionWeightedResidualValueNorm']=float(np.sum(a*v_residual_norm[c]))
                row['AttentionWeightedSpecificValueNorm']=float(np.sum(a*v_delta_norm[c]))
                row['AttentionWeightedResidualKeyNorm']=float(np.sum(a*k_residual_norm[c]))
                row['ConstantValueComponentNorm']=constant_v_norm
                vb=captured['Stage3B']['value'][ix];vo=captured['Stage7A']['value'][ix]
                row['MeanValueDeltaNorm']=float(np.linalg.norm(vo-vb,axis=1).mean()) if len(ix) else np.nan
                denom=np.linalg.norm(vb,axis=1)*np.linalg.norm(vo,axis=1)+1e-8
                row['MeanValueCosine']=float((np.sum(vb*vo,axis=1)/denom).mean()) if len(ix) else np.nan
                rows.append(row)
        filename=ROOT/f'01_attention_capture/stage7c_edges_batch_{batchno:04d}.npz'
        payload={'scene_token':np.asarray([g.scene_token for g in graphs]),'sample_token':np.asarray([g.sample_token for g in graphs]),
            'scene_name':np.asarray([g.scene_name for g in graphs]),'actor_instance_token':np.asarray([x for g in graphs for x in g.instance_tokens]),
            'actor_node_index':actor_node.astype(np.int32),'actor_graph_index':actor_graph.astype(np.int32),
            'lane_token':np.asarray([x for g in graphs for x in g.lane_tokens]),'lane_local_index':lane_local.astype(np.int32),
            'lane_graph_index':lane_graph.astype(np.int32),'lane_semantic':sem,'lane_positions':lane_positions,'lane_vectors':lane_vectors,
            'edge_lane_index':ei[0].astype(np.int32),'edge_actor_index':ei[1].astype(np.int32),
            'edge_actor_node_index':actor_node[ei[1]].astype(np.int32),'edge_lane_local_index':lane_local[ei[0]].astype(np.int32),
            'edge_graph_index':actor_graph[ei[1]].astype(np.int32),'distance_actor_lane':dist_actor.astype(np.float32),
            'distance_actor_lane_start':np.linalg.norm(captured['Stage3B']['edge_attr'],axis=1),
            'GT_future_segment_distance_full_target_only':edge_gt}
        for name,cap in captured.items():
            payload[name+'_attention_per_head']=cap['alpha'];payload[name+'_attention_mean']=cap['alpha'].mean(axis=1)
            payload[name+'_attention_max']=cap['alpha'].max(axis=1)
        np.savez_compressed(filename,**payload)
        edges_total+=ei.shape[1]
        archives.append({'path':str(filename.relative_to(ROOT)),'bytes':filename.stat().st_size,'sha256':sha256(filename),
            'graphs':len(graphs),'edges':ei.shape[1],'first_dataset_index':start})
        if (batchno+1)%10==0:print('PAIRED_CAPTURE',min(start+16,len(ds)),'/3603','actors',len(rows),'edges',edges_total,'seconds',round(time.monotonic()-t,1),flush=True)
    for h in hooks.values():h.close()
    for name,m in models.items():assert state_digest(m.state_dict())==digests[name]
    ds.clear();assert len(rows)==54990 and sum(r['TurningVehicle_GT'] for r in rows)==1663
    frame=pd.DataFrame(rows);assert not frame.duplicated(IDS).any()
    errors=paired_errors();merged=frame.merge(errors[IDS+['Stage3B_minFDE6','Stage7A_minFDE6','Stage3B_Top1FDE6','Stage7A_Top1FDE6','DeltaMinFDE','DeltaTop1FDE']],on=IDS,validate='one_to_one')
    reproduction={}
    for name in models:
        reproduction[name]={}
        for metric in ['minFDE','Top1FDE']:
            diff=(merged[name+'_forward_'+metric]-merged[name+'_'+metric+'6']).to_numpy()
            reproduction[name][metric]={'maximum_actor_difference':float(np.abs(diff).max()),'mean_difference':float(diff.mean())}
            assert abs(diff.mean())<1e-5
    merged['DeltaRelevantMass']=merged.Stage7A_GTRelevantMass2m-merged.Stage3B_GTRelevantMass2m
    merged['DeltaGTNearestAttentionRank']=merged.Stage7A_GTNearestAttentionRank-merged.Stage3B_GTNearestAttentionRank
    merged['DeltaCorrectTurnMass']=merged.Stage7A_CorrectTurnMass-merged.Stage3B_CorrectTurnMass
    merged.to_csv(ROOT/'02_relevance_analysis/stage7c_actor_attention.csv',index=False)
    counts=np.array([pattern_counts.get(i,0) for i in range(512)],dtype=np.int64)
    np.savez_compressed(ROOT/'03_representation_analysis/stage7c_residual_patterns.npz',patterns=patterns,
        residual=residual,r0=r0,norm_r=norm_r,norm_delta_r=norm_delta,ratio=ratio,counts=counts,
        residual_value_norm=v_residual_norm,specific_value_norm=v_delta_norm,residual_key_norm=k_residual_norm)
    atomic_json(ROOT/'01_attention_capture/stage7c_capture_manifest.json',{'status':'PASS','VAL_scenes':150,'windows':3603,
        'full_targets':len(rows),'TurningVehicle_GT_count':1663,'captured_edges':edges_total,'batches':len(archives),
        'edge_alpha_sum_max_error':sum_max,'edge_indices_identical':True,'model_states_unchanged':True,
        'Stage3B_semantic_input':False,'GT_relevance_model_input':False,'archives':archives,
        'normalized_storage':'edge_graph_index joins scene/sample; edge_actor_index joins actor_instance_token; edge_lane_index joins lane_token/semantic. These lossless indices store all required per-edge fields.',
        'GT_distance_NaNs':'intentional only for context or partial targets; predictions have no NaN/Inf',
        'prediction_reproduction':reproduction,'geometry_numerical_checks':geometry_checks,
        'zero_edge_full_targets':int((merged.IncomingLaneCount==0).sum()),
        'GT_nearest_outside_incoming_edges':int((merged.Stage7A_GTNearestIncoming==0).sum()),
        'segment_occurrences':int(counts.sum()),'observed_patterns':int((counts>0).sum()),'training':False,'test_used':False})
    print('PAIRED_CAPTURE COMPLETE',len(rows),edges_total,reproduction,flush=True)

if __name__=='__main__':main()
