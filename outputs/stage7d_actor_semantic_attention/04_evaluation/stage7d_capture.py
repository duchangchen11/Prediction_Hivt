"""Frozen Stage7C GT-segment analysis with Stage7D attention, no forward alteration."""
from stage7d_analysis_common import *
sys.path[:0]=[str(STAGE7C/'01_attention_capture'),str(STAGE7C/'02_relevance_analysis'),str(STAGE7C/'00_manifest')]
from stage7c_hooks import LaneCapture
from stage7c_capture import attention_metrics
from stage7c_geometry import segments,future_distances,numerical_distance_checks
from torch_geometric.data import Batch
from stage7d_dataset import SEMANTIC_FIELDS

class BiasGroups:
    def __init__(self,model):
        self.stats={};self.patterns={}
        self.handle=model.local_encoder.al_encoder.semantic_score_mlp.register_forward_hook(self.observe)
    def observe(self,module,args,output):
        z=args[0].detach().cpu().numpy();bias=output.detach().cpu().numpy();assert np.isfinite(bias).all()
        s=z[:,:9];types=z[:,11:].argmax(1)
        codes=(s.astype(np.int32)*(1<<np.arange(9))).sum(1)
        for code in np.unique(codes):self.patterns[int(code)]=s[np.flatnonzero(codes==code)[0]].copy()
        semantic={'ordinary lane':s[:,0]==0,'connector':s[:,0]==1,
             **{name:s[:,i]==1 for i,name in enumerate(SEMANTIC_FIELDS) if i>0}}
        for g,actor in [('Overall',np.ones(len(z),bool)),*[(n,types==i) for i,n in enumerate(('Vehicle','Pedestrian','Bicycle'))]]:
            for label,mask in semantic.items():
                b=bias[mask&actor];key=(g,label)
                if key not in self.stats:self.stats[key]=[0,np.zeros(8),np.zeros(8)]
                self.stats[key][0]+=len(b);self.stats[key][1]+=b.sum(0,dtype=np.float64);self.stats[key][2]+=np.abs(b).sum(0,dtype=np.float64)
    def finish(self,model):
        self.handle.remove();rows=[]
        for (g,s),(n,total,absolute) in self.stats.items():
            rows.append({'ActorGroup':g,'SemanticGroup':s,'Edges':n,
                **{f'mean_bias_head{h}':float(total[h]/n) if n else np.nan for h in range(8)},
                **{f'mean_abs_bias_head{h}':float(absolute[h]/n) if n else np.nan for h in range(8)},
                'mean_bias':float(total.sum()/n/8) if n else np.nan,'mean_abs_bias':float(absolute.sum()/n/8) if n else np.nan})
        write_csv(ROOT/'06_tables/stage7d_bias_statistics.csv',rows)
        controlled=[];mlp=model.local_encoder.al_encoder.semantic_score_mlp
        for code,s in sorted(self.patterns.items()):
            for x,y in [(0.,0.),(-20.,0.),(20.,0.),(0.,-20.),(0.,20.)]:
                values=[];effects=[]
                for i,name in enumerate(('Vehicle','Pedestrian','Bicycle')):
                    onehot=np.eye(3,dtype=np.float32)[i];z=np.r_[s,x/50,y/50,onehot].astype(np.float32)
                    b=mlp(torch.from_numpy(z).cuda()).detach().cpu().numpy()
                    ordinary=z.copy();ordinary[:9]=0
                    contrast=b-mlp(torch.from_numpy(ordinary).cuda()).detach().cpu().numpy()
                    values.append(b);effects.append(contrast)
                    controlled.append({'PatternCode':code,**dict(zip(SEMANTIC_FIELDS,s.astype(int))),
                        'relative_x_m':x,'relative_y_m':y,'ActorType':name,
                        **{f'bias_head{h}':float(b[h]) for h in range(8)},
                        **{f'semantic_vs_allzero_head{h}':float(contrast[h]) for h in range(8)}})
        write_csv(ROOT/'06_tables/stage7d_controlled_actor_bias.csv',controlled)
        table=pd.DataFrame(controlled);maximum=0.;semantic_max=0.
        for _,a in table.groupby(['PatternCode','relative_x_m','relative_y_m']):
            maximum=max(maximum,float(np.ptp(a[[f'bias_head{h}' for h in range(8)]].to_numpy(),axis=0).max()))
            semantic_max=max(semantic_max,float(np.ptp(a[[f'semantic_vs_allzero_head{h}' for h in range(8)]].to_numpy(),axis=0).max()))
        atomic_json(ROOT/'02_model_audit/stage7d_actor_conditioning_audit.json',{
            'status':'PASS' if maximum>0 and semantic_max>0 else 'NO_DIFFERENTIAL_RESPONSE',
            'observed_semantic_patterns':len(self.patterns),'controlled_positions':5,'actor_types':3,
            'same_semantics_same_position_type_response_max_difference':maximum,
            'type_difference_in_semantic_vs_zero_contrast_max':semantic_max,
            'interpretation':'descriptive controlled network response, not causality or forecasting support'})

@torch.no_grad()
def main():
    assert read_json(ROOT/'02_model_audit/stage7d_attention_capture_integrity.json')['status']=='PASS'
    old=pd.read_csv(STAGE7C/'02_relevance_analysis/stage7c_actor_attention.csv',dtype={'future_mask_bits':str}).set_index(FULL_IDS)
    assert len(old)==54990
    ds=Stage7DSemanticDataset('val');model=model_new()
    saved=torch.load(BEST,map_location='cpu',weights_only=False);model.load_state_dict(saved['state_dict'],strict=True)
    model.eval().requires_grad_(False);before=state_digest(model.state_dict())
    hook=LaneCapture(model,representations=False);bias=BiasGroups(model);rows=[];edges=0;max_sum_diff=0.;archives=[]
    for batchno,start in enumerate(range(0,len(ds),16)):
        graphs=[ds[i] for i in range(start,min(start+16,len(ds)))];data=Batch.from_data_list(graphs).cuda()
        out=model(model_input(data));_,er=errors_with_top1(model,out,data)
        assert all(torch.isfinite(out[k]).all() for k in ('raw_prediction','mode_logits','mode_prob'))
        ei=hook.data['edge_index'];alpha=hook.data['alpha'];assert alpha.shape==(ei.shape[1],8) and (alpha>=0).all()
        frozen_archive=STAGE7C/f'01_attention_capture/stage7c_edges_batch_{batchno:04d}.npz'
        with np.load(frozen_archive) as reference:
            assert np.array_equal(ei[0],reference['edge_lane_index'])
            assert np.array_equal(ei[1],reference['edge_actor_index'])
        archive=ROOT/f'04_evaluation/stage7d_edges_batch_{batchno:04d}.npz'
        np.savez_compressed(archive,edge_lane_index=ei[0].astype(np.int32),edge_actor_index=ei[1].astype(np.int32),
                            Stage7D_attention_per_head=alpha)
        archives.append({'path':str(archive.relative_to(ROOT)),'sha256':sha256(archive),'bytes':archive.stat().st_size,
            'edges':len(alpha),'first_dataset_index':start,'identity_semantic_geometry_reference':str(frozen_archive.relative_to(PROJECT)),
            'reference_sha256':sha256(frozen_archive),'edge_indices_bitwise_identical':True})
        ptr=np.cumsum([0]+[g.num_nodes for g in graphs]);lp=np.cumsum([0]+[len(g.lane_tokens) for g in graphs])
        incoming=[np.flatnonzero(ei[1]==i) for i in range(ptr[-1])]
        for ix in incoming:
            if len(ix):max_sum_diff=max(max_sum_diff,float(np.abs(alpha[ix].sum(0,dtype=np.float64)-1).max()))
        assert max_sum_diff<1e-5
        errors={k:er[k].cpu().numpy() for k in ('minFDE_K','Top1FDE6')}
        for j,g in enumerate(graphs):
            sg=segments(g.lane_positions.numpy(),g.lane_vectors.numpy());sem=g.lane_semantic.numpy()
            for node in np.flatnonzero(g.full_horizon_mask.numpy()):
                key=(g.scene_token,g.sample_token,g.instance_tokens[node]);a=old.loc[key]
                fp=GT_fingerprint(g.positions[node,5:],g.future_mask[node],g.agent_type[node])
                assert fp['GT_trajectory_sha256']==a.GT_trajectory_sha256 and fp['future_mask_bits']==a.future_mask_bits
                ix=incoming[ptr[j]+node];lanes=ei[0,ix]-lp[j]
                distances=future_distances(g.positions[node,5:].numpy(),sg);assert np.isfinite(distances).all()
                measures=attention_metrics(lanes,alpha[ix],distances,sem,bool(a.TurningVehicle_GT),a.GT_heading_change_deg)
                row=dict(zip(FULL_IDS,key));row.update({k:a[k] for k in ['agent_type','motion_state','GT_endpoint_displacement_m',
                    'TurningVehicle_GT','GT_heading_change_deg','TurnOptionCount20','node_in_graph','IncomingLaneCount','GraphLaneCount']})
                assert len(ix)==a.IncomingLaneCount and len(sem)==a.GraphLaneCount
                for n in ('Stage3B','Stage7A'):
                    for k in ATTENTION_METRICS+TURN_METRICS:row[n+'_'+k]=a[n+'_'+k]
                row.update({'Stage7D_'+k:v for k,v in measures.items()})
                row.update(Stage7D_forward_minFDE=float(errors['minFDE_K'][ptr[j]+node]),
                           Stage7D_forward_Top1FDE=float(errors['Top1FDE6'][ptr[j]+node]))
                rows.append(row)
        edges+=len(alpha)
        if (batchno+1)%10==0:print('ATTENTION_CAPTURE',min(start+16,len(ds)),'/3603','actors',len(rows),flush=True)
    hook.close();bias.finish(model);assert state_digest(model.state_dict())==before;ds.clear()
    f=pd.DataFrame(rows);assert len(f)==54990 and f.TurningVehicle_GT.sum()==1663 and not f.duplicated(FULL_IDS).any()
    performance=paired_frames();performance=performance[performance.horizon=='full_horizon']
    f=f.merge(performance[FULL_IDS+['Stage3B_minFDE6','Stage7D_minFDE6','Stage3B_Top1FDE6','Stage7D_Top1FDE6','DeltaMinFDE','DeltaTop1FDE']],on=FULL_IDS,validate='one_to_one')
    reproduction={}
    for short,key in [('minFDE','minFDE6'),('Top1FDE','Top1FDE6')]:
        delta=f['Stage7D_forward_'+short]-f['Stage7D_'+key]
        reproduction[short]={'max_actor_difference':float(np.abs(delta).max()),'mean_difference':float(delta.mean())}
        assert abs(delta.mean())<1e-5
    f['DeltaRelevantMass']=f.Stage7D_GTRelevantMass2m-f.Stage3B_GTRelevantMass2m
    f['DeltaGTNearestAttentionRank']=f.Stage7D_GTNearestAttentionRank-f.Stage3B_GTNearestAttentionRank
    f['DeltaCorrectTurnMass']=f.Stage7D_CorrectTurnMass-f.Stage3B_CorrectTurnMass
    f.to_csv(ROOT/'04_evaluation/stage7d_actor_attention.csv',index=False)
    atomic_json(ROOT/'04_evaluation/stage7d_capture_manifest.json',{'status':'PASS','VAL_scenes':150,'windows':3603,'full_targets':len(f),
        'TurningVehicle_GT':1663,'edges':edges,'alpha_sum_max_error':max_sum_diff,'model_state_unchanged':True,
        'geometry_checks':numerical_distance_checks(),'baseline_attention_frozen_sha256':sha256(STAGE7C/'02_relevance_analysis/stage7c_actor_attention.csv'),
        'relevance_definition_reused':'Stage7C true GT polyline to segment distance; all-graph nearest; average ties',
        'archives':archives,'storage':'lossless Stage7D alpha[8] and identical edge indices join frozen Stage7C scene/sample/actor/lane/semantic metadata',
        'forward_reproduction':reproduction,'GT_in_model_input':False,'training':False,'test_used':False})
    print('ATTENTION_CAPTURE_PASS',len(f),edges,flush=True)

if __name__=='__main__':torch.set_num_threads(4);main()
