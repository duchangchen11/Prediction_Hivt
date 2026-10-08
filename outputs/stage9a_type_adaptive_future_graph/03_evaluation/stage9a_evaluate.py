"""One frozen secondary VAL150 pass; no checkpoint or architecture selection."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'01_training/00_code'))
from stage9a_common import *
MODELS=('R0','R2','G1','G3','T1','T2','T3')
METRICS=('minADE6','minFDE6','MR6','Top1ADE','Top1FDE','OracleGap','HitRate','BestModeRank','MRR')

@torch.no_grad()
def main():
    seed();verify();freeze=read_json(FROZEN);assert freeze['status']=='FROZEN_ALL_COMPLETE'
    models={}
    for row in freeze['checkpoints']:
        p=PROJECT/row['Path'];assert sha256(p)==row['SHA256'];m=fresh(row['Variant']);m.load_state_dict(torch.load(p,map_location='cpu',weights_only=False)['state_dict']);models[row['Variant']]=m.eval().requires_grad_(False)
    r2=frozen_r2();predictor=frozen_predictor();states=[state_sha(x) for x in (r2,predictor)]
    r2norm=read_json(S6/'02_features/stage6a_normalization.json')
    # Semantic computation exists solely to reproduce the immutable G3 reference.
    # Neither these tensors nor modules can enter Stage9 graph/model arguments.
    sys.path.insert(0,str(S8/'01_training/00_code'));import stage8a1_common as reference
    index=reference.SparseSemanticIndex();refnorm=read_json(reference.NORM)['statistics']
    oldfreeze=read_json(reference.FROZEN)
    for v in ('G1','G3'):
        row=next(r for r in oldfreeze['checkpoints'] if r['Variant']==v);p=PROJECT/row['Path'];assert sha256(p)==row['SHA256']
        m=reference.fresh(v);m.load_state_dict(torch.load(p,map_location='cpu',weights_only=False)['state_dict']);models[v]=m.eval().requires_grad_(False)
    dest=ROOT/'03_evaluation';register=dest/'stage9a_val_registration.json'
    spec={'status':'LOCKED_BEFORE_SECONDARY_VAL_PASS','checkpoint_manifest_sha256':sha256(FROZEN),'protocol_sha256':sha256(ROOT/'00_manifest/stage9a_protocol.json'),'evaluation_source_sha256':sha256(Path(__file__)),'models':MODELS,'Official_VAL_role':'development-reuse evidence','VAL_model_selection':False,'fresh_predictor':True,'Stage9_semantic_inputs':False,'G3_reference_only':True,'test_used':False}
    if register.exists():assert read_json(register)==json.loads(json.dumps(spec))
    else:atomic_json(register,spec)
    assert not (dest/'stage9a_complete.json').exists(),'Secondary VAL already complete'
    ds=SceneDataset('val');records=[r for r in read_json(reference.C/'02_graph_cache/stage8a0c_selector_manifest.json')['batches'] if r['split']=='val']
    allrows=[];windows_checked=0
    for bi,rec in enumerate(records):
        p=dest/'cache'/f'stage9a_val_batch_{bi*16:05d}.pt'
        if p.exists():
            result=torch.load(p,map_location='cpu',weights_only=False);assert result['registration_sha256']==sha256(register)
        else:
            live=forward_batch(predictor,ds,bi*16);old=torch.load(S6/rec['source_path'],map_location='cpu',weights_only=False)
            assert sha256(S6/rec['source_path'])==rec['source_sha256'] and sha256(reference.C/rec['path'])==rec['sha256']
            framepath=S8/'02_graph_cache'/rec['frame_path'];assert sha256(framepath)==rec['frame_sha256']
            rows=[];window_data=[]
            with np.load(reference.C/rec['path']) as sel,np.load(framepath) as frame:
                for j,(w,ow) in enumerate(zip(live['windows'],old['windows'])):
                    assert w['dataset_index']==ow['dataset_index']==bi*16+j
                    assert w['raw_prediction_sha256']==ow['raw_prediction_sha256'] and w['ego_prediction_sha256']==ow['ego_prediction_sha256']
                    assert w['instance_tokens']==ow['instance_tokens']
                    for k in ('GT','future_mask','target_mask','agent_type','history','history_padding','mode_logits','mode_prob'):assert torch.equal(w[k],ow[k]),k
                    targets=torch.where(w['target_mask']&w['future_mask'].all(-1))[0];n=len(targets)
                    if not n:continue
                    args,gate,context,idx,keep=graph_from_window(w,targets)
                    args=tuple(x.cuda() for x in args);gate=gate.cuda()
                    baseargs=[w[k].cuda() for k in ('history','history_padding','agent_type','ego_prediction','mode_logits','mode_prob')]
                    feat,flags,_,_=observable_features(*baseargs);base=r2(r2normalize(feat,flags,r2norm,'R2')[targets],w['mode_logits'][targets].cuda())['mode_logits']
                    prob={'R0':w['mode_prob'][targets].cuda(),'R2':base.softmax(-1)};gates={};alphas={}
                    for v in VARIANTS:
                        out=models[v](*args,base,gate);prob[v]=out['mode_prob'];gates[v]=out['gate'].cpu();alphas[v]=out['interaction_alpha'].cpu()
                        assert torch.isfinite(out['gate']).all() and bool(((out['gate']>=0)&(out['gate']<=1)).all())
                    obs,selected,node,ri,rk,mapfeature=reference.graph_window(w,rec,j,index,sel,frame)
                    edge=interaction_edges(obs,ri,rk,targets);raw=reference.pack_targets(node,ri,rk,edge,selected,mapfeature,targets)
                    ra=tuple(x.cuda() for x in reference.normalize_args(raw,refnorm))
                    assert all(torch.equal(a,b) for a,b in zip(args,ra[:3])),'Stage9/reference future graph identity'
                    for v in ('G1','G3'):prob[v]=models[v](*ra)['mode_prob']
                    pred=w['ego_prediction'][targets].cuda();predsha=tensor_sha(pred);distance=(pred-w['GT'][targets,None].cuda()).norm(dim=-1)
                    fde=distance[:,:,-1];ade=distance.mean(-1);best=fde.argmin(-1);ii=torch.arange(n,device='cuda');minimum=fde[ii,best];minade=ade[ii,best]
                    metrics={};tops={}
                    for v in MODELS:
                        q=prob[v];assert torch.isfinite(q).all();top=q.argmax(-1);tops[v]=top.cpu();rank=(q.argsort(dim=-1,descending=True,stable=True)==best[:,None]).long().argmax(-1)+1
                        metrics[v]=torch.stack((minade,minimum,(minimum>2).float(),ade[ii,top],fde[ii,top],fde[ii,top]-minimum,(top==best).float(),rank.float(),1/rank.float()),-1).cpu().numpy()
                        assert np.array_equal(metrics[v][:,:3],metrics['R0'][:,:3])
                    assert tensor_sha(pred)==predsha
                    currentdist=(w['history'][targets,4,None].double()-w['history'][idx[targets],4].double()).norm(dim=-1).masked_fill(~keep[targets],float('inf')).min(-1).values
                    for a,t in enumerate(targets.tolist()):
                        count=int(keep[t].sum());nearest=float(currentdist[a]) if count else 50.
                        distancegroup='NoNeighbor' if not count else '0-5m' if nearest<5 else '5-10m' if nearest<10 else '10-20m' if nearest<20 else '20-50m'
                        countgroup='Neighbor0' if count==0 else 'Neighbor1-2' if count<=2 else 'Neighbor3-5' if count<=5 else 'Neighbor6-8'
                        row={'scene_token':w['scene_token'],'sample_token':w['sample_token'],'instance_token':w['instance_tokens'][t],'dataset_index':w['dataset_index'],'node_in_graph':t,'agent_type':CLASSES[int(w['agent_type'][t])],'motion_state':w['motion_state'][t],'GT_endpoint_displacement_m':float((w['GT'][t,-1]-w['current_position'][t]).norm()),'neighbor_count':count,'nearest_current_neighbor_distance_m':nearest,'NeighborCountGroup':countgroup,'NearestDistanceGroup':distancegroup,'best_mode':int(best[a])}
                        for v in MODELS:row.update({v+'_'+m:float(x) for m,x in zip(METRICS,metrics[v][a])});row[v+'_top1_mode']=int(tops[v][a])
                        for v in VARIANTS:row[v+'_gate']=float(gates[v][a])
                        rows.append(row)
                    window_data.append({'dataset_index':w['dataset_index'],'scene_token':w['scene_token'],'sample_token':w['sample_token'],'targets':targets,'probabilities':{v:q.cpu() for v,q in prob.items()},'gates':gates,'interaction_alpha':alphas,'fde':fde.cpu(),'ade':ade.cpu(),'raw_prediction_sha256':w['raw_prediction_sha256'],'ego_prediction_sha256':w['ego_prediction_sha256']})
            result={'registration_sha256':sha256(register),'rows':rows,'windows':window_data,'all_source_windows_bitwise_checked':len(live['windows'])};atomic_torch(p,result)
        allrows.extend(result['rows']);windows_checked+=result['all_source_windows_bitwise_checked']
        if (bi+1)%10==0:print('SECONDARY_VAL',bi+1,'/',len(records),'FULL',len(allrows),flush=True)
    assert len(allrows)==54990 and len({r['scene_token'] for r in allrows})==150 and len({(r['scene_token'],r['sample_token'],r['instance_token']) for r in allrows})==54990
    assert states==[state_sha(x) for x in (r2,predictor)] and not any(p.grad is not None or p.requires_grad for m in (r2,predictor) for p in m.parameters());verify()
    write_csv(dest/'stage9a_actor_results.csv',allrows)
    atomic_json(dest/'stage9a_candidate_identity.json',{'status':'PASS','targets':54990,'scenes':150,'all_source_windows_checked':windows_checked,'shared_prediction_tensor':True,'fresh_equals_frozen_cache_bitwise':True,'candidate_maxdiff':0.,'minADE6_maxdiff':0.,'minFDE6_maxdiff':0.,'MR6_maxdiff':0.,'predictor_gradient_count':0,'R2_gradient_count':0,'predictor_R2_states_unchanged':True,'Stage9_semantic_inputs':False})
    atomic_json(dest/'stage9a_complete.json',{'status':'PASS','models':MODELS,'targets':54990,'scenes':150,'successful_secondary_VAL_passes':1,'Official_VAL_role':'development-reuse evidence','actor_csv_sha256':sha256(dest/'stage9a_actor_results.csv'),'checkpoint_manifest_sha256':sha256(FROZEN)})
    ds.clear();print('STAGE9_SECONDARY_VAL_PASS',flush=True)

if __name__=='__main__':main()
