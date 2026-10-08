"""Single locked fresh VAL forward pipeline; all five ranks share one tensor."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'01_training/00_code'))
from stage8a1_common import *
sys.path[:0]=[str(S6/'00_manifest'),str(S6/'01_cache')]
from stage6a_common import frozen_predictor,SceneDataset,tensor_sha
from stage6a_cache import forward_batch
from stage6a_features import observable_features,normalize as r2normalize
from stage6a_head import ReliabilityHead
MODELS=('R0','R2','G1','G2','G3')
METRICS=('minADE6','minFDE6','MR6','Top1ADE','Top1FDE','OracleGap','HitRate','BestModeRank','MRR')
CLASSES=('Vehicle','Pedestrian','Bicycle')

def loaded():
    frozen=read_json(FROZEN);assert frozen['status']=='FROZEN_ALL_COMPLETE'
    assert frozen['normalization_sha256']==sha256(NORM)
    models={}
    for row in frozen['checkpoints']:
        path=PROJECT/row['Path'];assert sha256(path)==row['SHA256'];m=fresh(row['Variant'])
        m.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['state_dict'],strict=True)
        m.eval().requires_grad_(False);models[row['Variant']]=m
    r2path=S6/'07_checkpoints/stage6a_r2_best.pt'
    assert sha256(r2path)=='e3de457d635cd30c629ce316d73a87e419a3ded8abbe0e1f2601f1071527e314'
    m=ReliabilityHead().cuda();m.load_state_dict(torch.load(r2path,map_location='cpu',weights_only=False)['state_dict'],strict=True)
    models['R2']=m.eval().requires_grad_(False)
    return models

def semantic_sidecar():
    path=ROOT.parent/'stage7a_semantic_map_hivt/01_data_audit/stage7a_formal_actor_val_groups.csv'
    with path.open() as f:rows=list(csv.DictReader(f))
    out={tuple(r[k] for k in ('scene_token','sample_token','instance_token')):r for r in rows}
    assert len(out)==54990 and sum(int(r['TurningVehicle_GT']) for r in out.values())==1663
    return out,path

def main():
    seed();verify();models=loaded();predictor=frozen_predictor();predictorsha=state_sha(predictor)
    norm=read_json(NORM)['statistics'];r2norm=read_json(S6/'02_features/stage6a_normalization.json')
    index=SparseSemanticIndex();groups,sidecar=semantic_sidecar();ds=SceneDataset('val')
    dest=ROOT/'03_evaluation';(dest/'cache').mkdir(exist_ok=True)
    register=dest/'stage8a1_val_registration.json'
    spec={'status':'LOCKED_BEFORE_SINGLE_OFFICIAL_PASS','checkpoint_manifest_sha256':sha256(FROZEN),
        'normalization_sha256':sha256(NORM),'sidecar_sha256':sha256(sidecar),'evaluation_source_sha256':sha256(Path(__file__)),
        'models':MODELS,'checkpoint_selection_from_VAL':False,'fresh_predictor':True,'shared_geometry':True,'test_used':False}
    if register.exists():assert read_json(register)==json.loads(json.dumps(spec))
    else:atomic_json(register,spec)
    assert not (dest/'stage8a1_complete.json').exists(),'Formal VAL already complete'
    records=[r for r in read_json(C/'02_graph_cache/stage8a0c_selector_manifest.json')['batches'] if r['split']=='val']
    allrows=[];diag_sums={};topentities=[];capturemax=0.;validated=0
    for bi,rec in enumerate(records):
        path=dest/'cache'/(Path(rec['path']).stem+'.pt');start=bi*16
        if path.exists():
            payload=torch.load(path,map_location='cpu',weights_only=False);assert payload['registration_sha256']==sha256(register)
        else:
            live=forward_batch(predictor,ds,start);old=torch.load(S6/rec['source_path'],map_location='cpu',weights_only=False)
            assert sha256(S6/rec['source_path'])==rec['source_sha256']
            assert sha256(C/rec['path'])==rec['sha256']
            frames=ROOT/'02_graph_cache'/rec['frame_path'];assert sha256(frames)==rec['frame_sha256']
            rows=[];diagnostic=[];entityrows=[];windows=[];maxdiff=0.
            with np.load(C/rec['path']) as sel,np.load(frames) as frame:
                for j,(w,ow) in enumerate(zip(live['windows'],old['windows'])):
                    assert w['raw_prediction_sha256']==ow['raw_prediction_sha256'] and w['ego_prediction_sha256']==ow['ego_prediction_sha256']
                    assert w['instance_tokens']==ow['instance_tokens'] and torch.equal(w['mode_logits'],ow['mode_logits']) and torch.equal(w['mode_prob'],ow['mode_prob'])
                    obs,selected,node,idx,keep,feature=graph_window(w,rec,j,index,sel,frame)
                    targets=torch.where(w['target_mask']&w['future_mask'].all(-1))[0];n=len(targets)
                    if not n:continue
                    edge=interaction_edges(obs,idx,keep,targets);rawargs=pack_targets(node,idx,keep,edge,selected,feature,targets)
                    args=tuple(a.cuda() for a in normalize_args(rawargs,norm))
                    pred=w['ego_prediction'][targets].cuda();predsha=tensor_sha(pred)
                    dist=(pred-w['GT'][targets,None].cuda()).norm(dim=-1);fde=dist[:,:,-1];ade=dist.mean(-1)
                    best=fde.argmin(-1);ii=torch.arange(n,device='cuda');minimum=fde[ii,best];minimumade=ade[ii,best]
                    baseargs=[w[k].cuda() for k in ('history','history_padding','agent_type','ego_prediction','mode_logits','mode_prob')]
                    rx,flags,_,_=observable_features(*baseargs);rx=r2normalize(rx,flags,r2norm,'R2')[targets]
                    probabilities={'R0':w['mode_prob'][targets].cuda(),'R2':models['R2'](rx,args[-1])['mode_prob']}
                    attentions={};maxdiff=0.
                    for v in PARAMS:
                        normal=models[v](*args);out,att=capture(models[v],args)
                        diff=float((normal['mode_logits']-out['mode_logits']).abs().max());assert diff<1e-7;maxdiff=max(maxdiff,diff)
                        probabilities[v]=out['mode_prob'];attentions[v]=att
                        assert tensor_sha(pred)==predsha # reranker has no candidate output; same shared tensor all variants
                    values={};tops={}
                    for v,p in probabilities.items():
                        top=p.argmax(-1);tops[v]=top.cpu();order=p.argsort(dim=-1,descending=True,stable=True)
                        rank=(order==best[:,None]).long().argmax(-1)+1
                        values[v]=torch.stack((minimumade,minimum,(minimum>2).float(),ade[ii,top],fde[ii,top],fde[ii,top]-minimum,
                            (top==best).float(),rank.float(),1/rank.float()),-1).cpu().numpy()
                    for v in MODELS:assert np.array_equal(values[v][:,:3],values['R0'][:,:3])
                    lookup={int(v):k for k,v in enumerate(selected)};rr=np.array([lookup[int(t)] for t in targets])
                    entityids=feature['entity_ids'][rr]
                    for a,t in enumerate(targets.tolist()):
                        key=(w['scene_token'],w['sample_token'],w['instance_tokens'][t]);oldgroup=groups[key]
                        assert int(oldgroup['node_in_graph'])==t
                        trajectory=torch.cat((w['current_position'][t,None],w['GT'][t]),0).numpy()
                        vin=trajectory[2]-trajectory[0];vout=trajectory[-1]-trajectory[-3];disp=float(np.linalg.norm(trajectory[-1]-trajectory[0]))
                        eligible=int(w['agent_type'][t])==0 and disp>5 and np.linalg.norm(vin)>.5 and np.linalg.norm(vout)>.5
                        heading=float(np.degrees(np.arctan2(np.sin(np.arctan2(vout[1],vout[0])-np.arctan2(vin[1],vin[0])),
                            np.cos(np.arctan2(vout[1],vout[0])-np.arctan2(vin[1],vin[0]))))) if eligible else 0.
                        assert int(eligible and abs(heading)>20)==int(oldgroup['TurningVehicle_GT'])
                        mask=rawargs[5][a];entityty=rawargs[3][a,:,:6].argmax(-1);ty=int(w['agent_type'][t])
                        row={'scene_token':key[0],'sample_token':key[1],'instance_token':key[2],'dataset_index':w['dataset_index'],'node_in_graph':t,
                            'agent_type':CLASSES[ty],'motion_state':w['motion_state'][t],'GT_endpoint_displacement_m':disp,'GT_heading_change_deg':heading,
                            **{g:int(oldgroup[g]) for g in ('IntersectionVehicle20','NearTurnConnector20','NearTrafficControl20','TurningVehicle_GT')},
                            'MapNonEmpty':int(mask.any()),'ZeroMap':int(not mask.any()),
                            'RouteCenterlineAvailable':int(ty in (0,2) and ((entityty<2)&mask).any()),
                            'PedSemanticSpecificAvailable':int(ty==1 and ((entityty>=4)&mask).any()),'best_mode':int(best[a])}
                        for v in MODELS:
                            row.update({v+'_'+m:float(x) for m,x in zip(METRICS,values[v][a])});row[v+'_top1_mode']=int(tops[v][a])
                        rows.append(row)
                        for v in ('G2','G3'):
                            k=int(tops[v][a]);alpha=attentions[v]['map_alpha'][a,k].numpy();valid=mask[k].numpy();tys=entityty[k].numpy()
                            for et in range(6):diagnostic.append({'Model':v,'Branch':'map','ActorType':CLASSES[ty],'Category':TYPES[et],
                                'Weight':float(alpha[(tys==et)&valid].sum()),'Targets':1})
                            if valid.any():
                                s=int(np.argmax(alpha));eid=int(entityids[a,k,s]);entry=index.dictionary[eid]
                                entityrows.append({'scene_token':key[0],'sample_token':key[1],'instance_token':key[2],'Model':v,'Mode':k,
                                    'EntityId':eid,'EntityToken':entry['token'],'EntityType':entry['entity_type'],'Attention':float(alpha[s])})
                        for v in ('G1','G3'):
                            k=int(tops[v][a]);alpha=attentions[v]['interaction_alpha'][a,k].numpy().reshape(8,6)
                            nt=rawargs[0][a,1:,0,:3].argmax(-1).numpy();valid=rawargs[2][a].numpy()
                            for ntp in range(3):diagnostic.append({'Model':v,'Branch':'interaction','ActorType':CLASSES[ty],
                                'Category':CLASSES[ty]+'→'+CLASSES[ntp],'Weight':float(alpha[(nt==ntp)&valid].sum()),'Targets':1})
                            distance=rawargs[1][a,k,:,:,0].numpy();bins=[0,2,5,10,20,50,float('inf')]
                            for lo,hi in zip(bins[:-1],bins[1:]):diagnostic.append({'Model':v,'Branch':'distance','ActorType':CLASSES[ty],
                                'Category':f'[{lo},{hi})m','Weight':float(alpha[(distance>=lo)&(distance<hi)&valid[:,None]].sum()),'Targets':1})
                    windows.append({'dataset_index':w['dataset_index'],'scene_token':w['scene_token'],'sample_token':w['sample_token'],
                        'targets':targets,'probabilities':{v:p.cpu() for v,p in probabilities.items()},'fde':fde.cpu(),'ade':ade.cpu(),
                        'raw_prediction_sha256':w['raw_prediction_sha256'],'ego_prediction_sha256':w['ego_prediction_sha256'],
                        'interaction_alpha':{v:attentions[v]['interaction_alpha'] for v in ('G1','G3')},
                        'map_alpha':{v:attentions[v]['map_alpha'] for v in ('G2','G3')}})
            payload={'registration_sha256':sha256(register),'rows':rows,'diagnostics':diagnostic,'top_entities':entityrows,'windows':windows,'capture_maxdiff':maxdiff}
            atomic_torch(path,payload)
        allrows.extend(payload['rows']);topentities.extend(payload['top_entities'])
        for d in payload['diagnostics']:
            key=tuple(d[k] for k in ('Model','Branch','ActorType','Category'));s=diag_sums.setdefault(key,[0.,0])
            s[0]+=d['Weight'];s[1]+=d['Targets']
        capturemax=max(capturemax,payload['capture_maxdiff']);validated+=len(payload['windows'])
        if (bi+1)%10==0:print('UNIFIED_VAL',bi+1,'/',len(records),'FULL',len(allrows),flush=True)
    assert len(allrows)==54990 and len({r['scene_token'] for r in allrows})==150
    assert len({(r['scene_token'],r['sample_token'],r['instance_token']) for r in allrows})==54990
    assert sum(r['TurningVehicle_GT'] for r in allrows)==1663
    write_csv(dest/'stage8a1_actor_results.csv',allrows)
    # Pool target-level masses; no causal claims. Keep per-entity sidecar local.
    pooled=[dict(zip(('Model','Branch','ActorType','Category'),key),AttentionMass=s[0]/s[1],Count=s[1]) for key,s in sorted(diag_sums.items())]
    write_csv(ROOT/'06_tables/stage8a1_attention_diagnostics.csv',pooled)
    write_csv(ROOT/'05_diagnostics/stage8a1_top_entities.csv',topentities)
    assert state_sha(predictor)==predictorsha and not any(p.grad is not None or p.requires_grad for p in predictor.parameters());verify()
    atomic_json(dest/'stage8a1_candidate_identity.json',{'status':'PASS','targets':54990,'scenes':150,'shared_prediction_tensor':True,
        'fresh_all_windows_SHA_equals_frozen_cache':True,'candidate_bitwise_identity':True,'candidate_maxdiff':0.,
        'minADE6_maxdiff':0.,'minFDE6_maxdiff':0.,'MR6_maxdiff':0.,'attention_capture_maxdiff':capturemax,'predictor_gradient_count':0,
        'predictor_state_unchanged':True,'evaluated_nonempty_windows':validated})
    atomic_json(dest/'stage8a1_complete.json',{'status':'PASS','models':MODELS,'targets':54990,'VAL_scenes':150,
        'official_passes':1,'actor_csv_sha256':sha256(dest/'stage8a1_actor_results.csv'),'checkpoint_manifest_sha256':sha256(FROZEN)})
    ds.clear();print('UNIFIED_VAL_PASS',flush=True)

if __name__=='__main__':
    with torch.no_grad():main()
