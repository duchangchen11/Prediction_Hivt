"""Population accounting and preregistered readiness, without any model code."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'00_manifest'))
from stage8a0b_common import *

def rate(x):return float(np.mean(x)) if np.size(x) else None
def summary(x):
    return {'MeanCount':float(np.mean(x)),'MedianCount':float(np.median(x)),
            'P95Count':float(np.percentile(x,95)),'MaxCount':int(np.max(x))}
def main():
    data={}
    for split in ('train','val'):
        with np.load(ROOT/'04_coverage'/f'stage8a0b_{split}_accounting.npz') as z:data[split]={k:z[k] for k in z.files}
    data['combined']={k:np.concatenate([data[s][k] for s in ('train','val')]) for k in data['train']}
    coverage=[];fallback=[];pedestrian=[];bicycle=[];entities=[];graph=[];bins=[];actor_fallback=[];contributions=[]
    historical_memory=read_json(STAGE8/'03_model_audit/stage8a_memory_audit.json')
    old_peak=max(r['PeakCUDAAllocatedMiB'] for r in historical_memory['rows'])
    # This is a resource budget, not an implemented feature vector or new G2/G3.
    budget_floats_node=64;budget_floats_edge=64
    bytes_per_edge=(budget_floats_node+budget_floats_edge)*4+4+1
    old_input=historical_memory['dense_padded_float32_input_bytes_per_target']
    old_map_input=6*8*(13*4+17*4+4+1)
    input_without_map=old_input-old_map_input
    dense_bytes=input_without_map+48*bytes_per_edge
    # Fourfold historical non-input inference peak plus a 32 MiB margin. Training unknown.
    estimated_peak=4*(old_peak-old_input*128/2**20)+dense_bytes*128/2**20+32
    gpu_total=int(torch.cuda.get_device_properties(0).total_memory) if torch.cuda.is_available() else 0
    graph_feasible=bool(gpu_total and estimated_peak*2**20<gpu_total)
    for split,d in data.items():
        for population,pmask in [('current_valid',np.ones(len(d['actor_type']),dtype=bool)),('full_horizon_ranking_targets',d['full'])]:
            masks=groups(d['actor_type'],d['motion'])
            vehicle_lane_fallback=((d['counts'][...,:2].sum(-1)==0)&masks['Vehicle'][:,None]&pmask[:,None]).sum()
            for name,gmask in masks.items():
                mask=gmask&pmask;count=int(mask.sum())
                if not count:
                    if name=='UnknownVehicle':
                        empty={k:None for k in coverage[0]};empty.update(Split=split,Population=population,ActorGroup=name,Actors=0,Candidates=0)
                        coverage.append(empty)
                        fr={k:None for k in fallback[0]};fr.update(Split=split,Population=population,MotionState='unknown',Actors=0,Candidates=0,LaneFallbackCandidates=0)
                        fallback.append(fr)
                        ar={k:None for k in actor_fallback[0]};ar.update(Split=split,Population=population,ActorGroup=name,Actors=0,Candidates=0)
                        actor_fallback.append(ar)
                        contributions.append({'Split':split,'Population':population,'ActorGroup':name,'Actors':0,'Candidates':0,
                            'LaneFallbackCandidates':0,'ShareOfVehicleLaneFallback':0.})
                    continue
                c=d['counts'][mask];c2=c[...,:6];route=c[...,:2].sum(-1)>0
                covered=((c2>0)&RELEVANT[d['actor_type'][mask],None,:]).any(-1)
                relevant=(c2*RELEVANT[d['actor_type'][mask],None,:]).sum(-1)
                generic=(c[...,2]>0)&(c2[...,[0,1,3,4,5]].sum(-1)==0)
                prefix={'Split':split,'Population':population,'ActorGroup':name,'Actors':count,'Candidates':count*6}
                row={**prefix,'RouteCenterlineCoverage':rate(route),'AreaContextCoverage':rate(c[...,[2,3]].sum(-1)>0),
                    'CrosswalkCoverage':rate(c[...,4]>0),'WalkwayCoverage':rate(c[...,5]>0),'CarparkCoverage':rate(c[...,3]>0),
                    'DrivableCoverage':rate(c[...,2]>0),'TypeAwareAnyCoverage':rate(covered),'TypeAwareFallbackRate':rate(~covered),
                    'GenericOnlyRate':rate(generic),'GenericOnlyAmongCoveredRate':float(generic.sum()/covered.sum()) if covered.sum() else None,
                    'MeanRelevantEntitiesPerMode':float(relevant.mean()),'MeanCappedRelevantEntitiesPerMode':float(np.minimum(relevant,8).mean()),
                    'AtLeast1LaneWithin10m':rate(route),'LaneFallbackRate':rate(~route),
                    'AtLeast4LaneTokens':rate(c[...,:2].sum(-1)>=4),'MeanLaneTokensPerMode':float(c[...,:2].sum(-1).mean()),
                    'MeanTop8LaneTokensBeforeFallback':float(d['lane_top8_count'][mask].mean())}
                coverage.append(row)
                actor_fallback.append({**prefix,'SixModesAllLaneCoveredRate':rate(route.all(-1)),
                    'AtLeastOneModeLaneFallbackRate':rate((~route).any(-1)),'AllSixModesLaneFallbackRate':rate((~route).all(-1)),
                    'SixModesAllTypeAwareCoveredRate':rate(covered.all(-1)),
                    'AtLeastOneModeTypeAwareFallbackRate':rate((~covered).any(-1)),
                    'AllSixModesTypeAwareFallbackRate':rate((~covered).all(-1)),
                    'SixModesAllLaneCoveredActors':int(route.all(-1).sum()),
                    'AtLeastOneModeLaneFallbackActors':int((~route).any(-1).sum()),
                    'AllSixModesLaneFallbackActors':int((~route).all(-1).sum()),
                    'SixModesAllTypeAwareCoveredActors':int(covered.all(-1).sum()),
                    'AtLeastOneModeTypeAwareFallbackActors':int((~covered).any(-1).sum()),
                    'AllSixModesTypeAwareFallbackActors':int((~covered).all(-1).sum())})
                for entity_id,entity_type in enumerate(TYPES):
                    values=c[...,entity_id].ravel()
                    entities.append({**prefix,'EntityType':entity_type,'MatchedCandidates':int((values>0).sum()),
                        'Coverage':rate(values>0),**summary(values),'PrimaryGraphEntity':entity_id<6})
                entities.append({**prefix,'EntityType':'total_distinct_primary','MatchedCandidates':int((c2.sum(-1)>0).sum()),
                    'Coverage':rate(c2.sum(-1)>0),**summary(c2.sum(-1).ravel()),'PrimaryGraphEntity':True})
                capped=np.minimum(relevant,8);edges=capped.sum(-1);input_bytes=input_without_map+edges*bytes_per_edge
                graph.append({'Split':split,'Population':population,'Group':name,'Targets':count,
                    'MeanMapEntitiesPerMode':float(capped.mean()),'P95MapEntitiesPerMode':float(np.percentile(capped,95)),
                    'MaxMapEntitiesPerMode':int(capped.max()),'UncappedMeanRelevantEntitiesPerMode':float(relevant.mean()),
                    'UncappedP95RelevantEntitiesPerMode':float(np.percentile(relevant,95)),
                    'OverflowRateTop8':rate(c2.sum(-1)>8),'RelevantOverflowRateTop8':rate(relevant>8),
                    'MeanModeMapEdges':float(edges.mean()),'P95ModeMapEdges':float(np.percentile(edges,95)),
                    'MaxModeMapEdges':int(edges.max()),'UncappedMeanModeMapEdges':float(relevant.sum(-1).mean()),
                    'UncappedP95ModeMapEdges':float(np.percentile(relevant.sum(-1),95)),
                    'UncappedMaxModeMapEdges':int(relevant.sum(-1).max()),
                    'EstimatedBytesPerTarget':float(input_bytes.mean()),'Batch128EstimatedInputBytes':int(np.ceil(input_bytes.mean()*128)),
                    'Batch128DenseMaxInputBytes':dense_bytes*128,'EstimatedPeakCUDAMiB':estimated_peak,
                    'MemoryEstimateIsMeasured':False})
                for bi,label in enumerate(('0-1m','1-5m','>5m')):
                    bm=d['net_bin'][mask]==bi;n=int(bm.sum())
                    bins.append({**prefix,'PredictedNetDisplacementBin':label,'BinCandidates':n,
                        'LaneOnlyFallbackRate':float((~route)[bm].mean()) if n else None,
                        'TypeAwareFallbackRate':float((~covered)[bm].mean()) if n else None})
                if name in ('Vehicle','MovingVehicle','StoppedVehicle','ParkedVehicle','UnknownVehicle'):
                    no_lane=~route;n=int(no_lane.sum());driv=no_lane&(c[...,2]>0);car=no_lane&(c[...,3]>0)
                    fallback.append({'Split':split,'Population':population,'MotionState':{'Vehicle':'all','MovingVehicle':'vehicle.moving',
                        'StoppedVehicle':'vehicle.stopped','ParkedVehicle':'vehicle.parked','UnknownVehicle':'unknown'}[name],
                        'Actors':count,'Candidates':count*6,'LaneFallbackCandidates':n,'LaneOnlyFallbackRate':rate(no_lane),
                        'TypeAwareFallbackRate':rate(~covered),'RecoveredByDrivableRate':float(driv.sum()/n) if n else None,
                        'RecoveredByCarparkRate':float(car.sum()/n) if n else None,
                        'RecoveredByEitherAreaRate':float((driv|car).sum()/n) if n else None,
                        'RecoveredByBothAreaRate':float((driv&car).sum()/n) if n else None,
                        'StillUncoveredRate':rate(~covered),'StillUncoveredAmongLaneFallbackRate':float((~covered).sum()/n) if n else None,
                        'RecoveredByDrivableUnconditionalRate':rate(driv),'RecoveredByCarparkUnconditionalRate':rate(car)})
                    if name!='Vehicle':contributions.append({**prefix,'LaneFallbackCandidates':n,
                        'ShareOfVehicleLaneFallback':float(n/vehicle_lane_fallback)})
                if name=='Pedestrian':
                    w=c[...,5]>0;x=c[...,4]>0;dr=c[...,2]>0
                    pedestrian.append({'Split':split,'Population':population,'Candidates':count*6,
                        'WalkwayCoverage':rate(w),'CrosswalkCoverage':rate(x),'DrivableCoverage':rate(dr),
                        'SemanticSpecificCoverage':rate(w|x),'TypeAwareAnyCoverage':rate(covered),'GenericOnlyRate':rate(generic),
                        'WalkwayOnlyRate':rate(w&~x&~dr),'CrosswalkOnlyRate':rate(x&~w&~dr),
                        'DrivableOnlyPrimaryRate':rate(dr&~w&~x),'MultiplePrimaryRate':rate(w.astype(int)+x.astype(int)+dr.astype(int)>=2),
                        'NoPrimaryContextRate':rate(~covered),'SecondaryRouteCoverage':rate(route)})
                if name=='Bicycle':bicycle.append({'Split':split,'Population':population,'Candidates':count*6,
                    'LaneCoverage':rate(c[...,0]>0),'ConnectorCoverage':rate(c[...,1]>0),'DrivableCoverage':rate(c[...,2]>0),
                    'TypeAwareAnyCoverage':rate(covered),'GenericOnlyRate':rate(generic),'DescriptiveCarparkCoverage':rate(c[...,3]>0)})
    tables={'coverage':coverage,'vehicle_fallback_decomposition':fallback,'pedestrian_semantic_coverage':pedestrian,
        'bicycle_semantic_coverage':bicycle,'entity_statistics':entities,'graph_estimate':graph,
        'candidate_displacement':bins,'actor_fallback':actor_fallback,'fallback_contributions':contributions}
    for name,rows in tables.items():write_csv(ROOT/'06_tables'/f'stage8a0b_{name}.csv',rows)
    gates=[]
    for split in ('train','val'):
        by={r['ActorGroup']:r for r in coverage if r['Split']==split and r['Population']=='full_horizon_ranking_targets'}
        for group in ('MovingVehicle','Vehicle','StoppedVehicle','ParkedVehicle','Pedestrian','Bicycle'):
            metric='RouteCenterlineCoverage' if group=='MovingVehicle' else 'TypeAwareAnyCoverage'
            gates.append({'Split':split,'Group':group,'Metric':metric,'Value':by[group][metric],'Threshold':.95,
                          'PASS':by[group][metric]>=.95})
        for group,threshold in [('Vehicle',.30),('Pedestrian',.50)]:
            gates.append({'Split':split,'Group':group,'Metric':'GenericOnlyRate','Value':by[group]['GenericOnlyRate'],
                          'Threshold':threshold,'PASS':by[group]['GenericOnlyRate']<=threshold})
    write_csv(ROOT/'06_tables/stage8a0b_readiness_gates.csv',gates)
    coverage_pass=all(g['PASS'] for g in gates if g['Metric']!='GenericOnlyRate')
    quality=all(g['PASS'] for g in gates if g['Metric']=='GenericOnlyRate')
    integ=read_json(ROOT/'04_coverage/stage8a0b_integrity_audit.json')
    schema=read_json(ROOT/'01_schema_audit/stage8a0b_schema_audit.json')
    combined={r['ActorGroup']:r for r in coverage if r['Split']=='combined' and r['Population']=='full_horizon_ranking_targets'}
    shares={r['ActorGroup']:r for r in contributions if r['Split']=='combined' and r['Population']=='full_horizon_ranking_targets'}
    candidate_bins={r['PredictedNetDisplacementBin']:r for r in bins if r['Split']=='combined' and r['Population']=='full_horizon_ranking_targets' and r['ActorGroup']=='Vehicle'}
    predicted_low_motion_support=(candidate_bins['0-1m']['LaneOnlyFallbackRate'] is not None
        and candidate_bins['>5m']['LaneOnlyFallbackRate'] is not None
        and candidate_bins['0-1m']['LaneOnlyFallbackRate']>candidate_bins['>5m']['LaneOnlyFallbackRate'])
    low_motion=(shares['StoppedVehicle']['ShareOfVehicleLaneFallback']+shares['ParkedVehicle']['ShareOfVehicleLaneFallback']>.5
        and combined['MovingVehicle']['LaneFallbackRate']<combined['Vehicle']['LaneFallbackRate'] and predicted_low_motion_support)
    missing_or_unusable=any(r['Status']=='MISSING' or int(r['UsableGeometry'])==0 for r in csv.DictReader((ROOT/'06_tables/stage8a0b_map_schema.csv').open()))
    # Any genuine residual uncovered candidate is a limit of the authorized map entity/threshold coverage.
    map_limitation=missing_or_unusable or combined['Vehicle']['TypeAwareFallbackRate']>0
    cause='MIXED' if low_motion and map_limitation else ('LOW_MOTION_OFF_CENTERLINE' if low_motion else ('MAP_LAYER_LIMITATION' if map_limitation else 'UNRESOLVED'))
    ready=coverage_pass and quality and graph_feasible and all(integ[k]=='PASS' for k in ['candidate_identity','coordinate','GT_leakage','semantic_attachment'])
    decision={'LaneOnlyFailureCause':cause,'TypeAwareSemanticCoverage':'PASS' if coverage_pass else 'FAIL',
        'SemanticContextQuality':'STRONG' if quality else 'WEAK','Stage8A_0B':'READY' if ready else 'NOT_READY',
        'historical_Stage8A_0':'NOT_READY','coverage_gates':gates,'graph_memory_feasible':graph_feasible,
        'inference_memory_budget_MiB':estimated_peak,'GPU_total_bytes':gpu_total,'memory_estimate_measured':False,
        'training_memory_feasibility':'not measured or authorized; model design required before any training claim',
        'low_motion_evidence':low_motion,'map_coverage_limitation_evidence':map_limitation,
        'predicted_low_motion_bin_support':predicted_low_motion_support,
        'no_training':True,'STOP':True,'training_authorized':False}
    atomic_json(ROOT/'00_manifest/stage8a0b_decision.json',decision)
    atomic_json(ROOT/'05_graph_estimate/stage8a0b_memory_budget.json',{'status':'PASS' if graph_feasible else 'FAIL',
        'kind':'conservative inference-only planning estimate, not benchmark','historical_measured_peak_MiB':old_peak,
        'resource_budget_floats_per_map_node':64,'resource_budget_floats_per_map_edge':64,
        'resource_budget_is_formal_feature_schema':False,'input_without_map_bytes_per_target':input_without_map,
        'dense_input_bytes_per_target':dense_bytes,'batch128_dense_input_bytes':dense_bytes*128,
        'estimated_peak_CUDA_MiB':estimated_peak,'formula':'4*(historical_peak - historical_batch_input) + expanded_dense_batch_input + 32 MiB',
        'GPU_total_bytes':gpu_total,'future_training_activations':'unmeasured','formal_G2_G3_implemented':False})
    print(json.dumps(decision,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
