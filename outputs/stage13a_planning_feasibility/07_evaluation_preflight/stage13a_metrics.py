"""Open-loop engineering proxies with masks, footprints and actual elapsed times."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'00_manifest'))
from stage13a_common import *
import shapely
def headings(path,current,initial):
    # path [N,K,T,2], current[N,2], initial[N]; static segments retain heading.
    delta=np.diff(np.concatenate([np.broadcast_to(current[:,None,None],(*path.shape[:2],1,2)),path],2),axis=2)
    moving=np.linalg.norm(delta,axis=-1)>1e-6;angle=np.arctan2(delta[...,1],delta[...,0]);out=angle.copy();prev=np.broadcast_to(initial[:,None],path.shape[:2]).copy()
    for j in range(path.shape[2]):prev=np.where(moving[...,j],angle[...,j],prev);out[...,j]=prev
    return out
def interpolate(path,angle,times,valid,current,current_angle,maxstep=.05):
    # Include t0 and sample each keyframe interval. Each future segment requires
    # its two endpoints; an unknown endpoint yields unknown collision coverage.
    pp=np.concatenate([current[...,None,:],path],-2);hh=np.concatenate([current_angle[...,None],angle],-1)
    vm=np.concatenate([np.ones((*valid.shape[:-1],1),bool),valid],-1);clock=np.r_[0.,times];positions=[];angles=[];masks=[];newtimes=[]
    for j in range(12):
        count=int(np.ceil((clock[j+1]-clock[j])/maxstep));u=np.arange(1,count+1)/count
        positions.append(pp[...,j,None,:]+u[:,None]*(pp[...,j+1,None,:]-pp[...,j,None,:]))
        angles.append(hh[...,j,None]+u*wrap_angle(hh[...,j+1,None]-hh[...,j,None]))
        masks.append(np.broadcast_to((vm[...,j]&vm[...,j+1])[...,None],(*vm.shape[:-1],count)))
        newtimes.extend((clock[j]+u*(clock[j+1]-clock[j])).tolist())
    return np.concatenate(positions,-2),np.concatenate(angles,-1),np.concatenate(masks,-1),np.array(newtimes)
def box_overlap(ego,eh,agents,ah,ego_size,agent_sizes):
    # Vectorized separating-axis test on both oriented rectangles' two axes.
    diff=agents-ego;ce=np.cos(eh);se=np.sin(eh);ca=np.cos(ah);sa=np.sin(ah);c=np.abs(np.cos(ah-eh));s=np.abs(np.sin(ah-eh))
    el,ew=np.asarray(ego_size)/2;agent_sizes=np.asarray(agent_sizes);al=agent_sizes[...,0]/2;aw=agent_sizes[...,1]/2
    return (np.abs(diff[...,0]*ce+diff[...,1]*se)<=el+al*c+aw*s)&(np.abs(-diff[...,0]*se+diff[...,1]*ce)<=ew+al*s+aw*c)&(
        np.abs(diff[...,0]*ca+diff[...,1]*sa)<=al+el*c+ew*s)&(np.abs(-diff[...,0]*sa+diff[...,1]*ca)<=aw+el*s+ew*c)
def collision_proxy(obs,plan,agent_path,agent_heading,agent_mask,probabilities=None,ego_heading=None):
    n,k=agent_path.shape[:2];times=obs.future_times
    ea=headings(plan[None,None],obs.ego_history[-1][None],np.array([obs.ego_history_heading[-1]]))[0,0] if ego_heading is None else ego_heading
    ep,eh,em,clock=interpolate(plan,ea,times,np.ones(12,bool),obs.ego_history[-1],np.array(obs.ego_history_heading[-1]))
    current=np.broadcast_to(obs.other_agent_current_position[:,None],(n,k,2));heading=np.broadcast_to(obs.other_agent_current_heading[:,None],(n,k))
    ap,ah,am,at=interpolate(agent_path,agent_heading,times,agent_mask,current,heading);assert np.array_equal(at,clock)
    size=obs.other_agent_sizes_length_width[:,None,None,:];overlap=box_overlap(ep,eh,ap,ah,read_json(PROTOCOL)['Collision']['EgoLengthWidth'],size)&am
    any_mode=overlap.any(-1);coverage=float(am.mean()) if am.size else 0.
    result=dict(AnyFootprintOverlap=bool(any_mode.any()),ActorModeOverlapCount=int(any_mode.sum()),ValidInterpolatedPairFraction=coverage,
        UnknownActorCount=int((~agent_mask.all(axis=(-1,-2))).sum()),InterpolationPoints=len(clock),MinimumGeometrySeparationM=None,
        PerActorModeOverlap=any_mode,OverlapTimeline=overlap,InterpolationTimes=clock)
    if probabilities is not None:
        choices=probabilities.argmax(-1);result['Top1Overlap']=bool(any_mode[np.arange(n),choices].any()) if n else False
        result['ModeWeightedOverlapScore']=float((any_mode*probabilities).sum())
    return result
def motion_metrics(path,times,current):
    pp=np.vstack([current,path]);tt=np.r_[0.,times];dt=np.diff(tt);v=np.diff(pp,axis=0)/dt[:,None];mid=(tt[:-1]+tt[1:])/2
    a=np.diff(v,axis=0)/np.diff(mid)[:,None];amid=(mid[:-1]+mid[1:])/2;j=np.diff(a,axis=0)/np.diff(amid)[:,None]
    return dict(MeanSpeedMPS=float(np.linalg.norm(v,axis=-1).mean()),MaxSpeedMPS=float(np.linalg.norm(v,axis=-1).max()),
        MeanAccelerationMPS2=float(np.linalg.norm(a,axis=-1).mean()),MaxAccelerationMPS2=float(np.linalg.norm(a,axis=-1).max()),
        MeanJerkProxyMPS3=float(np.linalg.norm(j,axis=-1).mean()),MaxJerkProxyMPS3=float(np.linalg.norm(j,axis=-1).max()))
def road_proxy(obs,path,ego_heading=None):
    angle=headings(path[None,None],obs.ego_history[-1][None],obs.ego_history_heading[-1:])[0,0] if ego_heading is None else ego_heading
    p,h,_,_=interpolate(path,angle,obs.future_times,np.ones(12,bool),obs.ego_history[-1],obs.ego_history_heading[-1])
    half=np.array(read_json(PROTOCOL)['Collision']['EgoLengthWidth'])/2;corners=np.array([[half[0],half[1]],[half[0],-half[1]],[-half[0],-half[1]],[-half[0],half[1]]])
    cc=np.cos(h);ss=np.sin(h);rotation=np.stack([cc,-ss,ss,cc],-1).reshape(-1,2,2);xy=corners[None]@rotation.transpose(0,2,1)+p[:,None]
    b=obs.map_geometry['patch_bounds'];inside=((xy[...,0]>=b[0])&(xy[...,0]<=b[1])&(xy[...,1]>=b[2])&(xy[...,1]<=b[3])).all(-1)
    polygons=shapely.polygons(xy);onroad=shapely.covers(obs.map_geometry['drivable'],polygons)
    return dict(DrivableAreaViolationFraction=float((~onroad[inside]).mean()) if inside.any() else None,
        DrivableEvaluationCoverage=float(inside.mean()),DrivableFootprintChecks=len(p),MapMissing=not obs.map_geometry['available'])
def evaluate(obs,labels,plan,reference):
    mask=labels.ego_future_mask;errors=np.linalg.norm(plan[mask]-labels.ego_future_gt[mask],axis=-1);eligible=bool(mask.all())
    pred=obs.other_agent_predictions;angles=headings(pred,obs.other_agent_current_position,obs.other_agent_current_heading)
    ego_heading=labels.ego_future_heading_gt if reference=='RecordedEgoFuture_EvaluationOnly' and eligible else None
    pm=np.broadcast_to(obs.prediction_valid_mask[:,None,None],pred.shape[:3]);pc=collision_proxy(obs,plan,pred,angles,pm,obs.other_agent_probabilities,ego_heading=ego_heading)
    gt=labels.agent_future_gt[:,None];gh=labels.agent_future_heading_gt[:,None];gm=labels.agent_future_mask[:,None]
    gc=collision_proxy(obs,plan,np.nan_to_num(gt),np.nan_to_num(gh),gm,ego_heading=ego_heading)
    row=dict(Reference=reference,EgoADE=float(errors.mean()) if len(errors) else None,EgoFDE=float(errors[-1]) if eligible else None,CompleteEgoEvaluation=eligible,
        EgoGTValidSteps=int(mask.sum()),PredictedAgentCollisionProxy=pc['AnyFootprintOverlap'],PredictedTop1CollisionProxy=pc['Top1Overlap'],
        ModeWeightedOverlapScore=pc['ModeWeightedOverlapScore'],PredictionActorCoverage=float(obs.prediction_valid_mask.mean()) if len(pred) else 0.,
        GTAgentCollisionProxy=gc['AnyFootprintOverlap'] if labels.agent_future_mask.any() else None,GTAgentValidTimestepFraction=float(labels.agent_future_mask.mean()) if len(pred) else 0.,
        GTAgentCompleteActorFraction=float(labels.agent_future_mask.all(-1).mean()) if len(pred) else 0.,GTInterpolatedPairCoverage=gc['ValidInterpolatedPairFraction'],
        GTUnknownActorCount=gc['UnknownActorCount'],SizeApproximationActors=int((obs.other_agent_size_source!='current_annotation').sum()),EgoSizeApproximation=True,
        ProxyInterpretation='open-loop footprint overlap approximation; unknown masks retained; not actual collision probability or closed-loop safety',
        **motion_metrics(plan,obs.future_times,obs.ego_history[-1]),**road_proxy(obs,plan,ego_heading))
    return row,pc,gc
def numerical_audit():
    dt=np.array([.47,.99,1.48,2.02,2.51,3.03,3.48,4.01,4.52,4.98,5.52,6.02]);path=np.c_[2*dt,np.zeros(12)];metrics=motion_metrics(path,dt,np.zeros(2));assert abs(metrics['MeanSpeedMPS']-2)<1e-12 and metrics['MaxAccelerationMPS2']<1e-12 and metrics['MaxJerkProxyMPS3']<1e-11
    assert bool(box_overlap(np.zeros(2),0.,np.array([2.,0.]),0.,[4.8,2.],[4.5,1.8]))
    assert not bool(box_overlap(np.zeros(2),0.,np.array([10.,0.]),0.,[4.8,2.],[4.5,1.8]))
    assert bool(box_overlap(np.zeros(2),0.,np.array([3.,0.]),np.pi/2,[4.8,2.],[4.5,1.8]))
    assert not bool(box_overlap(np.zeros(2),0.,np.array([3.5,0.]),np.pi/2,[4.8,2.],[4.5,1.8]))
    # Crossing happens between two keyframes; interpolation detects it although
    # both sampled centers are outside each other's footprint.
    p=np.zeros((12,2));a=np.tile(np.array([[[[-10.,0.]]]]),(1,1,12,1));a[0,0,0]=[10.,0.]
    ap,ah,am,clock=interpolate(a,np.zeros((1,1,12)),np.arange(1,13)*.5,np.ones((1,1,12),bool),np.array([[[-10.,0.]]]),np.zeros((1,1)))
    found=box_overlap(np.zeros_like(ap),np.zeros_like(ah),ap,ah,[4.8,2.],np.array([[[[4.5,1.8]]]]))&am;assert found.any()
    missing=np.zeros((1,1,12),bool);_,_,unknown,_=interpolate(a,np.zeros((1,1,12)),np.arange(1,13)*.5,missing,np.array([[[-10.,0.]]]),np.zeros((1,1)));assert not unknown.any()
    return dict(Status='PASS',ActualUnequalTimesUsed=True,CVSpeedMetersPerSecond=2,ZeroAccelerationJerkTest=True,OBBOverlapSeparatedCases=True,RotatedFootprintCases=True,
        BetweenKeyframesCrossingDetected=True,MissingGTNotEmptyObstacle=True,CollisionGeometry='four-axis rectangle SAT, wrapped-heading interpolation<=.05s',
        EgoDimensionsApproximate=True,LowFrequency='source~2Hz; interpolation adds no measured dynamics; no high-frequency or closed-loop claim')
