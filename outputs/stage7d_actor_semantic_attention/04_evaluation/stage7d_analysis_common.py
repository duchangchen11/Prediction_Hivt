"""Paired, identity-checked offline evaluation; no diagnostic GT enters forward."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'00_manifest'))
from stage7d_common import *
import numpy as np
import pandas as pd
import torch
STAGE7A=ROOT.parent/'stage7a_semantic_map_hivt'
STAGE7C=ROOT.parent/'stage7c_actor_semantic_relevance_audit'
IDS=['scene_token','sample_token','instance_token','horizon']
FULL_IDS=IDS[:-1]
AUDIT=['node_in_graph','GT_trajectory_sha256','future_mask_bits','agent_type_id','agent_type','motion_state','valid_future_steps']
MAIN_GROUPS=['Overall','Vehicle','Pedestrian','Bicycle','Vehicle >5m','Pedestrian <5m','Pedestrian >5m']
MOTION_GROUPS=['vehicle.moving','vehicle.stopped','vehicle.parked','unknown']
SEMANTIC_GROUPS=['IntersectionVehicle20','NearTurnConnector20','NearTrafficControl20','TurningVehicle_GT','GT-left','GT-right']
FINAL_GROUPS=MAIN_GROUPS+MOTION_GROUPS+SEMANTIC_GROUPS
ATTENTION_METRICS=['GTRelevantMass2m','GTRelevantMass4m','GTNearestAttentionWeight','GTNearestAttentionRank',
    'Top1LaneGTDistance','Top3MinGTDistance','Top5MinGTDistance','AttentionEntropy']
TURN_METRICS=['CorrectTurnMass','OppositeTurnMass','StraightMass','UnknownTurnMass','TopConnectorMatchRate']

def group_mask(f,g):
    v=f.agent_type=='vehicle';p=f.agent_type=='pedestrian'
    if g=='Overall':return np.ones(len(f),bool)
    if g in ('Vehicle','Pedestrian','Bicycle'):return (f.agent_type==g.lower()).to_numpy()
    if g in MOTION_GROUPS:return (v&(f.motion_state==g)).to_numpy()
    if g=='Vehicle >5m':return (v&(f.GT_endpoint_displacement_m>5)).to_numpy()
    if g in ('Pedestrian <5m','Pedestrian >5m'):return (p&((f.GT_endpoint_displacement_m<5) if '<' in g else (f.GT_endpoint_displacement_m>5))).to_numpy()
    if g in ('GT-left','GT-right'):return ((f.TurningVehicle_GT==1)&((f.GT_heading_change_deg>0) if g=='GT-left' else (f.GT_heading_change_deg<0))).to_numpy()
    return (f[g]==1).to_numpy()

def paired_frames():
    frames=[]
    for name in ('baseline','on'):
        f=pd.read_csv(ROOT/f'04_evaluation/stage7d_{name}_actor_errors.csv',dtype={'future_mask_bits':str})
        assert len(f)==85027 and not f.duplicated(IDS).any()
        frames.append(f.set_index(IDS).sort_index())
    b,d=frames;assert b.index.equals(d.index)
    for k in AUDIT+['GT_endpoint_displacement_m']:assert (b[k].to_numpy()==d[k].to_numpy()).all(),k
    for f in frames:
        assert np.isfinite(f[list(METRICS)].to_numpy()).all()
        assert (f.index.get_level_values('horizon')=='full_horizon').sum()==54990
    f=b.reset_index()
    for k in METRICS:
        f['Stage3B_'+k]=b[k].to_numpy();f['Stage7D_'+k]=d[k].to_numpy()
    side=pd.read_csv(STAGE7A/'01_data_audit/stage7a_formal_actor_val_groups.csv',dtype={'future_mask_bits':str})
    full=f[f.horizon=='full_horizon'].set_index(IDS);s=side.set_index(IDS).loc[full.index]
    for k in ['GT_trajectory_sha256','future_mask_bits','agent_type_id']:assert (full[k].to_numpy()==s[k].to_numpy()).all()
    f=f.merge(side[IDS+SEMANTIC_GROUPS[:-2]],on=IDS,how='left',validate='one_to_one')
    old=pd.read_csv(STAGE7C/'02_relevance_analysis/stage7c_actor_attention.csv',dtype={'future_mask_bits':str})
    f=f.merge(old[FULL_IDS+['GT_heading_change_deg','TurnOptionCount20']].assign(horizon='full_horizon'),on=IDS,how='left',validate='one_to_one')
    f['DeltaMinFDE']=f.Stage7D_minFDE6-f.Stage3B_minFDE6
    f['DeltaTop1FDE']=f.Stage7D_Top1FDE6-f.Stage3B_Top1FDE6
    atomic_json(ROOT/'04_evaluation/stage7d_pairing_audit.json',{'status':'PASS','total':len(f),'full':54990,'partial':30037,
        'NaN':0,'Inf':0,'identity_fields':IDS+AUDIT,'GT_sidecar_frozen':True,'Stage7C_turn_signs_reused':True,
        'Stage3B_checkpoint_sha256':sha256(BASE_BEST),'Stage7D_checkpoint_sha256':sha256(BEST)})
    return f

def bootstrap_row(f,mask,metric,draws,scene_index,group,prefix_b='Stage3B_',prefix_d='Stage7D_'):
    b=f[prefix_b+metric].to_numpy(dtype=float);d=f[prefix_d+metric].to_numpy(dtype=float)
    mask=mask&np.isfinite(b)&np.isfinite(d);idx=np.array([scene_index[t] for t in f.scene_token])
    counts=np.bincount(idx[mask],minlength=150)
    sums_b=np.bincount(idx[mask],weights=b[mask],minlength=150);sums_d=np.bincount(idx[mask],weights=d[mask],minlength=150)
    den=counts[draws].sum(1);assert (den>0).all()
    delta=(sums_d[draws].sum(1)-sums_b[draws].sum(1))/den
    low,high=np.quantile(delta,[.025,.975]);bm=sums_b.sum()/counts.sum();dm=sums_d.sum()/counts.sum()
    row={'Group':group,'Metric':metric,'Count':int(counts.sum()),'Stage3B':float(bm),'Stage7D':float(dm),
         'Delta':float(dm-bm),'CI_lower':float(low),'CI_upper':float(high),'Scenes':150,'NonemptyScenes':int((counts>0).sum()),
         'Replicates':1000,'Seed':2022,'Multiplicity':'unadjusted secondary' if (group,metric)!=('Overall','minFDE6') else 'primary'}
    sums=[{'scene_token':t,'Group':group,'Metric':metric,'Count':int(counts[i]),'Stage3B_sum':float(sums_b[i]),'Stage7D_sum':float(sums_d[i])} for t,i in scene_index.items()]
    return row,sums
