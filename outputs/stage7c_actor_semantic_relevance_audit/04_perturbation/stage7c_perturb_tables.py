"""Exact identity pairing for reused and OOD diagnostics, without model selection."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'00_manifest'))
from stage7c_common import *

def main():
    actors=pd.read_csv(ROOT/'02_relevance_analysis/stage7c_actor_attention.csv',dtype={'future_mask_bits':str}).set_index(IDS).sort_index()
    masks={'Overall':np.ones(len(actors),bool),'Vehicle':(actors.agent_type=='vehicle').to_numpy(),
        'MovingVehicle':((actors.agent_type=='vehicle')&(actors.motion_state=='vehicle.moving')).to_numpy(),
        'TurningVehicle_GT':(actors.TurningVehicle_GT==1).to_numpy()}
    frames={name:old_errors(name).set_index(IDS).sort_index() for name in ['Stage3B','Stage7A-ON','Stage7A-ZERO']}
    for name in ['SHUFFLE','CENTERED']:
        f=pd.read_csv(ROOT/f'04_perturbation/stage7c_{name.lower()}_actor_errors.csv',dtype={'future_mask_bits':str})
        assert len(f)==85027
        f=f[f.horizon=='full_horizon'].set_index(IDS).sort_index();assert len(f)==54990
        frames['Stage7A-'+name]=f
    records=[];cis=[]
    tokens=sorted(actors.index.get_level_values('scene_token').unique());assert len(tokens)==150
    mapping={t:i for i,t in enumerate(tokens)}
    indices=np.array([mapping[t] for t in actors.index.get_level_values('scene_token')])
    draws=np.random.default_rng(2022).integers(0,150,(1000,150))
    for model,f in frames.items():
        assert f.index.equals(actors.index)
        for k in ['GT_trajectory_sha256','future_mask_bits','agent_type_id','node_in_graph']:
            assert (f[k].to_numpy()==actors[k].to_numpy()).all(),(model,k)
        assert np.isfinite(f[['minFDE6','Top1FDE6']]).all().all()
        for group,mask in masks.items():
            records.append({'Model':model,'Group':group,'Count':int(mask.sum()),
                'minFDE':float(f.loc[mask,'minFDE6'].mean()),'Top1FDE':float(f.loc[mask,'Top1FDE6'].mean()),
                'status':'frozen independent baseline' if model=='Stage3B' else ('frozen same-checkpoint input ablation' if model in ('Stage7A-ON','Stage7A-ZERO') else 'OOD diagnostic, not baseline')})
            if model=='Stage7A-ON':continue
            count=np.bincount(indices[mask],minlength=150);den=count[draws].sum(1)
            for metric in ['minFDE6','Top1FDE6']:
                delta=f[metric].to_numpy()-frames['Stage7A-ON'][metric].to_numpy()
                sums=np.bincount(indices[mask],weights=delta[mask],minlength=150)
                values=sums[draws].sum(1)/den;lo,hi=np.quantile(values,[.025,.975])
                cis.append({'Model':model,'Reference':'Stage7A-ON','Group':group,'Metric':metric,
                    'Count':int(mask.sum()),'Delta':float(delta[mask].mean()),'CI_lower':float(lo),'CI_upper':float(hi),
                    'replicates':1000,'seed':2022,'unit':'paired whole-scene','interpretation':'OOD/exploratory; not method selection or causal test'})
    table=pd.DataFrame(records);table.to_csv(ROOT/'06_tables/stage7c_perturbation_results.csv',index=False)
    pd.DataFrame(cis).to_csv(ROOT/'06_tables/stage7c_perturbation_bootstrap_ci.csv',index=False)
    for model,key,expected in [('Stage3B','minFDE',1.3432604611189578),('Stage7A-ON','minFDE',1.3505458137541295),
        ('Stage7A-ON','Top1FDE',2.74948207715014),('Stage7A-ZERO','minFDE',1.3589283382660693)]:
        assert abs(float(table[(table.Model==model)&(table.Group=='Overall')][key].iloc[0])-expected)<1e-10
    atomic_json(ROOT/'04_perturbation/stage7c_pairing_audit.json',{'status':'PASS','models':list(frames),
        'paired_full_actor_windows':54990,'GT_ID_mask_type_node_identity':True,'NaN':0,'Inf':0,
        'ON_ZERO_reused':True,'historical_Stage7A_metrics_unchanged':True,'training':False})
    print(table.to_string(index=False),flush=True)

if __name__=='__main__':main()
