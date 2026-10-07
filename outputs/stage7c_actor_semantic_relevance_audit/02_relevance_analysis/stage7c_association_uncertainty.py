"""Check weak actor correlations with paired scene resampling, without causal claims."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'00_manifest'))
from stage7c_common import *
from scipy.stats import spearmanr

def main():
    f=pd.read_csv(ROOT/'02_relevance_analysis/stage7c_actor_attention.csv');gg=groups(f)
    tokens=sorted(f.scene_token.unique());mapping={t:i for i,t in enumerate(tokens)}
    scene=f.scene_token.map(mapping).to_numpy();draws=np.random.default_rng(2022).integers(0,150,(1000,150))
    multiplicities=np.array([np.bincount(row,minlength=150) for row in draws])
    rows=[]
    comparisons=[(g,'DeltaRelevantMass','DeltaMinFDE') for g in ['Overall Vehicle','vehicle.moving','TurningVehicle_GT','GT-left','GT-right']]
    comparisons.append(('TurningVehicle_GT','DeltaCorrectTurnMass','DeltaMinFDE'))
    for group,x,y in comparisons:
        mask=gg[group]&np.isfinite(f[x])&np.isfinite(f[y]);a=f.loc[mask,x].to_numpy();b=f.loc[mask,y].to_numpy();s=scene[mask]
        indices=np.arange(len(a));values=[]
        for counts in multiplicities:
            ix=np.repeat(indices,counts[s])
            if len(ix)>2 and np.unique(a[ix]).size>1 and np.unique(b[ix]).size>1:values.append(float(spearmanr(a[ix],b[ix]).statistic))
        lo,hi=np.quantile(values,[.025,.975]);rho=float(spearmanr(a,b).statistic)
        rows.append({'Group':group,'X':x,'Y':y,'Count':len(a),'SpearmanR':rho,'CI_lower':float(lo),'CI_upper':float(hi),
            'replicates':1000,'valid_replicates':len(values),'seed':2022,'scenes':150,'unit':'whole-scene resampling with Spearman reranked in each replicate',
            'scope':'exploratory association, no causal or multiple-testing claim'})
        print('ASSOCIATION_BOOTSTRAP',rows[-1],flush=True)
    pd.DataFrame(rows).to_csv(ROOT/'06_tables/stage7c_attention_error_scene_bootstrap.csv',index=False)

if __name__=='__main__':main()
