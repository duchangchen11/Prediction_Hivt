"""Frozen-source quantitative panels and independently judged forecasting/mechanism."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'05_figures'))
from stage7d_cases import *
FIGURES=ROOT/'05_figures'
def comparison(name,groups,labels):
    ci=pd.read_csv(ROOT/'06_tables/stage7d_bootstrap_ci.csv');ci=ci[ci.Metric=='minFDE6'].set_index('Group').loc[groups]
    source=[{'Group':g,'label':l,**ci.loc[g].to_dict()} for g,l in zip(groups,labels)]
    write_csv(FIGURES/(name+'_source.csv'),source)
    fig,axes=plt.subplots(1,2,figsize=(183/25.4,85/25.4),gridspec_kw={'width_ratios':[1.15,1]},layout='constrained')
    y=np.arange(len(groups));a=axes[0];a.barh(y-0.16,ci.Stage3B,height=.29,color=BASE,label='Stage3B')
    a.barh(y+0.16,ci.Stage7D,height=.29,color=BLUE,label='Stage7D');a.set_yticks(y,labels);a.invert_yaxis()
    a.set_xlabel('minFDE6 (m), lower is better');a.legend(loc='upper right',fontsize=6)
    a.set_title('a  Independent trained models',loc='left',fontweight='bold',fontsize=8)
    a=axes[1];delta=ci.Delta.to_numpy();low=ci.CI_lower.to_numpy();high=ci.CI_upper.to_numpy()
    # Draw endpoints directly: percentile intervals need not contain the sample estimate.
    for j,d,l,h in zip(y,delta,low,high):a.plot([l,h],[j,j],color=BLUE,lw=1.2);a.scatter(d,j,color=BLUE,s=14,zorder=3)
    a.axvline(0,color='#777777',lw=.7,ls='--');a.set_yticks(y,labels);a.invert_yaxis();a.set_xlabel('Δ minFDE6: Stage7D − Stage3B (m)')
    a.set_title('b  Paired scene bootstrap, 95% CI',loc='left',fontweight='bold',fontsize=8)
    for a in axes:a.grid(axis='x',color='#E9E9E9',lw=.5);a.set_axisbelow(True)
    export(fig,name,'06_tables/stage7d_bootstrap_ci.csv;1000 paired150-scene replicates; one training seed each; secondary CIs unadjusted')

def attention():
    f=pd.read_csv(ROOT/'06_tables/stage7d_attention_relevance.csv')
    groups=['Vehicle','vehicle.moving','TurningVehicle_GT'];labels=['Vehicle','Moving vehicle','Turning vehicle (GT)']
    fig,ax=plt.subplots(figsize=(183/25.4,85/25.4),layout='constrained');x=np.arange(3);source=[]
    for offset,model,color in [(-.24,'Stage3B',BASE),(0,'Stage7A','#A8AFB2'),(.24,'Stage7D',BLUE)]:
        a=f[f.Model==model].set_index('Group').loc[groups]
        ax.bar(x+offset,a.GTRelevantMass2m,width=.21,color=color,label=model)
        source+=a.reset_index().to_dict('records')
    ax.set_xticks(x,labels);ax.set_ylabel('Mean GT-relevant attention mass (2m)');ax.legend(loc='upper left')
    ax.set_ylim(0,max(f[f.Group.isin(groups)].GTRelevantMass2m.max()*1.3,.01))
    ax.grid(axis='y',color='#EEEEEE',lw=.5);ax.set_axisbelow(True)
    ax.set_title('GT-relative lane attention: actor-pooled full-horizon VAL',loc='left',fontsize=9)
    write_csv(FIGURES/'stage7d_attention_relevance_source.csv',source)
    export(fig,'stage7d_attention_relevance','06_tables/stage7d_attention_relevance.csv; point estimates; exact group counts in source; Stage7A historical frozen reference')

def turns():
    f=pd.read_csv(ROOT/'06_tables/stage7d_turn_attention.csv');f=f[f.Group=='TurningVehicle_GT'].set_index('Model')
    metrics=['CorrectTurnMass','OppositeTurnMass','StraightMass'];labels=['Correct turn','Opposite turn','Straight']
    fig,ax=plt.subplots(figsize=(183/25.4,85/25.4),layout='constrained');x=np.arange(3)
    for offset,model,color in [(-.24,'Stage3B',BASE),(0,'Stage7A','#A8AFB2'),(.24,'Stage7D',BLUE)]:
        ax.bar(x+offset,f.loc[model,metrics].to_numpy(dtype=float),width=.21,color=color,label=model)
    ax.set_xticks(x,labels);ax.set_ylabel('Mean connector attention mass');ax.legend(loc='upper left')
    ax.set_ylim(0,f[metrics].to_numpy().max()*1.3);ax.grid(axis='y',color='#EEEEEE',lw=.5);ax.set_axisbelow(True)
    ax.set_title('TurningVehicle_GT: same 1663 full-horizon actor targets',loc='left',fontsize=9)
    f.reset_index().to_csv(FIGURES/'stage7d_turn_attention_source.csv',index=False)
    export(fig,'stage7d_turn_attention','06_tables/stage7d_turn_attention.csv; point estimates; GT direction offline only; one training seed')

def main():
    comparison('stage7d_main_fde_comparison',['Overall','Vehicle','Pedestrian','vehicle.moving'],['Overall','Vehicle','Pedestrian','Moving vehicle'])
    comparison('stage7d_difficult_vehicle_groups',['Vehicle >5m','NearTurnConnector20','TurningVehicle_GT'],['Vehicle >5m','Near turn20','Turning vehicle (GT)'])
    attention();turns();print('QUANTITATIVE_FIGURES_COMPLETE',flush=True)

if __name__=='__main__':main()
