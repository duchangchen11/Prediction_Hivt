"""Supplementary conditional head-initialization panel after all fits/eval freeze."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import stage16_figures as s
def main():
 assert s.read_json(s.ROOT/'03_seed_stability/stage16_seed_evaluation_integrity.json')['Status']=='PASS'
 d=s.pd.read_csv(s.SD/'stage16_seed_metrics.csv');d=d[d.Group=='Overall'];names=['NG-A','NG-C','G-C','Matched-NG-C'];fig,axs=s.plt.subplots(2,2,figsize=(7.2,4.7));s.title(fig,'s2  Ranking-head initialization sensitivity')
 for ax,fold,tag in zip(axs.flat,[1,2,3,0],list('abcd')):
  part=d[d.Fold==fold];n=int(part.Count.iloc[0]);scenes=int(part.Scenes.iloc[0])
  for j,name in enumerate(names):
   model=part[part.Model==name].sort_values('InitializationReplicate')
   for rep,marker in zip(range(3),['o','^','s']):
    row=model[model.InitializationReplicate==rep].iloc[0];ax.scatter(j+(rep-1)*.10,row.Top1FDE,s=19,marker=marker,facecolor='white' if rep==0 else s.COLORS[name],edgecolor=s.COLORS[name],lw=.8,zorder=4)
   ax.plot([j-.2,j+.2],[model.Top1FDE.mean()]*2,color=s.COLORS[name],lw=1.1)
  label='Pooled threefold configurations' if fold==0 else f'Fold{fold}'
  panel_title=f'{tag}  {label}\nn={n:,}, scenes={scenes}' if fold==0 else f'{tag}  {label}: n={n:,}, scenes={scenes}'
  ax.set_title(panel_title,loc='left',fontsize=7.1);ax.set_xticks(range(4),['NG-A','NG-C','G-C','Matched-NG-C'],rotation=18,ha='right');ax.set_ylabel('Top1FDE (m; lower is better)');ax.grid(axis='y',alpha=.15)
 fig.subplots_adjust(left=.10,right=.98,wspace=.30,hspace=.65,top=.85,bottom=.20)
 s.footer(fig,'nuScenes TRAIN630 | Custom scene-isolated 3-fold CV | n=260,151 actor windows /630 scenes\n○ original init; ▲ +1000; ■ +2000; line=mean. Three head initializations per fold, predictors and data order fixed.')
 s.save(fig,'figS2_head_initialization',['06_source_data/stage16_seed_metrics.csv','06_source_data/stage16_seed_summary.csv'],'Three conditional head initializations; pooled configurations align registered replicate indices, no predictor-seed robustness claim.')
 manifest=s.read_json(s.OUT/'stage16_figure_manifest.json');manifest['Figures']=[x for x in manifest['Figures'] if x['Figure']!='figS2_head_initialization']+s.EXPORT;manifest['Status']='EXPORTED_PENDING_FINAL_VISUAL_QA';s.atomic_json(s.OUT/'stage16_figure_manifest.json',manifest);print('S2 exported SVG/PDF/PNG')
if __name__=='__main__':main()
