"""Actual-data comparison figure and main performance table; Python only."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'00_manifest'))
from stage17_common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.text import Text
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7,'svg.fonttype':'none','pdf.fonttype':42,
 'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,'legend.frameon':False})
def save(fig,name):
 folder=ROOT/'08_figures';fig.canvas.draw();renderer=fig.canvas.get_renderer();outside=[]
 for obj in fig.findobj(Text):
  if obj.get_visible() and obj.get_text():
   box=obj.get_window_extent(renderer)
   if box.width>0 and box.height>0 and (box.x0<-.5 or box.y0<-.5 or box.x1>fig.bbox.width+.5 or box.y1>fig.bbox.height+.5):outside.append(obj.get_text())
 assert not outside,('Text outside canvas',outside)
 for ext in ('svg','pdf','png'):fig.savefig(folder/(name+'.'+ext),dpi=400,facecolor='white')
 plt.close(fig)
def main():
 protect();df=pd.read_csv(ROOT/'stage17_external_comparison.csv');bs=pd.read_csv(ROOT/'stage17_bootstrap_comparison.csv');pool=df[df.Fold=='Pooled']
 main=pool[pool.Group=='Overall'];assert len(main)==8 and main.Count.nunique()==1 and main.Count.iloc[0]==260151
 fig,axes=plt.subplots(1,2,figsize=(180/25.4,95/25.4),gridspec_kw={'width_ratios':[1.05,1.15]})
 fig.subplots_adjust(left=.16,right=.98,bottom=.24,top=.80,wspace=.68)
 names=main.Model.tolist();labels=['Adapted TNT' if n=='Adapted TNT Scoring' else n for n in names];values=main.Top1FDE.to_numpy();a=axes[0]
 colors=['#254F70' if n=='G-C' else '#AB733F' if n=='Adapted TNT Scoring' else '#8B989F' for n in names]
 for i,(val,c) in enumerate(zip(values,colors)):
  a.plot(val,i,'o',ms=4,color=c);a.text(val+.006,i,f'{val:.3f}',va='center',fontsize=6)
 a.set_yticks(range(8),labels);a.invert_yaxis();a.set_xlim(values.min()-.065,values.max()+.065);a.set_xlabel('Overall Top1FDE (m)');a.grid(axis='x',alpha=.15)
 a.set_title('a  Eight scoring methods',loc='left',weight='bold',fontsize=8)
 b=axes[1];gs=['Overall','Vehicle','Pedestrian','MovingVehicle'];gcounts=[int(pool[(pool.Model=='G-C')&(pool.Group==g)].Count.iloc[0]) for g in gs]
 for k,(name,col,marker) in enumerate([('G-C','#254F70','o'),('NG-C','#7198AA','s')]):
  sub=bs[(bs.Comparison==name+' - Adapted TNT Scoring')&(bs.Metric=='Top1FDE')].set_index('Group').loc[gs]
  y=np.arange(4)+(k-.5)*.16
  b.errorbar(sub.Delta,y,xerr=np.stack((sub.Delta-sub.CI95Lower,sub.CI95Upper-sub.Delta)),fmt=marker,color=col,ms=3,capsize=2,label=name+' − TNT')
 b.axvline(0,color='#707070',lw=.8,ls='--');b.set_yticks(range(4),[g.replace('MovingVehicle','Moving vehicle')+'\n'+f'n={n:,}' for g,n in zip(gs,gcounts)],fontsize=6)
 b.invert_yaxis();b.set_xlabel('Paired Top1FDE difference (m)');b.grid(axis='x',alpha=.15);b.set_title('b  Paired scene uncertainty',loc='left',weight='bold',fontsize=8)
 b.legend(loc='center left',bbox_to_anchor=(.01,.54),fontsize=6,ncol=1)
 # Explicitly retain only ticks within the view; inactive locator ticks are not drawn.
 for axis in axes:
  lo,hi=axis.get_xlim();axis.set_xticks([v for v in axis.get_xticks() if lo<=v<=hi])
 fig.text(.5,.96,'External trajectory scoring on frozen HiVT candidates',ha='center',weight='bold',fontsize=9)
 fig.text(.5,.895,'nuScenes • custom scene-isolated CV • K=6 • 12 future points',ha='center',fontsize=7)
 fig.text(.5,.075,'630 scenes; 260,151 actor-windows. Bars: 95% descriptive scene-cluster CI (2,000 draws).',ha='center',fontsize=6)
 fig.text(.5,.032,'Supplementary comparison; adapted scorer component, not a full TNT or official nuScenes test reproduction.',ha='center',fontsize=6)
 save(fig,'stage17_external_method_comparison')
 head=pd.read_csv(ROOT/'07_efficiency/stage17_head_compute.csv');whole=pd.read_csv(ROOT/'07_efficiency/stage17_whole_compute.csv');rows=[]
 for name in names:
  sub=pool[pool.Model==name].set_index('Group');h=head[head.Model==name];w=whole[whole.Model==name]
  rows.append({'Model':name,'Parameters':int(h.Parameters.iloc[0]),'Count':int(sub.loc['Overall','Count']),
   'OverallTop1FDE_m':sub.loc['Overall','Top1FDE'],'OverallTop1ADE_m':sub.loc['Overall','Top1ADE'],
   **{g+'Top1FDE_m':sub.loc[g,'Top1FDE'] for g in ('Vehicle','Pedestrian','MovingVehicle')},
   'minFDE6_m':sub.loc['Overall','minFDE6'],'HitRate':sub.loc['Overall','HitRate'],
   'MeanFoldHeadMSPer1024':float(h.MeanMSPer1024.mean()),'MeanFoldWholeSecondsPer16Windows':float(w.MeanSeconds.mean())})
 dump(ROOT/'stage17_main_performance_table.csv',rows)
 headings=['Method','Params','FDE\n(m)','ADE\n(m)','Vehicle\nFDE (m)','Pedestrian\nFDE (m)','Moving V.\nFDE (m)','minFDE6\n(m)','HitRate\n(%)','Head\nms/1024','Whole\ns/16 windows']
 cells=[]
 for r in rows:cells.append(['Adapted TNT' if r['Model']=='Adapted TNT Scoring' else r['Model'],f"{r['Parameters']:,}",
  *[f'{r[k]:.3f}' for k in ('OverallTop1FDE_m','OverallTop1ADE_m','VehicleTop1FDE_m','PedestrianTop1FDE_m','MovingVehicleTop1FDE_m','minFDE6_m')],
  f"{100*r['HitRate']:.2f}",f"{r['MeanFoldHeadMSPer1024']:.2f}",f"{r['MeanFoldWholeSecondsPer16Windows']:.3f}"])
 fig,ax=plt.subplots(figsize=(10.7,3.2));ax.axis('off');fig.subplots_adjust(left=.018,right=.988,top=.76,bottom=.22)
 table=ax.table(cellText=cells,colLabels=headings,cellLoc='center',loc='center',colWidths=[.14,.065,.067,.067,.085,.087,.085,.077,.075,.095,.105]);table.auto_set_font_size(False);table.set_fontsize(7);table.scale(1,1.42)
 for (row,col),cell in table.get_celld().items():
  cell.set_linewidth(.35);cell.set_edgecolor('#BFC6CB')
  if row==0:cell.set_facecolor('#E7EDF1');cell.set_text_props(weight='bold')
  elif names[row-1]=='G-C':cell.set_facecolor('#EDF4F8')
  elif names[row-1]=='Adapted TNT Scoring':cell.set_facecolor('#F8F1E8')
 fig.text(.5,.955,'Eight-model performance on identical six candidate trajectories',ha='center',weight='bold',fontsize=10)
 fig.text(.5,.87,'nuScenes custom scene-isolated CV; 630 scenes; Overall n=260,151; V n=191,026; P n=66,145; Moving V n=41,728',ha='center',fontsize=8)
 fig.text(.5,.115,'HitRate: selected mode equals endpoint-FDE oracle (not a distance-threshold success rate). Parameters: scoring head only.',ha='center',fontsize=7)
 fig.text(.5,.065,'Compute: equal fold means; cached head forward and full path on 16 fixed InnerDev windows. Excludes raw I/O; actor counts vary by fold.',ha='center',fontsize=7)
 fig.text(.5,.02,'Adapted TNT uses frozen HiVT context; the external comparison is supplementary and not an official leaderboard result.',ha='center',fontsize=7)
 save(fig,'stage17_main_performance_table')
 sources={p:sha256(ROOT/p) for p in ('stage17_external_comparison.csv','stage17_bootstrap_comparison.csv','stage17_main_performance_table.csv','07_efficiency/stage17_head_compute.csv','07_efficiency/stage17_whole_compute.csv')}
 atomic_json(ROOT/'08_figures/stage17_figure_source_receipt.json',{'Status':'PASS_TEXT_BOUNDS_AND_SOURCE','SourceSHA256':sources,'Exports':6,'EditableSVGText':True,'PDFType42':True,'PNG_DPI':400,'Backend':'Python matplotlib','VisualReview':'pending'})
if __name__=='__main__':main()
