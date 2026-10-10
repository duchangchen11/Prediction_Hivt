"""Publication plots from frozen values and actual local coordinates only."""
from pathlib import Path
import sys,json,hashlib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'00_manifest'))
from stage16_common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,Circle,FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7,'axes.labelsize':7,'axes.titlesize':8,'xtick.labelsize':6.5,'ytick.labelsize':6.5,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False,'axes.linewidth':.6})
OUT=ROOT/'05_figures';SD=ROOT/'06_source_data'
COLORS={'R0':'#8B8E94','R2':'#595F68','NG-A':'#A1AFCC','NG-C':'#596FAD','G-A':'#9EBFB2','G-C':'#287D6F','Matched-NG-C':'#B3836B'}
LABELS={**dict.fromkeys(MODELS,''),**{m:m for m in MODELS},'Matched-NG-C':'Matched\nNG-C'}
MET=pd.read_csv(SD/'stage16_seven_model_metrics.csv');POOL=MET[MET.Fold=='Pooled'];CI=pd.read_csv(SD/'stage16_primary_bootstrap.csv');EXPORT=[]
def footer(fig,text):fig.text(.5,.012,text,ha='center',va='bottom',fontsize=5.8,color='#444444')
def title(fig,text):fig.suptitle(text,x=.03,y=.99,ha='left',fontsize=10,fontweight='bold')
def save(fig,name,sources,notes):
 fig.canvas.draw()
 for ext in ('svg','pdf','png'):fig.savefig(OUT/(name+'.'+ext),dpi=400,facecolor='white',bbox_inches='tight',pad_inches=.08)
 source_hashes={source:sha256(ROOT/source if (ROOT/source).exists() else PROJECT/source) for source in sources}
 EXPORT.append({'Figure':name,'SourceFiles':sources,'SourceSHA256':source_hashes,'Interpretation':notes,'DimensionsInches':fig.get_size_inches().tolist(),'Exports':{ext:{'path':f'05_figures/{name}.{ext}','sha256':sha256(OUT/(name+'.'+ext)),'bytes':(OUT/(name+'.'+ext)).stat().st_size} for ext in ('svg','pdf','png')}})
 plt.close(fig)
def box(ax,x,y,w,h,text,color='#F1F3F6',size=7):
 ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle='round,pad=.008,rounding_size=.015',facecolor=color,edgecolor='#68747D',linewidth=.6));ax.text(x,y,text,ha='center',va='center',fontsize=size)
def arrow(ax,a,b,color='#66717A',dash=False):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=7,lw=.75,color=color,linestyle='--' if dash else '-'))
def architecture():
 fig,ax=plt.subplots(figsize=(7.2,4.35));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off');title(fig,'a  Fixed candidate generator and regret-aware selection')
 box(ax,.12,.84,.22,.16,'Observed history (5 frames)\nAgent type + lane geometry',size=6.4)
 box(ax,.40,.84,.23,.16,'Original HiVT-64 backbone\nLocal + temporal encoders\nType embedding\nGlobal interaction modes',size=6.4)
 box(ax,.70,.84,.25,.16,'Stage5A decoder\nMotion-conditioned residual\nTwo experts + router\n6 candidates, 12 samples',size=6.5)
 arrow(ax,(.24,.84),(.275,.84));arrow(ax,(.525,.84),(.565,.84))
 box(ax,.39,.58,.22,.13,'Frozen candidate trajectories\nUnchanged geometry, K = 6','#E7EDF5',6.4)
 box(ax,.12,.58,.22,.13,'Original mode logits\nand probabilities\nPi from HiVT embeddings',size=6.1)
 arrow(ax,(.7,.75),(.5,.645));arrow(ax,(.29,.76),(.16,.655))
 box(ax,.69,.58,.25,.16,'Future Interaction Graph\nG1: 15D node / 17D edge\n≤8 neighbors, ≤50 m','#E5F0EA')
 arrow(ax,(.51,.58),(.56,.58));ax.plot([.12,.12,.63],[.645,.695,.695],color='#66717A',lw=.75);arrow(ax,(.63,.695),(.63,.67))
 box(ax,.69,.34,.25,.15,'Masked attention message\nLayerNorm → residual score\nz = original logits + Δz','#E5F0EA')
 arrow(ax,(.69,.49),(.69,.425));ax.plot([.12,.12,.84],[.515,.46,.46],color='#66717A',lw=.75);arrow(ax,(.84,.46),(.84,.425))
 box(ax,.39,.34,.22,.15,'Softmax ranking scores\nTop1 candidate selection\nBicycle: samefold R2','#E7EDF5')
 arrow(ax,(.56,.34),(.51,.34))
 box(ax,.69,.12,.25,.17,'Loss C (head training only)\nExpected normalized regret\nScale: max(1 m, mean regret)','#F4ECE6',6.4)
 box(ax,.39,.12,.22,.13,'Detached GT\nendpoint errors\nTraining / Dev labels','#F4ECE6',6.5)
 arrow(ax,(.51,.12),(.56,.12),dash=True);arrow(ax,(.64,.255),(.66,.195),dash=True)
 ax.text(.04,.25,'Generator frozen for\nranking-head fitting.\nGT: training/Dev only,\nabsent from inference.',fontsize=6.3,ha='left',va='center')
 footer(fig,'Code schematic | nuScenes adaptation | K=6; 650,403 predictor + 24,066 G1 + 673 routed R2 parameters')
 save(fig,'fig1_architecture',['outputs/stage5a_motion_aware_decoder/00_manifest/stage5a_decoder.py','outputs/stage8a_future_scene_compatibility_graph/00c_sparse_type_aware_graph_spec/03_model_audit/stage8a0c_model.py','outputs/stage11b_error_aware_ranking/00_manifest/stage11b_common.py'],'Structural diagram, no experimental coordinate or performance claim.')
def graph():
 fig,ax=plt.subplots(figsize=(7.2,3.9));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off');title(fig,'b  Candidate-conditioned future interaction graph')
 xs=np.linspace(.075,.52,6)
 for k,x in enumerate(xs):
  for y,label,c in ((.75,f'k={k}',COLORS['G-C']),(.43,f'q={k}',COLORS['NG-C'])):
   ax.add_patch(Circle((x,y),.026,facecolor=c,edgecolor='white'));ax.text(x,y,label,ha='center',va='center',fontsize=6,color='white')
 for k in range(6):
  for q in range(6):ax.plot([xs[k],xs[q]],[.723,.457],color='#ADC3BA',lw=.45,alpha=.55,zorder=0)
 ax.text(.04,.88,'Target actor i: six 15D candidate nodes',fontsize=7.5)
 ax.text(.04,.33,'One neighbor j shown: six 15D nodes\nActual neighborhood: ≤8 actors, t0 distance ≤50 m',fontsize=7.1)
 ax.text(.04,.58,'Directed pair relations: 6 × 6; 17D each',fontsize=7,bbox={'facecolor':'white','edgecolor':'none','pad':2})
 box(ax,.79,.75,.30,.16,'For target mode k:\n47D message input = 15+15+17\n145D attention input = 64+64+17','#E5F0EA',6.8)
 box(ax,.79,.49,.30,.16,'Masked softmax over ≤48 modes\n64D weighted interaction message\nAdd to target encoding → score','#E5F0EA',6.8)
 arrow(ax,(.56,.74),(.63,.74));arrow(ax,(.79,.66),(.79,.585))
 ax.text(.045,.15,'Edges summarize predicted pair geometry: proximity, timing, relative heading/displacement,\noriginal probabilities and participant types. They do not contain future GT.',fontsize=7)
 footer(fig,'nuScenes adaptation | Structural illustration, not a case trajectory | G1: candidate relations, no joint-mode output')
 save(fig,'fig2_future_interaction_graph',['outputs/stage8a_future_scene_compatibility_graph/00_manifest/stage8a_graph.py','outputs/stage8a_future_scene_compatibility_graph/00c_sparse_type_aware_graph_spec/03_model_audit/stage8a0c_model.py'],'One illustrative neighbor drawn; actual scoring aggregates all valid neighbor modes with the documented mask.')
def seven():
 fig,axs=plt.subplots(1,2,figsize=(7.2,3.4),sharey=True);title(fig,'c  Seven-model comparison on shared frozen candidates')
 for ax,metric,tag in zip(axs,['Top1FDE','Top1ADE'],['a','b']):
  for j,name in enumerate(MODELS):
   row=POOL[(POOL.Group=='Overall')&(POOL.Model==name)].iloc[0];folds=MET[(MET.Group=='Overall')&(MET.Model==name)&(MET.Fold!='Pooled')]
   ax.scatter(folds[metric],[j]*len(folds),color=COLORS[name],marker='|',s=80,lw=1,alpha=.6)
   ax.scatter(row[metric],j,color=COLORS[name],s=30,zorder=4)
   ax.text(.98,j,f'{row[metric]:.3f}',transform=ax.get_yaxis_transform(),ha='right',va='center',fontsize=6.6,bbox={'facecolor':'white','edgecolor':'none','pad':.5})
  ax.set_yticks(range(7),[LABELS[x].replace('\n',' ') for x in MODELS]);ax.set_ylim(6.5,-.5);ax.set_xlabel(metric+' (m; lower is better)');ax.set_title(tag+'  Weighted pooled mean + three fold values',loc='left',fontsize=7)
  ax.grid(axis='x',alpha=.17)
 axs[0].set_xlim(2.02,2.72);axs[1].set_xlim(.91,1.18)
 fig.subplots_adjust(left=.16,right=.96,wspace=.20,top=.78,bottom=.20)
 footer(fig,'nuScenes TRAIN630 | Custom scene-isolated 3-fold CV | n=260,151 actor windows / 630 scenes\nShared oracle minFDE6=1.266202 m; fixed K6 geometry. Points are means, ticks are folds (not confidence intervals).')
 save(fig,'fig3_seven_model_performance',['06_source_data/stage16_seven_model_metrics.csv'],'Pooled means are actor-window weighted; fold ticks show three existing fitted folds, not independent seed repeats.')
def ablation():
 fig,axs=plt.subplots(1,2,figsize=(7.2,3.25),gridspec_kw={'width_ratios':[1,1.1]});title(fig,'d  Graph × ranking objective: 2 × 2 ablation')
 values=np.array([[float(POOL[(POOL.Group=='Overall')&(POOL.Model==n)].Top1FDE.iloc[0]) for n in ['NG-A','NG-C']],[float(POOL[(POOL.Group=='Overall')&(POOL.Model==n)].Top1FDE.iloc[0]) for n in ['G-A','G-C']]])
 cmap=LinearSegmentedColormap.from_list('restrained',['#DEECE5','#F2F4F6','#C9D3E6']);im=axs[0].imshow(values,cmap=cmap,vmin=2.25,vmax=2.38,aspect='auto')
 for i in range(2):
  for j in range(2):axs[0].text(j,i,f'{values[i,j]:.6f}\nm',ha='center',va='center',fontsize=9)
 axs[0].set_xticks([0,1],['Loss A: soft CE','Loss C: normalized regret']);axs[0].set_yticks([0,1],['NoGraph','Graph G1']);axs[0].set_title('a  Overall Top1FDE (m)',loc='left',fontsize=8)
 main=CI[(CI.Group=='Overall')&(CI.Metric=='Top1FDE')]
 for j,cmp in enumerate(['G-C-NG-C','G-C-Matched-NG-C','NG-C-NG-A']):
  row=main[main.Comparison==cmp].iloc[0];axs[1].errorbar(row.Delta,j,xerr=[[row.Delta-row.BonferroniCILower],[row.BonferroniCIUpper-row.Delta]],fmt='o',ms=4,color=COLORS['G-C'] if j<2 else COLORS['NG-C'],capsize=3)
 axs[1].axvline(0,color='#666',ls=':',lw=.7);axs[1].set_yticks(range(3),['Graph: G-C − NG-C','Capacity: G-C − Matched','Loss: NG-C − NG-A']);axs[1].invert_yaxis();axs[1].set_xlabel('Paired ΔTop1FDE (m; negative favors first)');axs[1].set_title('b  Frozen co-primary contrasts',loc='left',fontsize=8);axs[1].grid(axis='x',alpha=.17)
 fig.subplots_adjust(left=.09,right=.97,wspace=.80,top=.78,bottom=.26)
 footer(fig,'nuScenes TRAIN630 | Custom scene-isolated 3-fold CV | n=260,151 actor windows / 630 scenes\n2,000 paired scene bootstrap draws, seed2022; family3 Bonferroni98.333% intervals (conditional on fitted models).')
 save(fig,'fig4_two_by_two_ablation',['06_source_data/stage16_seven_model_metrics.csv','06_source_data/stage16_primary_bootstrap.csv'],'Heatmap and primary adjusted intervals retain Stage15B selection and inference decisions.')
def participant():
 fig,axs=plt.subplots(2,2,figsize=(7.2,4.5));title(fig,'e  Participant and motion-state groups')
 for ax,group,tag in zip(axs.flat,['Vehicle','Pedestrian','Bicycle','MovingVehicle'],list('abcd')):
  sub=POOL[POOL.Group==group].set_index('Model');means=[sub.loc[n].Top1FDE for n in MODELS];count=int(sub.Count.iloc[0]);scenes=int(sub.Scenes.iloc[0])
  for j,n in enumerate(MODELS):ax.scatter(j,means[j],color=COLORS[n],s=22,zorder=3)
  ax.plot(range(7),means,color='#CCD0D4',lw=.8,zorder=1);ax.set_xticks(range(7),['R0','R2','NG-A','NG-C','G-A','G-C','M-NG-C'],rotation=30,ha='right');ax.set_ylabel('Top1FDE (m)');ax.set_title(f'{tag}  {group}: n={count:,}, scenes={scenes}',loc='left',fontsize=7.3);ax.grid(axis='y',alpha=.17)
  if group=='Bicycle':ax.text(.97,.92,'NG/G/Matched share R2 route',transform=ax.transAxes,ha='right',fontsize=6)
  if group=='MovingVehicle':ax.text(.97,.92,'t0 vehicle.moving attribute',transform=ax.transAxes,ha='right',fontsize=6)
 fig.subplots_adjust(left=.10,right=.98,wspace=.33,hspace=.70,top=.84,bottom=.18)
 footer(fig,'nuScenes TRAIN630 | Custom scene-isolated 3-fold CV | Fixed K6 candidates; n denotes actor windows\nSeparate y-axis scales for group readability; lower FDE is better. M-NG-C=Matched-NG-C.')
 save(fig,'fig5_multitype_groups',['06_source_data/stage16_seven_model_metrics.csv'],'Group axes have explicit distinct scales; Bicycle scoring policy is disclosed.')
def cases():
 manifest=read_json(ROOT/'07_cases/stage16_case_manifest.json');fig,axs=plt.subplots(2,2,figsize=(7.2,5.3));title(fig,'f  Actual beneficial and harmful mode switches')
 for ax,rec,tag in zip(axs.flat,manifest['Cases'],list('abcd')):
  csv=ROOT/rec['LocalCoordinateFile'];assert sha256(csv)==rec['CoordinateFileSHA256'];d=pd.read_csv(csv);current=np.array(rec['CurrentXY']);tr={key:x[['x_m','y_m']].to_numpy() for key,x in d.groupby('Entity',sort=False)}
  for k in range(6):xy=np.vstack((current,tr[f'candidate{k}']));ax.plot(xy[:,0],xy[:,1],color='#CBD0D5',lw=.6,zorder=1)
  h=tr['history'];ax.plot(h[:,0],h[:,1],':',color='#555',lw=1,label='Observed history')
  gt=np.vstack((current,tr['GT']));ax.plot(gt[:,0],gt[:,1],color='#181A1E',lw=1.3,label='GT',zorder=5)
  for name,style in [('NG-C','--'),('G-C','-')]:
   k=rec['SelectedModes'][name];xy=np.vstack((current,tr[f'candidate{k}']));ax.plot(xy[:,0],xy[:,1],style,color=COLORS[name],lw=1.3,label=f'{name} (mode{k})',zorder=4);ax.scatter(*xy[-1],s=15,marker='o' if name=='G-C' else 's',color=COLORS[name],zorder=6)
  ax.scatter(*current,s=13,color='#222',marker='x',zorder=6);ax.set_aspect('equal',adjustable='datalim');ax.set_xlabel('t0 ego x (m)');ax.set_ylabel('t0 ego y (m)');ax.set_title(f"{tag}  {rec['Group']} {rec['Outcome'].lower()}\nΔTop1FDE={rec['DeltaTop1FDE']:+.3f} m; fold{rec['Fold']}",loc='left',fontsize=7.4);ax.grid(alpha=.15);ax.legend(fontsize=5.4,loc='best');ax.margins(.12)
 fig.subplots_adjust(left=.10,right=.98,wspace=.28,hspace=.55,top=.84,bottom=.18)
 footer(fig,'nuScenes TRAIN630 | Custom scene-isolated 3-fold CV | 4 extrema from260,151 actor windows /630 scenes\nGT + unchanged six candidates; shared axes within each case, equal spatial aspect. Δ=G-C−NG-C; coordinates in meters.')
 save(fig,'fig6_improvement_failure_cases',['07_cases/stage16_case_manifest.json',*[r['LocalCoordinateFile'] for r in manifest['Cases']]],'Exact cached local coordinates and timestamp identity; extrema are not a prevalence sample; local coordinate CSVs excluded from Git.')
def reliability():
 d=pd.read_csv(SD/'stage16_reliability_bins.csv');d=d[(d.Fold==0)&(d.Group=='Overall')];metrics=pd.read_csv(SD/'stage16_probability_metrics.csv');metrics=metrics[(metrics.Fold==0)&(metrics.Group=='Overall')].set_index('Model')
 fig,axs=plt.subplots(1,2,figsize=(7.2,3.25));title(fig,'s1  Oracle-mode selection calibration and score concentration')
 axs[0].plot([0,1],[0,1],':',color='#555',lw=.8)
 for name in ['NG-A','NG-C','G-A','G-C']:
  x=d[(d.Model==name)&(d.Count>0)];axs[0].plot(x.MeanConfidence,x.OracleHitRate,'o-',ms=3,lw=.9,color=COLORS[name],label=f'{name}: ECE={metrics.loc[name].ECE15:.3f}')
 axs[0].set(xlim=(0,1),ylim=(0,1),xlabel='Mean Top1 probability in fixed bin',ylabel='FDE-oracle agreement rate');axs[0].set_title('a  Reliability: selected mode = endpoint oracle',loc='left',fontsize=7);axs[0].legend(fontsize=6,loc='upper left');axs[0].grid(alpha=.15)
 for j,name in enumerate(MODELS):
  value=metrics.loc[name].BrierScore;axs[1].bar(j,value,color=COLORS[name],width=.7);axs[1].text(j,value+.015,f'{value:.3f}',ha='center',fontsize=6)
 axs[1].set_xticks(range(7),['R0','R2','NG-A','NG-C','G-A','G-C','M-NG-C'],rotation=35,ha='right');axs[1].set(ylabel='Categorical oracle Brier score (0–2)',ylim=(0,1.10));axs[1].set_title('b  Oracle-label probability error (lower is better)',loc='left',fontsize=7);axs[1].grid(axis='y',alpha=.15)
 fig.subplots_adjust(left=.10,right=.98,wspace=.37,top=.80,bottom=.25)
 footer(fig,'nuScenes TRAIN630 | Custom scene-isolated 3-fold CV | n=260,151 actor windows / 630 scenes\n15 equal-width ECE bins; oracle one-hot labels. Scores are not calibrated future densities; no calibration fitting.')
 save(fig,'figS1_probability_quality',['06_source_data/stage16_probability_metrics.csv','06_source_data/stage16_reliability_bins.csv'],'Oracle-mode label is explicit; entropy/Brier in nats/dimensionless; no claim of calibrated physical-event probability.')
def main():
 architecture();graph();seven();ablation();participant();cases();reliability()
 atomic_json(OUT/'stage16_figure_manifest.json',{'Status':'EXPORTED_PENDING_VISUAL_QA','Backend':'Python/matplotlib','Figures':EXPORT,'ContractSHA256':sha256(OUT/'stage16_figure_contract.md')})
 print('Rendered',len(EXPORT),'figures × SVG/PDF/PNG')
if __name__=='__main__':main()
