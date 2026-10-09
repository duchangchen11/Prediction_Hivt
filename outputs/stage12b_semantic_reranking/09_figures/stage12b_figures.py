"""Registered development plots and ten error-selected pedestrian BEV cases."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'00_manifest'))
from stage12b_common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shapely
from shapely.plotting import plot_polygon
sys.path[:0]=[str(S8/'00c_sparse_type_aware_graph_spec/00_manifest'),str(S8/'00c_sparse_type_aware_graph_spec/01_selector')]
from stage8a0c_selector import SparseSemanticIndex
from stage8a0c_common import TYPES as MAP_TYPES
from preprocessing.coordinates import ego_to_global,global_to_ego
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
COLORS=['#555555','#227da4','#c7622c','#936eab'];files={}
def save(fig,stem):
    fig.tight_layout()
    for ext in ('png','pdf','svg'):
        path=ROOT/'09_figures'/f'{stem}.{ext}';fig.savefig(path,dpi=140,bbox_inches='tight')
        if ext=='svg':path.write_text('\n'.join(x.rstrip() for x in path.read_text().splitlines())+'\n')
        files[path.name]=sha256(path)
    plt.close(fig)
def source_function(name):
    p=S12A/'09_cases/stage12a_cases.py';node=next(n for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(p),'exec'),globals())
source_function('unique_scenes');source_function('draw_map')
def main():
    verify();assert read_json(ROOT/'06_oof_evaluation/stage12b_oof_identity_audit.json')['Status']=='PASS'
    curves=pd.read_csv(ROOT/'04_training/stage12b_training_curves.csv');met=pd.read_csv(ROOT/'06_oof_evaluation/stage12b_oof_metrics.csv')
    ci=pd.read_csv(ROOT/'07_bootstrap/stage12b_model_mean_ci.csv');boot=pd.read_csv(ROOT/'07_bootstrap/stage12b_bootstrap_ci.csv')
    fig,ax=plt.subplots(2,3,figsize=(10,5))
    for fold in (1,2,3):
        for j,name in enumerate(VARIANTS,1):
            d=curves[(curves.Fold==fold)&(curves.Model==name)];ax[0,fold-1].plot(d.Epoch,d.TrainLoss,color=COLORS[j],label=name)
            ax[1,fold-1].plot(d.Epoch,d.DevPedestrianTop1FDE,color=COLORS[j],label=name)
        ax[0,fold-1].set(title=f'Fold {fold}',ylabel='Train total loss',xlabel='Epoch');ax[1,fold-1].set(ylabel='InnerDev P Top1FDE (m)',xlabel='Epoch')
    ax[0,0].legend(frameon=False);save(fig,'stage12b_01_training_curves')
    fig,ax=plt.subplots(figsize=(6,4));d=ci[ci.Group=='Pedestrian'].set_index('Model').loc[list(MODELS)];means=d.Top1FDE.to_numpy()
    ax.bar(MODELS,means,color=COLORS,width=.55);ax.errorbar(np.arange(4),means,yerr=np.array([means-d.CI95Lower.to_numpy(),d.CI95Upper.to_numpy()-means]),fmt='none',ecolor='black',capsize=3)
    ax.set(ylabel='Pedestrian Top1FDE (m)',title='Development OOF; 95% descriptive scene intervals');ax.set_ylim(0,max(d.CI95Upper)*1.12);save(fig,'stage12b_02_pedestrian_fde')
    labels=['Pedestrian<5m','Pedestrian5-8m','Pedestrian>5m','Pedestrian_walkway_valid_1','Pedestrian_walkway_valid_0'];short=['GT <5m','GT 5–8m','GT >5m','Any walkway','No walkway']
    fig,ax=plt.subplots(figsize=(9,4));xx=np.arange(len(labels))
    for j,name in enumerate(MODELS):
        values=[float(met[(met.Group==g)&(met.Model==name)].Top1FDE.iloc[0]) for g in labels];ax.bar(xx+(j-1.5)*.19,values,width=.18,color=COLORS[j],label=name)
    ax.set(xticks=xx,xticklabels=short,ylabel='Top1FDE (m)',title='Fixed paired populations; GT groups are offline only');ax.legend(frameon=False,ncol=4);save(fig,'stage12b_03_motion_and_coverage')
    fig,ax=plt.subplots(figsize=(7,4));comparisons=['S-G','S-P','S-C0','G-C0','P-C0'];d=boot[boot.Group=='Pedestrian'].set_index('Comparison').loc[comparisons]
    vv=d.DeltaTop1FDE.to_numpy();ax.errorbar(vv,np.arange(5),xerr=np.array([vv-d.CI95Lower.to_numpy(),d.CI95Upper.to_numpy()-vv]),fmt='o',capsize=4,color='#c7622c')
    ax.axvline(0,color='#777777',lw=.8);ax.set(yticks=np.arange(5),yticklabels=comparisons,xlabel='Δ Pedestrian Top1FDE (m); negative improves',title='Real vs controls; 95% descriptive scene intervals')
    for idx,c in enumerate(comparisons[:2]):r=d.loc[c];ax.plot([r.FamilyAdjustedLower,r.FamilyAdjustedUpper],[idx+.08,idx+.08],color='#227da4',lw=2)
    save(fig,'stage12b_04_real_and_shuffled_controls')
    switches=pd.read_csv(ROOT/'08_diagnostics/stage12b_mode_switch_cost.csv');d=switches[(switches.Group=='Pedestrian')&switches.Comparison.isin(['G-C0','S-C0','P-C0'])].set_index('Comparison').loc[['G-C0','S-C0','P-C0']]
    fig,ax=plt.subplots(1,2,figsize=(9,4));xx=np.arange(3)
    ax[0].bar(xx-.16,d.improved_count,width=.3,color='#227da4',label='Improved');ax[0].bar(xx+.16,d.worsened_count,width=.3,color='#c7622c',label='Worsened');ax[0].set(xticks=xx,xticklabels=['G','S','P'],ylabel='Changed actors',title='C0 → variant');ax[0].legend(frameon=False)
    ax[1].bar(xx-.16,d.gross_gain,width=.3,color='#227da4',label='Gross gain');ax[1].bar(xx+.16,d.gross_harm,width=.3,color='#c7622c',label='Gross harm');ax[1].set(xticks=xx,xticklabels=['G','S','P'],ylabel='Sum FDE difference (m)',title='Fewer wrong switches can still cause more harm');ax[1].legend(frameon=False);save(fig,'stage12b_05_switch_counts_and_cost')
    dest=ROOT/'06_oof_evaluation/cache';f=pd.read_csv(dest/'stage12b_actor_records.csv');values=np.load(dest/'stage12b_metrics.npy',mmap_mode='r');top=np.load(dest/'stage12b_modes.npy',mmap_mode='r');prob=np.load(dest/'stage12b_probabilities.npy',mmap_mode='r')
    fd,ad=label_arrays();change=values[:,2,0]-values[:,0,0];ped=f.agent_type.to_numpy()=='Pedestrian';manifest=[]
    for kind,eligible,key in [('Gain',ped&(change<0),change),('Harm',ped&(change>0),-change)]:
        order=sorted(np.flatnonzero(eligible),key=lambda i:(float(key[i]),f.iloc[i].actor_id))
        for ordinal,i in enumerate(unique_scenes(order,f,5),1):
            r=f.iloc[i];manifest.append(dict(Category=kind,Ordinal=ordinal,HeadTrainIndex=i,ActorID=r.actor_id,SceneToken=r.scene_token,SampleToken=r.sample_token,
                Fold=int(r.Fold),C0Mode=int(top[i,0]),SMode=int(top[i,2]),C0FDE=float(values[i,0,0]),SFDE=float(values[i,2,0]),DeltaFDE=float(change[i])))
    dump('09_figures/stage12b_case_manifest.csv',manifest)
    index=SparseSemanticIndex();regions=sorted(index.regions);source=f.source_index.to_numpy();fields,semantic=semantic_source()
    geo=fields['GeometryFields'];walk=np.load(CACHE/'stage12b_walkway.npy',mmap_mode='r');candidate=np.load(S11A/'01_identity_audit/cache/candidates.npy',mmap_mode='r');gt=np.load(S11A/'01_identity_audit/cache/GT.npy',mmap_mode='r')
    origin=np.load(S12A/'01_map_integrity/cache/stage12a_origin.npy',mmap_mode='r');yaw=np.load(S12A/'01_map_integrity/cache/stage12a_yaw.npy',mmap_mode='r');entities=np.load(S12A/'03_feature_statistics/cache/stage12a_entity_ids.npy',mmap_mode='r')
    for r in manifest:
        i=r['HeadTrainIndex'];prediction=candidate[source[i]];truth=gt[source[i]];current=semantic['geometry'][i,0,[geo.index('current_x'),geo.index('current_y')]]
        pts=np.concatenate([prediction.reshape(-1,2),truth,current[None]]);low=pts.min(0)-8;high=pts.max(0)+8;bounds=(low[0],high[0],low[1],high[1])
        location=regions[int(semantic['map_region'][i])];selected=set(int(x) for x in entities[i].ravel() if x>=0)
        fig=plt.figure(figsize=(10,8));grid=fig.add_gridspec(3,1,height_ratios=[4,.8,1.4]);ax=fig.add_subplot(grid[0])
        handles=draw_map(ax,index,location,origin[i],float(yaw[i]),bounds,selected,'Pedestrian')
        for k in range(6):
            ax.plot(prediction[k,:,0],prediction[k,:,1],color='#888888',lw=1,alpha=.65,label='Six frozen candidates' if k==0 else None,zorder=3)
            ax.annotate(f'k{k}',prediction[k,-1],xytext=(2,3+(k%3)*7),textcoords='offset points',fontsize=7,zorder=6)
        for label,k,color,style in [('C0 Top1',r['C0Mode'],'#227da4','--'),('S Top1',r['SMode'],'#c7622c','-')]:ax.plot(prediction[k,:,0],prediction[k,:,1],color=color,ls=style,lw=2.3,zorder=4,label=f'{label} (k{k})')
        ax.plot(truth[:,0],truth[:,1],color='black',lw=2.4,zorder=5,label='GT: offline reference');ax.scatter(*current,c='black',marker='x',s=35,zorder=6,label='Observed t0')
        ax.set(xlim=(low[0],high[0]),ylim=(low[1],high[1]),xlabel='t0 ego x (m)',ylabel='t0 ego y (m)',title=f'P {r["Category"]} {r["Ordinal"]} | Fold{r["Fold"]} | S−C0 ΔFDE={r["DeltaFDE"]:+.3f}m\nActor: {r["ActorID"]}');ax.set_aspect('equal',adjustable='box')
        h,_=ax.get_legend_handles_labels();legend=fig.add_subplot(grid[1]);legend.axis('off');legend.legend(handles+h,[p.get_label() for p in handles+h],loc='center',ncol=3,fontsize=7,frameon=False)
        tab=fig.add_subplot(grid[2]);tab.axis('off');table=tab.table(cellText=[[f'k{k}',f'{fd[i,k]:.3f}',f'{prob[i,0,k]:.3f}',f'{prob[i,2,k]:.3f}',f'{walk[i,k,0]:.3f}',str(int(walk[i,k,1]))] for k in range(6)],
            colLabels=['Mode','FDE (m)','C0 raw p','S raw p','Walkway fraction','Walkway valid'],cellLoc='center',loc='center');table.auto_set_font_size(False);table.set_fontsize(8);table.scale(1,1.25)
        fig.suptitle('Stage12B development OOF: cases selected by gain/harm, without map filtering',fontsize=10)
        stem=f'stage12b_06_pedestrian_{r["Category"].lower()}_{r["Ordinal"]:02d}';save(fig,stem)
        atomic_json(ROOT/'09_figures'/f'{stem}.json',dict(Identity=r,Location=location,CoordinateFrame='t0 ego',Origin=origin[i].tolist(),Yaw=float(yaw[i]),
            CandidateGeometryIDs=semantic['candidate_geometry_id'][i].astype('U64').tolist(),Walkway=walk[i].tolist(),
            MapBackground='original visible HD Map components; display clipping only; not semantic fallback',GTOfflineOnly=True))
        print('STAGE12B_BEV_CASE',r['Category'],r['Ordinal'],flush=True)
    atomic_json(ROOT/'09_figures/stage12b_figure_audit.json',dict(Status='PASS',SummaryFigures=5,BEVCases=10,RegisteredFigureFamilies=6,Files=files,
        CaseSelection='5 largest gains and5 largest harms, actor-ID tie and distinct-scene preference, no map filter',CoordinateFrame='same t0 ego for GT and6 candidates and map'))
if __name__=='__main__':main()
