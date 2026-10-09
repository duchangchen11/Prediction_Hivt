"""Thirty BEV cases chosen by the registered error/gain order, not semantics."""
from pathlib import Path
import sys
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'00_manifest'),str(Path(__file__).resolve().parents[1]/'03_feature_statistics')]
from stage12a_analysis_common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from shapely.plotting import plot_polygon
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'svg.fonttype':'none','pdf.fonttype':42,
    'axes.spines.top':False,'axes.spines.right':False})
def unique_scenes(order,f,count):
    chosen=[];seen=set()
    for i in order:
        scene=f.iloc[i].scene_token
        if scene in seen:continue
        chosen.append(int(i));seen.add(scene)
        if len(chosen)==count:return chosen
    for i in order:
        if int(i) not in chosen:chosen.append(int(i))
        if len(chosen)==count:return chosen
    raise RuntimeError('Insufficient registered case population')
def draw_map(ax,index,location,origin,yaw,bounds,selected,actor_type):
    x0,x1,y0,y1=bounds;corners=ego_to_global(np.array([[x0,y0],[x0,y1],[x1,y0],[x1,y1]]),origin,yaw)
    view=shapely.box(*np.r_[corners.min(0),corners.max(0)]);reg=index.regions[location]
    allowed=[0,1,2,3] if actor_type=='Vehicle' else [2,4,5]
    color={0:'#73a05b',1:'#946ec1',2:'#e6dba8',3:'#c8c8c8',4:'#dd91a3',5:'#9bd7cd'}
    for pi in reg['polygon_tree'].query(view):
        entity=reg['entities'][int(reg['part_entities'][pi])];ty=entity['type']
        if ty not in allowed:continue
        # Display clipping keeps vector figures small; semantic extraction still
        # uses the unchanged complete original components and quotas.
        visible=shapely.intersection(reg['polygon_parts'][pi],view)
        for part in shapely.get_parts(visible):
            if part.geom_type!='Polygon' or part.is_empty:continue
            polygon=shapely.transform(part,lambda coordinates:global_to_ego(coordinates,origin,yaw))
            plot_polygon(polygon,ax=ax,add_points=False,color=color[ty],alpha=.28 if entity['id'] in selected else .14,
                edgecolor=color[ty],linewidth=.9 if entity['id'] in selected else .35)
    if actor_type=='Vehicle':
        for li in reg['lane_tree'].query(view):
            entity=reg['entities'][int(reg['lane_indices'][li])];xy=global_to_ego(entity['coordinates'],origin,yaw)
            ax.plot(xy[:,0],xy[:,1],color=color[entity['type']],lw=1.5 if entity['id'] in selected else .6,alpha=.75 if entity['id'] in selected else .35,zorder=2)
    from matplotlib.lines import Line2D
    return [Line2D([0],[0],color=color[k],lw=3,label=MAP_TYPES[k]) for k in allowed]
def main():
    verify();data=load_semantics();f,fd,ad,top,best,prob,logits,calibrated=load_labels();n=len(f);ii=np.arange(n)
    gap=fd[ii,top[:,4]].astype(np.float64)-fd.min(-1);delta=fd[ii,top[:,4]].astype(np.float64)-fd[ii,top[:,1]]
    picked=[]
    for typ in ['Vehicle','Pedestrian']:
        m=f.agent_type.to_numpy()==typ
        for kind,count,eligible,key in [('Error',10,m&(top[:,4]!=best),-gap),('Success',5,m&(top[:,4]!=top[:,1])&(delta<0),delta)]:
            ids=np.flatnonzero(eligible);order=sorted(ids,key=lambda i:(float(key[i]),f.iloc[i].actor_id))
            picked.extend((typ,kind,j+1,i) for j,i in enumerate(unique_scenes(order,f,count)))
    assert len(picked)==30
    manifest=[]
    for typ,kind,num,i in picked:
        r=f.iloc[i];manifest.append({'Category':typ+kind,'Ordinal':num,'HeadTrainIndex':i,'SceneID':r.scene_token,
            'SampleToken':r.sample_token,'ActorID':r.actor_id,'InstanceToken':r.instance_token,'ActorType':typ,'Fold':int(r.Fold),
            'R2Mode':int(top[i,1]),'CMode':int(top[i,4]),'OracleMode':int(best[i]),'R2FDE':float(fd[i,top[i,1]]),
            'CFDE':float(fd[i,top[i,4]]),'OracleFDE':float(fd[i,best[i]]),'DeltaFDE':float(delta[i]),'OracleGap':float(gap[i]),
            'CMapValid':bool(data['semantic'][i,top[i,4],SEM_FIELDS.index('map_valid_mask')]),
            'Figure':f'stage12a_{typ.lower()}_{kind.lower()}_{num:02d}.png'})
    dump('09_cases/stage12a_case_manifest.csv',manifest)
    index=SparseSemanticIndex();regions=sorted(index.regions);src=f.source_index.to_numpy()
    candidate=np.load(S11A/'01_identity_audit/cache/candidates.npy',mmap_mode='r');GT=np.load(S11A/'01_identity_audit/cache/GT.npy',mmap_mode='r')
    frames=ROOT/'01_map_integrity/cache';origins=np.load(frames/'stage12a_origin.npy',mmap_mode='r');yaws=np.load(frames/'stage12a_yaw.npy',mmap_mode='r')
    files={}
    for record in manifest:
        i=record['HeadTrainIndex'];r=f.iloc[i];prediction=candidate[src[i]];truth=GT[src[i]];current=data['geometry'][i,0,[GEO_FIELDS.index('current_x'),GEO_FIELDS.index('current_y')]]
        allpoints=np.concatenate([prediction.reshape(-1,2),truth,current[None]]);low=allpoints.min(0)-12;high=allpoints.max(0)+12
        bounds=(low[0],high[0],low[1],high[1]);selected=set(int(x) for x in data['entity_ids'][i].ravel() if x>=0)
        location=regions[int(data['map_region'][i])];fig=plt.figure(figsize=(12,9));grid=fig.add_gridspec(3,1,height_ratios=[4,.7,1.35])
        ax=fig.add_subplot(grid[0]);handles=draw_map(ax,index,location,origins[i],float(yaws[i]),bounds,selected,record['ActorType'])
        for k in range(6):
            ax.plot(prediction[k,:,0],prediction[k,:,1],color='#777777',alpha=.65,lw=1,zorder=4,label='All six frozen candidates' if k==0 else None)
            ax.annotate('k'+str(k),prediction[k,-1],xytext=(3,3+(k%3)*8),textcoords='offset points',fontsize=7,zorder=8)
        roles=[('R2 Top1',record['R2Mode'],'#176b91','--',2.2),('Oracle candidate',record['OracleMode'],'#89339a',':',2.8),('C Raw Top1',record['CMode'],'#db6331','-',2.1)]
        for name,k,color,style,lw in roles:ax.plot(prediction[k,:,0],prediction[k,:,1],color=color,ls=style,lw=lw,zorder=5,label=f'{name} (k{k})')
        ax.plot(truth[:,0],truth[:,1],color='black',lw=2.8,zorder=6,label='GT (offline reference)');ax.scatter(*current,c='black',marker='x',s=45,zorder=7,label='Observed t0')
        ordinary,labels=ax.get_legend_handles_labels();legendax=fig.add_subplot(grid[1]);legendax.axis('off')
        legendax.legend(handles+ordinary,[h.get_label() for h in handles]+labels,loc='center',fontsize=7,frameon=False,ncol=3)
        ax.set(xlim=(low[0],high[0]),ylim=(low[1],high[1]),xlabel='t0 ego x (m)',ylabel='t0 ego y (m)');ax.set_aspect('equal',adjustable='box')
        ax.set_title(f'{record["Category"]} {record["Ordinal"]:02d} | Fold {record["Fold"]} | C−R2 ΔFDE={record["DeltaFDE"]:+.3f} m | {location}\nScene ID: {record["SceneID"]}\nActor ID: {record["ActorID"]}',fontsize=8)
        fields=['centerline_mean_distance','centerline_heading_error','drivable_inside_fraction','carpark_inside_fraction'] if record['ActorType']=='Vehicle' else ['crosswalk_distance','crosswalk_intersects','walkway_inside_fraction','drivable_boundary_crossing']
        cell=[]
        for k in range(6):
            flags=' '.join(name for name,mode in [('R2',record['R2Mode']),('C',record['CMode']),('O',record['OracleMode'])] if mode==k)
            values=[f'k{k} {flags}',f'{fd[i,k]:.3f}',f'{prob[i,4,k]:.3f}',f'{calibrated[i,k]:.3f}']
            values.extend(f'{data["semantic"][i,k,SEM_FIELDS.index(s)]:.3f}' if data['feature_valid'][i,k,SEM_FIELDS.index(s)] else 'missing' for s in fields)
            cell.append(values)
        tableax=fig.add_subplot(grid[2]);tableax.axis('off');short=['Lane mean (m)','Heading err (rad)','Drivable inside','Carpark inside'] if record['ActorType']=='Vehicle' else ['Crossing dist (m)','Crossing intersects','Walkway inside','Boundary crosses']
        table=tableax.table(cellText=cell,colLabels=['Mode','FDE (m)','C Raw p','C Cal p',*short],loc='center',cellLoc='center');table.auto_set_font_size(False);table.set_fontsize(7);table.scale(1,1.35)
        fig.suptitle('Stage12A — OOF Development Evidence; GT/oracle are offline references',fontsize=11,y=.995)
        fig.tight_layout(rect=(0,.02,1,.96));stem=Path(record['Figure']).stem
        for ext in ['png','svg','pdf']:
            path=ROOT/'09_cases'/f'{stem}.{ext}';fig.savefig(path,dpi=150,bbox_inches='tight')
            if ext=='svg':path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
            files[path.name]=sha256(path)
        plt.close(fig)
        atomic_json(ROOT/'09_cases'/f'{stem}.json',{'Identity':record,'MapRegion':location,'Origin':origins[i].tolist(),'Yaw':float(yaws[i]),
            'CoordinateFrame':'t0 ego for map, GT and all candidates','ModeSemanticValues':data['semantic'][i].tolist(),
            'FeatureValidMasks':data['feature_valid'][i].tolist(),'SemanticFields':SEM_FIELDS,'SelectedEntityIDs':data['entity_ids'][i].tolist(),
            'CandidateGeometryIDs':data['candidate_geometry_id'][i].astype('U64').tolist(),
            'MapBackground':'original HD Map components within plotting bounds; background context is not inference fallback'})
        print('BEV_CASE_RENDERED',record['Category'],record['Ordinal'],flush=True)
    atomic_json(ROOT/'09_cases/stage12a_case_render_audit.json',{'Status':'PASS','Cases':30,'Counts':{'VehicleError':10,'PedestrianError':10,'VehicleSuccess':5,'PedestrianSuccess':5},
        'CaseSelectionProtocolSHA256':sha256(PROTOCOL),'MapCoverageOrSemanticFeatureSelection':False,
        'GTAndOracleOfflineOnly':True,'SameCoordinateFrame':True,'Files':files})
if __name__=='__main__':main()
