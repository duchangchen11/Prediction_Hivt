"""Fixed 30-instance illustrations; no case-based model or threshold selection."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'01_identity_audit'))
from stage11a_common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from itertools import combinations

def main():
    verify();f=actor_frame();z,p=scores();fd,ad=fde_ade()
    cand=np.load(CACHE/'candidates.npy',mmap_mode='r');gt=np.load(CACHE/'GT.npy',mmap_mode='r')
    cases=pd.read_csv(ROOT/'08_cases/stage11a_case_selection.csv');assert len(cases)==30 and cases.instance_token.nunique()==30
    scores_rows=[];pair_rows=[];files=[];colors=plt.get_cmap('tab10').colors[:6]
    for r in cases.itertuples():
        i=int(r.source_index);c=np.asarray(cand[i]);y=np.asarray(gt[i]);fde=np.asarray(fd[i]);models=['R2','G1','DualExpert']
        if r.new_model=='G3':models.append('G3')
        fig=plt.figure(figsize=(14,10),layout='constrained');grid=fig.add_gridspec(2,2)
        ax=fig.add_subplot(grid[0,0]);axt=fig.add_subplot(grid[1,0]);axp=fig.add_subplot(grid[0,1]);axd=fig.add_subplot(grid[1,1])
        best=int(fde.argmin());selection={m:int(p[i,MODELS.index(m)].argmax()) for m in models}
        for k in range(6):
            ax.plot(c[k,:,0],c[k,:,1],color=colors[k],lw=2 if k==best else 1.3,
                ls='--' if k==best else '-',label=f'm{k}: FDE={fde[k]:.2f}m'+(' (oracle)' if k==best else ''))
            ax.scatter(*c[k,-1],color=colors[k],s=45,marker='*' if k==best else 'o')
            ax.annotate(f'm{k}',c[k,-1],xytext=(5,5),textcoords='offset points',color=colors[k],fontsize=9)
        ax.plot(y[:,0],y[:,1],'k-',lw=3,label='GT');ax.scatter(*y[-1],c='k',s=70,marker='x')
        ax.set_aspect('equal',adjustable='box');ax.set_xlabel('Ego-frame x (m)');ax.set_ylabel('Ego-frame y (m)')
        ax.grid(alpha=.2)
        entries=[f'{m}: m{selection[m]} ({fde[selection[m]]:.2f}m)' for m in models]
        detail='\n'.join(' | '.join(entries[j:j+2]) for j in range(0,len(entries),2))
        ax.set_title('GT (black), six shared candidates; dashed = oracle\n'+detail,fontsize=10)
        x=np.arange(6);width=.8/len(models)
        for j,m in enumerate(models):
            probs=p[i,MODELS.index(m)];logits=z[i,MODELS.index(m)]
            axp.bar(x+(j-(len(models)-1)/2)*width,probs,width,label=m)
            for k in range(6):
                scores_rows.append({'CaseNumber':r.CaseNumber,'Model':m,'Mode':k,'Logit':float(logits[k]),
                    'Probability':float(probs[k]),'FDE':float(fde[k]),'ADE':float(ad[i,k]),
                    'Selected':k==selection[m],'Oracle':k==best})
        axp.set_xticks(x,[f'm{k}' for k in x]);axp.set_ylim(0,1);axp.set_ylabel('Frozen mode probability')
        axp.set_title('Argmax selection shown in trajectory panel');axp.legend(fontsize=9)
        q=np.exp(-fde.astype(np.float64));q/=q.sum()
        headers=['Mode','FDE(m)','q']+[label for m in models for label in (m+' z',m+' p')]
        table=[]
        for k in range(6):
            table.append([f'm{k}',f'{fde[k]:.3f}',f'{q[k]:.3f}']+
                [value for m in models for value in (f'{z[i,MODELS.index(m),k]:.3f}',f'{p[i,MODELS.index(m),k]:.3f}')])
        axt.axis('off');tab=axt.table(cellText=table,colLabels=headers,loc='center',cellLoc='center')
        tab.auto_set_font_size(False);tab.set_fontsize(7);tab.scale(1,2)
        for k in range(6):tab[k+1,0].set_facecolor(tuple(.7+.3*np.array(colors[k])))
        axt.set_title('Frozen logits z, probabilities p and diagnostic target q',fontsize=10)
        e=c[:,-1].astype(np.float64);dist=np.linalg.norm(e[:,None]-e[None],axis=-1)
        axd.imshow(dist,cmap='Blues',vmin=0,vmax=max(.01,dist.max()))
        for a in range(6):
            for b in range(6):axd.text(b,a,f'{dist[a,b]:.2f}',ha='center',va='center',fontsize=9,color='white' if dist[a,b]>.6*dist.max() else 'black')
        axd.set_xticks(x,[f'm{k}' for k in x]);axd.set_yticks(x,[f'm{k}' for k in x]);axd.set_title('Candidate endpoint distances (m)')
        for a,b in combinations(range(6),2):
            pair_rows.append({'CaseNumber':r.CaseNumber,'ModeA':a,'ModeB':b,'EndpointDistance_m':float(dist[a,b]),
                'TrajectoryRMS_m':float(np.sqrt(np.mean(np.sum((c[a].astype(np.float64)-c[b])**2,-1))))})
        fig.suptitle(f'Case {r.CaseNumber:02d} | {r.CaseCategory} | {r.Comparison}\n'
            f'delta Top1FDE={r.delta_FDE:+.3f}m | GT displacement={r.GT_displacement:.2f}m | recent speed={r.recent_speed:.2f}m/s\n'
            f'scene {r.scene_id} | sample {r.sample_token} | instance {r.instance_token}',fontsize=10)
        path=ROOT/f'08_cases/stage11a_case_{r.CaseNumber:02d}.png';fig.savefig(path,dpi=130);plt.close(fig);files.append(path)
    pd.DataFrame(scores_rows).to_csv(ROOT/'08_cases/stage11a_case_mode_scores.csv',index=False,float_format='%.12g')
    pd.DataFrame(pair_rows).to_csv(ROOT/'08_cases/stage11a_case_endpoint_pairs.csv',index=False,float_format='%.12g')
    # Six contact sheets allow checking every panel without changing the fixed selection.
    for page in range(5):
        fig,axes=plt.subplots(3,2,figsize=(16,17),layout='constrained')
        for ax,path in zip(axes.flat,files[page*6:(page+1)*6]):
            ax.imshow(plt.imread(path));ax.set_axis_off()
        fig.savefig(ROOT/f'08_cases/stage11a_contact_sheet_{page+1}.png',dpi=90);plt.close(fig)
    atomic_json(ROOT/'08_cases/stage11a_case_audit.json',{'Status':'PASS','Count':30,'DistinctInstances':30,
        'counts':cases.CaseCategory.value_counts().to_dict(),'selection':'registered union of3 comparisons; signed-delta sort; unique instances',
        'case_driven_parameter_selection':False,'scores_rows':len(scores_rows),'endpoint_pairs_rows':len(pair_rows),
        'same_GT_candidate_axis':True,'figures':{p.name:sha256(p) for p in files}})
    verify();print('STAGE11_CASE_RENDER_PASS',flush=True)

if __name__=='__main__':main()
