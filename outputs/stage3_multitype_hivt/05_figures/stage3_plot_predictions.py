"""Twelve source-only qualitative plots with strict best/Top1 numerical audits."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'00_manifest'))
from stage3_common import atomic_json,read_json,sha256,update_manifest
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from PIL import Image


def main():
    contract={'conclusion':'Class-wise examples reveal where the No-Type model fits or misses actual futures, distinguishing oracle best-FDE and model Top1.',
              'evidence_chain':'saved GT/history/12-step predictions and mode probabilities; paired ADE/FDE numerical audits',
              'archetype':'asymmetric mixed-modality figure','backend':'Python/matplotlib',
              'exports':['PNG300dpi','editable SVG','TrueType PDF','saved source JSON'],
              'risks':['selected cases do not replace full VAL results','best-FDE uses GT; Top1 uses model probability','no smoothing, interpolation or trajectory edits']}
    atomic_json(ROOT/'00_manifest/stage3_prediction_figure_contract.json',contract)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.fonttype':'none','pdf.fonttype':42,'path.simplify':False,
                         'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False})
    cases=read_json(ROOT/'04_evaluation/stage3_case_manifest.json')['cases'];audits=[]
    for case in cases:
        path=ROOT/case['source_json'];d=read_json(path);source_hash=sha256(path)
        history=np.asarray(d['history_trajectory_m'],dtype=float);hm=np.array(d['history_mask'],bool)
        gt=np.asarray(d['GT_trajectory_m'],dtype=float);modes=np.asarray(d['HiVT_trajectories_m'],dtype=float)
        probability=np.asarray(d['mode_probabilities'],dtype=float);b=int(d['best_FDE_mode_zero_based']);t=int(d['top1_mode_zero_based'])
        assert gt.shape==(12,2) and modes.shape==(6,12,2) and all(d['future_mask'])
        assert b==int(np.linalg.norm(modes[:,-1]-gt[-1],axis=1).argmin()) and t==int(probability.argmax())
        best,top=modes[b],modes[t];be=np.linalg.norm(best-gt,axis=1);te=np.linalg.norm(top-gt,axis=1)
        actual={'minADE6':float(be.mean()),'minFDE6':float(be[-1]),'Top1ADE6':float(te.mean()),'Top1FDE6':float(te[-1])}
        delta={k:abs(v-d['numeric_metrics'][k]) for k,v in actual.items()}
        assert max(delta.values())<1e-4, f"Refuse figure {case['name']}: JSON metric mismatch {delta}"
        visible=np.r_[history[hm],gt,best,top];lo=visible.min(0)-7;hi=visible.max(0)+7
        center=(lo+hi)/2;half=(hi-lo).max()/2
        starts=np.asarray(d['lane_positions_m']).reshape(-1,2);vectors=np.asarray(d['lane_vectors_m']).reshape(-1,2)
        segments=np.stack([starts,starts+vectors],1);near=((segments.max(1)>=lo)&(segments.min(1)<=hi)).all(1);segments=segments[near]
        fig,ax=plt.subplots(figsize=(10,8));fig.subplots_adjust(left=.09,right=.98,bottom=.12,top=.77)

        def plot(target,show_history=True,labels=False):
            target.add_collection(LineCollection(segments,colors='#B9C1C9',linewidths=.5,alpha=.3,zorder=1))
            if show_history:
                h=history.copy();h[~hm]=np.nan
                target.plot(h[:,0],h[:,1],'o-',color='#777D86',lw=1.5,ms=5,label='History' if labels else None,zorder=3)
            target.plot(gt[:,0],gt[:,1],'s-',color='#161616',lw=2.5,ms=7,markerfacecolor='white',markeredgewidth=1.2,label='GT' if labels else None,zorder=10)
            target.plot(top[:,0],top[:,1],'^--',color='#C87B24',lw=2,ms=8,markerfacecolor='white',markeredgewidth=1.1,label=f'HiVT Top-1 (Mode {t+1})' if labels else None,zorder=9)
            target.plot(best[:,0],best[:,1],'o-',color='#246C9E',lw=2.5,ms=5.5,markeredgecolor='white',markeredgewidth=.6,label=f'HiVT best-FDE (Mode {b+1})' if labels else None,zorder=11)
            for xy,color,marker in ((gt[-1],'#161616','s'),(top[-1],'#C87B24','^'),(best[-1],'#246C9E','o')):
                target.plot(*xy,marker=marker,ms=11,color=color,markerfacecolor='white' if marker=='s' else color,
                            markeredgecolor=color if marker=='s' else 'white',markeredgewidth=1.3,zorder=12)
            target.set_aspect('equal',adjustable='box');target.grid(alpha=.1,linewidth=.5)

        plot(ax,labels=True);ax.set_xlim(center[0]-half,center[0]+half);ax.set_ylim(center[1]-half,center[1]+half)
        ax.set_xlabel('t0 ego forward x (m)');ax.set_ylabel('t0 ego left y (m)')
        ax.legend(loc='lower center',bbox_to_anchor=(.5,1.015),ncol=2,fontsize=10)
        white=dict(boxstyle='square,pad=.15',facecolor='white',edgecolor='none',alpha=.9)
        # Text offsets affect labels only; all trajectory/endpoint coordinates remain exact.
        annotations=[]
        for xy,text,color,offset in (
            (gt[-1],'GT endpoint','#161616',(-75,-33)),
            (best[-1],f"Best endpoint\nFDE = {d['numeric_metrics']['minFDE6']:.2f} m",'#246C9E',(12,27)),
            (top[-1],f"Top-1 endpoint\nFDE = {d['numeric_metrics']['Top1FDE6']:.2f} m",'#C87B24',(-100,42))):
            annotations.append(ax.annotate(text,xy=xy,xytext=offset,textcoords='offset points',fontsize=9.5,color=color,zorder=16,bbox=white,
                        arrowprops={'arrowstyle':'-','color':color,'lw':.75,'shrinkB':7}))
        late=np.r_[gt[-6:],best[-6:],top[-6:]];zl,zh=late.min(0)-2.5,late.max(0)+2.5
        # Choose an inset corner with fewest actual trajectory points underneath.
        normalized=(visible-(center-half))/(2*half)
        corner_bounds=[(.03,.59,.42,.34),(.55,.59,.42,.34),(.03,.04,.42,.34),(.55,.04,.42,.34)]
        counts=[int(((normalized[:,0]>=x)&(normalized[:,0]<=x+w)&(normalized[:,1]>=y)&(normalized[:,1]<=y+h)).sum()) for x,y,w,h in corner_bounds]
        inset=ax.inset_axes(corner_bounds[int(np.argmin(counts))],zorder=20)
        plot(inset,show_history=False);inset.set_xlim(zl[0],zh[0]);inset.set_ylim(zl[1],zh[1])
        inset.set_title('Zoom: last 6 future points',fontsize=9,pad=4);inset.tick_params(labelsize=8)
        for spine in inset.spines.values():spine.set_visible(True);spine.set_color('#8D969F')
        ax.indicate_inset_zoom(inset,edgecolor='#77818B',alpha=.5,zorder=2)
        fig.suptitle(f"{d['scene_name']} | sample {d['sample_token'][:8]} | actor {d['instance_token'][:8]} | {d['agent_type']}\n"
                     f"Best ADE/FDE: {actual['minADE6']:.2f}/{actual['minFDE6']:.2f} m    Top-1 ADE/FDE: {actual['Top1ADE6']:.2f}/{actual['Top1FDE6']:.2f} m",
                     fontsize=13,y=.97)
        fig.text(.5,.035,'12 original future observations per trajectory; best-FDE uses GT, Top-1 uses highest saved probability.',ha='center',fontsize=9,color='#5C646D')
        fig.canvas.draw()
        renderer=fig.canvas.get_renderer();canvas=fig.bbox
        for annotation in annotations:
            box=annotation.get_window_extent(renderer)
            shift_x=max(0,12-box.x0)-max(0,box.x1-canvas.x1+12)
            shift_y=max(0,12-box.y0)-max(0,box.y1-canvas.y1+12)
            ox,oy=annotation.xyann
            annotation.xyann=(ox+shift_x*72/fig.dpi,oy+shift_y*72/fig.dpi)
        stem=ROOT/'05_figures'/case['name']
        for ext in ('png','pdf','svg'):fig.savefig(str(stem)+'.'+ext,dpi=300)
        svg=Path(str(stem)+'.svg');svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
        plt.close(fig)
        assert sha256(path)==source_hash
        with Image.open(str(stem)+'.png') as image:image.verify()
        audit={'status':'PASS','source_json':case['source_json'],'source_sha256':source_hash,
               'recomputed_metrics':actual,'metric_absolute_differences_m':delta,'metric_tolerance_m':1e-4,
               'GT_endpoint':gt[-1].tolist(),'best_endpoint':best[-1].tolist(),'top1_endpoint':top[-1].tolist(),
               'best_mode_zero_based':b,'top1_mode_zero_based':t,'future_marker_count_per_trajectory':12,
               'zoom_min_xy_m':zl.tolist(),'zoom_max_xy_m':zh.tolist(),'zoom_rule':'last6 GT/best/Top1 points+2.5m margin',
               'trajectory_coordinates_changed':False,'smoothing':False,'interpolation':False,'other_modes_shown':False,
               'exports':{ext:{'relative_path':str(Path(str(stem)+'.'+ext).relative_to(ROOT)),'sha256':sha256(Path(str(stem)+'.'+ext))} for ext in ('png','pdf','svg')}}
        atomic_json(Path(str(stem)+'_audit.json'),audit);audits.append(audit)
    atomic_json(ROOT/'00_manifest/stage3_prediction_figure_qa.json',{'status':'PASS','case_count':len(audits),'numeric_audits':audits})
    update_manifest();print('PREDICTION_FIGURES=PASS',len(audits),flush=True)


if __name__=='__main__':main()
