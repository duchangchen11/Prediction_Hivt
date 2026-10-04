"""Source-only coordinate QA figures; the saved JSON is the sole plot input."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'00_manifest'))
from stage3_common import CLASSES,atomic_json,read_json,sha256,update_manifest
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def main():
    contract={'conclusion':'Ten randomly sampled actor-windows per class preserve instance identity and align histories/futures with the t0 ego map frame.',
              'evidence_chain':'three class panels use saved annotations, masks, real coordinates and source map centerlines; no model predictions',
              'archetype':'quantitative grid','backend':'Python/matplotlib','exports':['PNG300dpi','editable SVG','TrueType PDF'],
              'risks':['partial future masks must be respected','one coordinate QA sample is not a prediction result']}
    atomic_json(ROOT/'00_manifest/stage3_data_figure_contract.json',contract)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'svg.fonttype':'none','path.simplify':False,'axes.spines.top':False,'axes.spines.right':False})
    for name in CLASSES:
        source=ROOT/f'01_data_audit/stage3_data_{name}_examples.json';saved=read_json(source)
        assert len(saved['cases'])==10
        fig,axes=plt.subplots(2,5,figsize=(17,7.5),layout='constrained')
        for ax,c in zip(axes.flat,saved['cases']):
            assert c['audit']['status']=='PASS'
            history=np.array(c['history_trajectory_m']);gt=np.array(c['GT_trajectory_m'])
            hm=np.array(c['history_mask'],bool);fm=np.array(c['future_mask'],bool)
            starts=np.array(c['lane_positions_m']).reshape(-1,2);vectors=np.array(c['lane_vectors_m']).reshape(-1,2)
            ax.add_collection(LineCollection(np.stack([starts,starts+vectors],1),colors='#BAC1C8',lw=.55,alpha=.55,zorder=1))
            hplot=history.copy();hplot[~hm]=np.nan;fplot=gt.copy();fplot[~fm]=np.nan
            ax.plot(hplot[:,0],hplot[:,1],'o-',color='#777D86',lw=1.5,ms=3,label='History',zorder=3)
            ax.plot(fplot[:,0],fplot[:,1],'s-',color='#151515',lw=2,ms=3,label='GT Future',zorder=4)
            valid=np.r_[history[hm],gt[fm]];lo=valid.min(0)-8;hi=valid.max(0)+8
            center=(lo+hi)/2;half=(hi-lo).max()/2
            ax.set_xlim(center[0]-half,center[0]+half);ax.set_ylim(center[1]-half,center[1]+half)
            ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.1,lw=.4)
            ax.set_title(f"{c['scene_name']} | {c['instance_token'][:8]}\n{name} | history {hm.sum()}/5, future {fm.sum()}/12",fontsize=8)
            ax.set_xlabel('ego forward x (m)',fontsize=8);ax.set_ylabel('ego left y (m)',fontsize=8)
            ax.tick_params(labelsize=7)
        fig.suptitle(f'{name.title()}: 10 source-verified actor-windows | History + GT + HD Map | no predictions',fontsize=13)
        axes.flat[0].legend(fontsize=7,loc='best')
        stem=ROOT/f'05_figures/stage3_data_{name}_examples'
        for ext in ('png','pdf','svg'):fig.savefig(str(stem)+'.'+ext,dpi=300)
        plt.close(fig)
        atomic_json(Path(str(stem)+'_audit.json'),{'status':'PASS','source_json':str(source.relative_to(ROOT)),
                    'source_sha256':sha256(source),'actor_windows':10,'all_masks_respected':True,'predictions_plotted':False,
                    'trajectory_smoothing':False,'trajectory_interpolation':False,'source_coordinates_changed':False})
    update_manifest();print('DATA_FIGURES=PASS',flush=True)


if __name__=='__main__':main()
