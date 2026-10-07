"""Reproducible geometric evidence; all labels are untrained audit candidates."""
from pathlib import Path
import sys,csv,random
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'00_manifest'),str(ROOT/'01_data_audit')]
from stage7a_common import *
from stage7a_map_inventory import load_maps
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as Patch
from matplotlib.lines import Line2D
from shapely.geometry import box
from shapely.strtree import STRtree

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7,'axes.titlesize':7,'pdf.fonttype':42,'svg.fonttype':'none'})
COLORS={'left':'#2878B5','straight':'#707070','right':'#E18727','unknown':'#BA55A2'}
def candidate(d):return 'unknown' if abs(d)>=150 else ('left' if d>20 else 'right' if d<-20 else 'straight')
def table(name):return list(csv.DictReader(open(ROOT/'06_tables'/name)))
def save(fig,name):
    for ext in ('png','pdf','svg'):fig.savefig(ROOT/'05_figures'/(name+'.'+ext),dpi=300,facecolor='white')
    # Keep generated vector XML whitespace clean without changing its geometry.
    svg=ROOT/'05_figures'/(name+'.svg');svg.write_text('\n'.join(s.rstrip() for s in svg.read_text().splitlines())+'\n')
    plt.close(fig)
def bundle():
    out={}
    for path,raw,api in load_maps():
        pol=[api.extract_polygon(r['polygon_token']) for layer in ('lane','lane_connector') for r in raw[layer]]
        out[path.stem]={'raw':raw,'api':api,'polygons':pol,'tree':STRtree(pol)}
    return out
def background(ax,b,padbox):
    ids=b['tree'].query(box(*padbox),predicate='intersects')
    for i in ids:ax.add_patch(Patch(np.asarray(b['polygons'][i].exterior.coords),facecolor='#ECECEC',edgecolor='#CBCBCB',linewidth=0.3))
    ax.set_xlim(padbox[0],padbox[2]);ax.set_ylim(padbox[1],padbox[3]);ax.set_aspect('equal');ax.set_xticks([]);ax.set_yticks([])
    for s in ax.spines.values():s.set_linewidth(0.3);s.set_color('#BDBDBD')
def draw_turn(ax,r,maps,lines):
    loc=r['location'];token=r['token'];p=lines[loc+'__'+token];low=p.min(0)-8;high=p.max(0)+8
    background(ax,maps[loc],(*low,*high));c=COLORS[candidate(float(r['delta_theta_deg']))]
    ax.plot(p[:,0],p[:,1],color=c,lw=1.6);scale=min(8,max(3,np.linalg.norm(high-low)*0.10))
    for endpoint,name,yaw in [(p[0],'IN',float(r['theta_in_rad'])),(p[-1],'OUT',float(r['theta_out_rad']))]:
        v=scale*np.array([np.cos(yaw),np.sin(yaw)]);start=endpoint-v/2;end=endpoint+v/2
        ax.annotate('',xy=end,xytext=start,arrowprops={'arrowstyle':'-|>','color':'#181818','lw':0.8,'mutation_scale':7})
        ax.annotate(name,xy=start,xytext=(2,3),textcoords='offset points',fontsize=5,color='#181818')
    ax.scatter(*p[0],s=9,facecolors='white',edgecolors=c,zorder=6,lw=0.7)
def turn_figures():
    assert read_json(ROOT/'00_manifest/stage7a_figure_contract.json')['backend']=='python'
    maps=bundle();lines=np.load(ROOT/'02_semantic_cache/stage7a_centerlines.npz');rows=table('stage7a_connector_tangent_validation.csv');rng=random.Random(2022);selected=[]
    fig,axes=plt.subplots(1,2,figsize=(7.2,2.7),layout='constrained')
    bins=[-180,-135,-90,-60,-30,-15,15,30,60,90,135,180]
    for ax,col,title in zip(axes,['coarse_2m_delta_deg','delta_theta_deg'],['Original 2 m chord directions','Analytic centerline tangents']):
        ax.hist([float(r[col]) for r in rows],bins=bins,color='#2878B5',edgecolor='white',linewidth=0.5)
        ax.set(xlabel='Entrance-to-exit heading change (degrees)',ylabel='Connector count',title=title);ax.set_xticks([-180,-90,0,90,180])
    fig.suptitle('All four maps; n = 4,589 connectors; no trained model',fontsize=9)
    save(fig,'stage7a_connector_angle_distribution')
    for typ in ('left','straight','right'):
        pool=[r for r in rows if candidate(float(r['delta_theta_deg']))==typ];chosen=rng.sample(pool,min(20,len(pool)))
        fig,axes=plt.subplots(5,4,figsize=(7.2,10.2));fig.subplots_adjust(left=0.025,right=0.99,bottom=0.055,top=0.94,hspace=0.27,wspace=0.10)
        for i,(ax,r) in enumerate(zip(axes.flat,chosen)):
            draw_turn(ax,r,maps,lines);case=f'{typ[0].upper()}{i+1:02d}';short=r['location'].replace('singapore-','SG-').replace('boston-seaport','Boston')
            ax.set_title(f"{case}  {short}\n{float(r['delta_theta_deg']):+.1f} deg  {r['token'][:8]}",fontsize=6,pad=2)
            selected.append({'case_id':case,'candidate_turn':typ,**r,'figure':f'stage7a_turn_manual_{typ}.png'})
        fig.suptitle(f'Random {typ} connector candidates (seed 2022; n = {len(chosen)})',fontsize=10)
        fig.text(0.5,0.02,'Global map frame: +x east, +y north. Hollow circle = entrance; arrows = analytic IN / OUT tangents.',ha='center',fontsize=7)
        save(fig,'stage7a_turn_manual_'+typ)
    for suffix,loc in [('boston','boston-seaport'),('singapore','singapore-onenorth')]:
        chosen=[next(r for r in rows if r['location']==loc and candidate(float(r['delta_theta_deg']))==t and 60<abs(float(r['delta_theta_deg']))<110) if t!='straight' else next(r for r in rows if r['location']==loc and abs(float(r['delta_theta_deg']))<2) for t in ('left','straight','right')]
        fig,axes=plt.subplots(1,3,figsize=(7.2,3.0));fig.subplots_adjust(top=0.65,bottom=0.12,wspace=0.1)
        for ax,r in zip(axes,chosen):draw_turn(ax,r,maps,lines);ax.set_title(f"{candidate(float(r['delta_theta_deg']))} ({float(r['delta_theta_deg']):+.1f} deg)\n{r['token'][:8]}")
        fig.suptitle(loc+' — actual connector geometry',fontsize=10)
        fig.legend([Line2D([0],[0],color=COLORS[k],lw=2) for k in COLORS],list(COLORS),ncol=4,loc='upper center',bbox_to_anchor=(0.5,0.90),frameon=False)
        fig.text(0.5,0.025,'Global +x east / +y north; IN and OUT show travel direction.',ha='center',fontsize=7)
        save(fig,'stage7a_turn_semantics_'+suffix)
    # Show both finer-sampling boundary disagreements, without changing thresholds.
    special=[r for r in rows if abs(float(r['delta_theta_deg']))>=150 or candidate(float(r['delta_theta_deg']))!=candidate(float(r['fine_0p2m_delta_deg'])) or float(r['fine_analytic_difference_deg'])>2]
    if special:
        nr=(len(special)+2)//3;fig,axes=plt.subplots(nr,3,figsize=(7.2,2.6*nr),squeeze=False)
        for ax,r in zip(axes.flat,special):
            draw_turn(ax,r,maps,lines);ax.set_title(f"{r['location']}  {r['token'][:8]}\nanalytic {float(r['delta_theta_deg']):+.2f} / fine {float(r['fine_0p2m_delta_deg']):+.2f} deg",fontsize=6)
        for ax in list(axes.flat)[len(special):]:ax.set_axis_off()
        fig.suptitle('U-turns (unknown) and sampling boundary / short-path cases',fontsize=10);fig.tight_layout(rect=(0,0,1,0.965))
        save(fig,'stage7a_turn_boundary_cases')
    atomic_json(ROOT/'01_data_audit/stage7a_turn_manual_samples.json',{'status':'AWAITING_VISUAL_REVIEW','random_seed':2022,'rows':selected,'extra_boundary_cases':special,
        'classification_is_audit_candidate':True,'candidate_threshold_degrees':20,'u_turn_unknown_absolute_degrees':150,'future_training_authorized':False})
    print('TURN_FIGURES_DONE',len(selected),flush=True)

def semantic_figures():
    samples=read_json(ROOT/'01_data_audit/stage7a_semantic_manual_samples.json');maps=bundle();lines=np.load(ROOT/'02_semantic_cache/stage7a_centerlines.npz')
    for group,rows in samples['groups'].items():
        fig,axes=plt.subplots(5,2,figsize=(7.2,10.5));fig.subplots_adjust(left=0.02,right=0.99,bottom=0.04,top=0.94,hspace=0.40,wspace=0.05)
        for i,(ax,r) in enumerate(zip(axes.flat,rows)):
            loc=r['location'];p=lines[loc+'__'+r['token']];b=maps[loc]
            controls=set(r['semantic_metadata'].get('stop_line_tokens',[]));cross=set(r['semantic_metadata'].get('crosswalk_intersects_tokens',[]))
            objects=[];coords=[p]
            for layer,tokens,color in [('stop_line',controls,'#9557A2'),('ped_crossing',cross,'#E18727')]:
                for source in b['raw'][layer]:
                    if source['token'] in tokens:
                        poly=b['api'].extract_polygon(source['polygon_token']);xy=np.asarray(poly.exterior.coords);coords.append(xy);objects.append((xy,color))
            extent=np.concatenate(coords);low=extent.min(0)-5;high=extent.max(0)+5;background(ax,b,(*low,*high))
            ax.plot(p[:,0],p[:,1],color='#2878B5',lw=1.3);v=p[1]-p[0];v=v/np.linalg.norm(v)*min(5,max(2,np.linalg.norm(high-low)/8))
            ax.annotate('',xy=p[0]+v,xytext=p[0],arrowprops={'arrowstyle':'-|>','lw':1,'color':'#181818','mutation_scale':8})
            for xy,color in objects:ax.add_patch(Patch(xy,facecolor=color,alpha=0.45,edgecolor=color,lw=0.7))
            ax.set_title(f"{group} {i+1:02d}  {loc}\n{r['token'][:8]}  {r['semantic_metadata']['traffic_control_type']}",fontsize=6,pad=2)
        fig.suptitle('Static geometry sample audit: '+group+' (n = '+str(len(rows))+')',fontsize=10)
        fig.text(0.5,0.013,'Blue: centerline. Purple: typed stop-line polygon. Orange: pedestrian crossing. No dynamic signal states.',ha='center',fontsize=6.5)
        save(fig,'stage7a_samples_'+group)
    print('SEMANTIC_FIGURES_DONE',flush=True)
if __name__=='__main__':
    if '--semantic' in sys.argv:semantic_figures()
    else:turn_figures()
