"""Labels become accessible only after all630 semantic scene caches freeze."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'00_manifest'))
from stage12a_common import *
def load_semantics():
    manifest=read_json(ROOT/'02_semantic_cache/stage12a_semantic_cache_manifest.json');assert manifest['Status']=='FROZEN_ALL_630'
    assert manifest['ProtocolSHA256']==sha256(PROTOCOL)
    root=ROOT/'03_feature_statistics/cache';root.mkdir(exist_ok=True);done=root/'stage12a_combined_manifest.json'
    names={'semantic':(260151,6,len(SEM_FIELDS)),'feature_valid':(260151,6,len(SEM_FIELDS)),
        'geometry':(260151,6,len(GEO_FIELDS)),'entity_ids':(260151,6,8),'candidate_geometry_id':(260151,6),'map_region':(260151,)}
    dtypes={'semantic':'float64','feature_valid':'bool','geometry':'float64','entity_ids':'int32','candidate_geometry_id':'S64','map_region':'int8'}
    if not done.exists():
        arrays={k:np.lib.format.open_memmap(root/f'stage12a_{k}.npy',mode='w+',dtype=dtypes[k],shape=v) for k,v in names.items()}
        filled=np.zeros(260151,bool)
        for rec in manifest['Records']:
            path=ROOT/rec['Path'];assert sha256(path)==rec['SHA256']
            with np.load(path) as z:
                ix=z['headtrain_indices'];assert not filled[ix].any()
                for k in arrays:arrays[k][ix]=z[k]
                filled[ix]=True
        assert filled.all()
        for a in arrays.values():a.flush()
        atomic_json(done,{'Status':'PASS','SemanticSourceManifestSHA256':sha256(ROOT/'02_semantic_cache/stage12a_semantic_cache_manifest.json'),
            'files':{p.name:sha256(p) for p in root.glob('*.npy')}})
    else:
        meta=read_json(done);assert meta['Status']=='PASS'
        for name,h in meta['files'].items():assert sha256(root/name)==h
    return {k:np.load(root/f'stage12a_{k}.npy',mmap_mode='r') for k in names}
def load_labels():
    assert read_json(ROOT/'02_semantic_cache/stage12a_semantic_cache_manifest.json')['Status']=='FROZEN_ALL_630'
    f=identities();src=f.source_index.to_numpy();fd=np.array(np.load(TRAIN/'fde.npy',mmap_mode='r')[src]);ad=np.array(np.load(TRAIN/'ade.npy',mmap_mode='r')[src])
    p=np.load(OOF/'stage11b_oof_predictions_probabilities.npy',mmap_mode='r');z=np.load(OOF/'stage11b_oof_predictions_logits.npy',mmap_mode='r')
    old=np.load(OOF/'stage11b_oof_metrics.npy',mmap_mode='r');top=p.argmax(-1);ii=np.arange(len(f))
    for j in range(5):
        assert np.array_equal(fd[ii,top[:,j]],old[:,j,0]) and np.array_equal(ad[ii,top[:,j]],old[:,j,1])
    bike=f.agent_type_id.to_numpy()==2;assert np.array_equal(p[bike,1],p[bike,4]) and np.array_equal(z[bike,1],z[bike,4])
    calibrated=np.empty_like(np.asarray(p[:,4]),dtype=np.float64)
    for name in ['stage11c_calibrated_vehicle_pedestrian.npz','stage11c_bicycle_foldr2_passthrough.npz']:
        with np.load(S11C/'03_oof_evaluation/cache'/name) as c:calibrated[c['headtrain_indices']]=c['probabilities']
    assert np.array_equal(calibrated.argmax(-1),top[:,4])
    return f,fd,ad,top,fd.argmin(-1),p,z,calibrated
def describe(a):
    a=np.asarray(a,np.float64);a=a[np.isfinite(a)]
    if not len(a):return {'Count':0,'Mean':None,'Median':None,'P90':None,'P95':None,'P99':None}
    quant=np.quantile(a,[.5,.9,.95,.99])
    return {'Count':len(a),'Mean':float(a.mean()),'Median':float(quant[0]),'P90':float(quant[1]),'P95':float(quant[2]),'P99':float(quant[3])}
def rho_rows(x,y,valid):
    """Average tied ranks and variable valid-mode counts, vectorized K6."""
    x=np.asarray(x,np.float64);y=np.asarray(y,np.float64);m=np.asarray(valid,bool)
    def ranks(a):
        less=((a[:,None,:]<a[:,:,None])&m[:,None,:]).sum(-1)
        equal=((a[:,None,:]==a[:,:,None])&m[:,None,:]).sum(-1)
        return np.where(m,1+less+.5*(equal-1),0.)
    rx=ranks(x);ry=ranks(y);count=m.sum(-1);center=(count+1)/2
    dx=np.where(m,rx-center[:,None],0);dy=np.where(m,ry-center[:,None],0)
    denominator=np.sqrt((dx*dx).sum(-1)*(dy*dy).sum(-1));eligible=(count>=3)&(denominator>1e-12)
    rho=np.divide((dx*dy).sum(-1),denominator,out=np.zeros(len(x)),where=eligible)
    assert np.isfinite(rho).all()
    return rho,eligible
class SceneStatistics:
    def __init__(self,f):
        audit=read_json(S11B/'06_bootstrap/stage11b_bootstrap_audit.json');self.scenes=audit['SceneOrder']
        assert len(self.scenes)==630
        self.si=f.scene_token.map({s:i for i,s in enumerate(self.scenes)}).to_numpy()
        assert np.array_equal(self.si//210+1,f.Fold.to_numpy())
        self.weights=np.load(S11B/'06_bootstrap/stage11b_scene_bootstrap_weights.npy',mmap_mode='r').astype(np.float64)
        assert self.weights.shape==(2000,630)
    def mean(self,values,mask,bootstrap=False):
        count=np.bincount(self.si[mask],minlength=630);sums=np.bincount(self.si[mask],weights=values[mask],minlength=630)
        has=count>0;means=np.divide(sums,count,out=np.zeros(630),where=has)
        if not has.any():return {'Count':0,'Scenes':0,'SceneMean':None,'SceneMedian':None,'SceneP90':None,'SceneP95':None,'CI95Low':None,'CI95High':None}
        quant=np.quantile(means[has],[.5,.9,.95])
        result={'Count':int(mask.sum()),'Scenes':int(has.sum()),'SceneMean':float(means[has].mean()),
            'SceneMedian':float(quant[0]),'SceneP90':float(quant[1]),'SceneP95':float(quant[2]),'CI95Low':None,'CI95High':None}
        if bootstrap:
            denominator=self.weights@has;assert np.all(denominator>0)
            rep=(self.weights@means)/denominator;low,high=np.percentile(rep,[2.5,97.5]);result.update(CI95Low=float(low),CI95High=float(high))
        return result
