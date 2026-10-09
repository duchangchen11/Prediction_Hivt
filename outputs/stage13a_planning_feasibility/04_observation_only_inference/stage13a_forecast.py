"""Future-free adaptation of the original Stage3 scene graph and frozen forwards."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'00_manifest'))
from stage13a_common import *
from stage3_dataset import VehicleGraph
from preprocessing.extract_lane_polylines import LaneExtractor
from torch_geometric.data import Batch
from stage6a_common import frozen_predictor
from stage6a_features import observable_features,normalize
from stage6a_head import ReliabilityHead
from stage8a_graph import ObservableWindow,node_features,neighbors,interaction_edges
from stage8a0c_model import SparseGraphReranker

def original_graph_normalize():
    p=S11B/'00_manifest/stage11b_common.py';node=next(n for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='graph_normalize');scope={'torch':torch}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(p),'exec'),scope);return scope['graph_normalize']
graph_normalize=original_graph_normalize()
def graph_from_history(obs,extractor):
    # Numerical history/x/rotation/bos/lane definitions exactly match the
    # historical Stage3MultiTypeDataset; all unused future fields are neutral.
    h=torch.from_numpy(obs['history'].astype(np.float32));mask=torch.from_numpy(obs['history_mask']);n=len(h);angles=torch.zeros(n)
    for i in range(n):
        observed=torch.where(mask[i])[0]
        if len(observed)>=2:delta=h[i,observed[-1]]-h[i,observed[-2]];angles[i]=torch.atan2(delta[1],delta[0])
    x=torch.zeros_like(h);consecutive=mask[:,:-1]&mask[:,1:];x[:,1:]=torch.where(consecutive[...,None],h[:,1:]-h[:,:-1],0)
    bos=torch.zeros_like(mask);bos[:,0]=mask[:,0];bos[:,1:]=~mask[:,:-1]&mask[:,1:];row,col=torch.where(~torch.eye(n,dtype=torch.bool))
    lane=extractor.extract(obs['location'],obs['origin'],obs['yaw'],h[:,-1].numpy());l=len(lane['lane_vectors'])
    return VehicleGraph(x=x,positions=torch.cat([h,torch.zeros((n,12,2))],1),num_nodes=n,edge_index=torch.stack([row,col]),
        padding_mask=torch.cat([~mask,torch.ones((n,12),dtype=torch.bool)],1),bos_mask=bos,rotate_angles=angles,
        agent_type=torch.from_numpy(obs['agent_type']),history_mask=mask,lane_vectors=torch.from_numpy(lane['lane_vectors']),lane_positions=torch.from_numpy(lane['lane_positions']),
        lane_actor_index=torch.from_numpy(lane['lane_actor_index']),lane_actor_vectors=torch.from_numpy(lane['lane_actor_vectors']),lane_mask=torch.ones(l,dtype=torch.bool),
        is_intersections=torch.zeros(l,dtype=torch.uint8),turn_directions=torch.zeros(l,dtype=torch.uint8),traffic_controls=torch.zeros(l,dtype=torch.uint8))
class FrozenForecaster:
    def __init__(self):
        seed();verify();self.predictor=frozen_predictor();self.extractor=LaneExtractor(str(MAP_ROOT),50.,2.);self.models={};self.norms={}
        for k in (1,2,3):
            c=SparseGraphReranker('G1',2022+100*(k-1)).cuda();r2=ReliabilityHead().cuda()
            for name,m in [('C',c),('R2',r2)]:
                saved=torch.load(S11B/f'04_checkpoints/fold{k}/{name}_best.pt',map_location='cpu',weights_only=False);assert saved['Fold']==k and saved['Model']==name
                m.load_state_dict(saved['state_dict']);m.eval().requires_grad_(False)
            self.models[k]=(c,r2);self.norms[k]=read_json(S11B/f'02_splits/stage11b_fold{k}_normalization.json')
        self.before=[state_sha(m) for m in self.all_models()]
    def all_models(self):return [self.predictor]+[m for k in (1,2,3) for m in self.models[k]]
    @torch.no_grad()
    def __call__(self,obs,poison_unused_future=False):
        assert 'GT' not in obs and 'future_mask' not in obs and 'ego_future_gt' not in obs
        fold=scene_folds()[obs['scene_token']];n=len(obs['instance_tokens'])
        if not n:return dict(prediction=np.empty((0,6,12,2),np.float32),logits=np.empty((0,6),np.float32),raw_probabilities=np.empty((0,6),np.float32),probabilities=np.empty((0,6),np.float32),valid=obs['prediction_valid_mask'])
        graph=graph_from_history(obs,self.extractor)
        if poison_unused_future:
            graph.positions[:,5:]=float('nan');graph.padding_mask[:,5:]=False
            graph.y=torch.full((n,12,2),float('nan'));graph.future_mask=torch.zeros((n,12),dtype=torch.bool);graph.target_mask=torch.zeros(n,dtype=torch.bool);graph.full_horizon_mask=torch.zeros(n,dtype=torch.bool)
        batch=Batch.from_data_list([graph]).cuda();out=self.predictor(batch);prediction=self.predictor.ego_predictions(out,batch)
        history=torch.from_numpy(obs['history'].astype(np.float32));padding=torch.from_numpy(~obs['history_mask']);typ=torch.from_numpy(obs['agent_type'])
        w=ObservableWindow(history,padding,typ,prediction.cpu(),out['mode_logits'].cpu(),out['mode_prob'].cpu(),obs['scene_token'],obs['sample_token'],obs['instance_tokens'],obs['location'],obs['origin'],obs['yaw'])
        node=node_features(w);idx,keep=neighbors(w);edge=interaction_edges(w,idx,keep);local=torch.cat([node[:,None],node[idx]],1)
        local,edge,keep=graph_normalize(local,edge,keep,self.norms[fold]['graph'])
        feat,flags,_,_=observable_features(history.cuda(),padding.cuda(),typ.cuda(),prediction,out['mode_logits'],out['mode_prob'])
        feat=normalize(feat,flags,self.norms[fold]['R2']);c,r2=self.models[fold];zz=[];raw=[];prob=[]
        for start in range(0,n,128):
            sl=slice(start,min(n,start+128));base=out['mode_logits'][sl];co=c(local[sl].cuda(),edge[sl].cuda(),keep[sl].cuda(),None,None,None,base);ro=r2(feat[sl],base);bike=typ[sl].cuda()==2
            z=co['mode_logits'].clone();p=co['mode_prob'].clone();z[bike]=ro['mode_logits'][bike];p[bike]=ro['mode_prob'][bike]
            scaled=(z/TEMPERATURES[fold]).softmax(-1);scaled[bike]=p[bike]
            assert torch.equal(p[bike],ro['mode_prob'][bike]) and torch.equal(scaled[bike],ro['mode_prob'][bike])
            zz.append(z.cpu());raw.append(p.cpu());prob.append(scaled.cpu())
        result=dict(prediction=prediction.cpu().numpy(),logits=torch.cat(zz).numpy(),raw_probabilities=torch.cat(raw).numpy(),probabilities=torch.cat(prob).numpy(),valid=obs['prediction_valid_mask'].copy())
        assert all(np.isfinite(result[k]).all() for k in ['prediction','logits','raw_probabilities','probabilities'])
        assert np.array_equal(result['raw_probabilities'].argmax(-1),result['probabilities'].argmax(-1))
        return result
    def verify_frozen(self):
        assert self.before==[state_sha(m) for m in self.all_models()] and all(not m.training for m in self.all_models())
        assert all(not p.requires_grad and p.grad is None for m in self.all_models() for p in m.parameters())
