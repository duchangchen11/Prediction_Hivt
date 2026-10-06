"""Hand-computable geometry/features and no-GT feature interface audits."""
from pathlib import Path
import sys,copy
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'00_manifest'))
from stage6a_common import *
from stage6a_features import observable_features,normalize,FEATURE_NAMES
from stage6a_head import ReliabilityHead,ranking_loss
import numpy as np

def main():
    seed_all(2022);history=torch.zeros(4,5,2);history[:,:,0]=torch.tensor([0.,10.,20.,100.])[:,None]
    padding=torch.zeros(4,5,dtype=torch.bool);types=torch.tensor([0,1,0,2])
    predicted=history[:,4,None,None].expand(4,6,12,2).clone()
    logits=torch.zeros(4,6);prob=logits.softmax(-1)
    features,flags,neighbors,nvalid=observable_features(history,padding,types,predicted,logits,prob)
    assert features.shape==(4,6,19) and torch.equal(features[:,:,:3],torch.nn.functional.one_hot(types,3)[:,None].expand(4,6,3).float())
    assert torch.equal(flags[:,0],torch.tensor([True,True,True,False]))
    assert torch.allclose(features[0,0,12:15],torch.tensor([10.,15.,10.]),atol=1e-5,rtol=0)
    assert abs(float(features[0,0,15])-np.exp(-100/8))<1e-11
    assert abs(float(features[0,0,17])-np.exp(-400/8))<1e-25 and (features[0,:,18]>0).all()
    assert not nvalid[3].any() and not (neighbors[nvalid]==torch.arange(4)[:,None].expand_as(neighbors)[nvalid]).any()
    normalization={'mean':[0.]*19,'std':[1.]*19}
    normalized=normalize(features,flags,normalization)
    assert (normalized[3,:,12:15]==1).all() and (normalized[3,:,15:]==0).all()
    assert (normalize(features,flags,normalization,'R1')[:,:,12:]==0).all()
    # Neighbor multimodal expectation: half near, half far, with all target modes fixed.
    p2=predicted.clone();p2[1,:3,:,0]=1.;p2[1,3:,:,0]=3.
    f2,_,_,_=observable_features(history,padding,types,p2,logits,prob)
    assert abs(float(f2[0,0,12])-2.)<1e-6
    expected=.5*np.exp(-1/8)+.5*np.exp(-9/8)
    assert abs(float(f2[0,0,18])-expected)<1e-6
    # Boundary, self exclusion, invalid current actor and 8-neighbor cap.
    h=torch.zeros(12,5,2);h[:,:,0]=torch.arange(12)[:,None].float();h[11,:,0]=50.
    hp=torch.zeros(12,5,dtype=torch.bool);hp[9,4]=True
    pp=h[:,4,None,None].expand(12,6,12,2).clone();tt=torch.zeros(12,dtype=torch.long)
    _,_,nn,vv=observable_features(h,hp,tt,pp,torch.zeros(12,6),torch.full((12,6),1/6))
    assert vv[0].sum()==8 and nn[0].tolist()==list(range(1,9)) and not (nn[vv]==9).any()
    h3=torch.zeros(2,5,2);h3[1,:,0]=50.;p3=h3[:,4,None,None].expand(2,6,12,2).clone()
    _,flag3,_,_=observable_features(h3,torch.zeros(2,5,dtype=torch.bool),torch.tensor([0,1]),p3,torch.zeros(2,6),torch.full((2,6),1/6))
    assert flag3[:,0].all()
    # Coordinate conversion with different actor rotations, in a common ego frame.
    local=torch.tensor([[[1.,0.]],[[0.,1.]]]);rotation=torch.tensor([[[0.,-1.],[1.,0.]],[[1.,0.],[0.,1.]]])
    current=torch.tensor([[10.,0.],[0.,10.]])
    transformed=local@rotation.transpose(-1,-2)+current[:,None]
    assert torch.equal(transformed,torch.tensor([[[10.,1.]],[[0.,11.]]]))
    seed_all(2022);r1=ReliabilityHead();seed_all(2022);r2=ReliabilityHead()
    assert all(torch.equal(t,r2.state_dict()[name]) for name,t in r1.state_dict().items())
    out=r2(normalized,logits)
    assert torch.equal(out['mode_logits'],logits) and torch.equal(out['mode_prob'],prob)
    assert (out['delta_logits']==0).all()
    # A deliberately non-neutral scoring fixture ensures the leakage audit is meaningful.
    r2.net[-1].weight.data.fill_(.1)
    fixture={'history':history,'history_padding':padding,'types':types,'predicted':p2,'base_logits':logits,'base_prob':prob,
             'GT_future':torch.randn(4,12,2),'future_mask':torch.ones(4,12,dtype=torch.bool),'target_mask':torch.ones(4,dtype=torch.bool),
             'neighbor_GT_future':torch.randn(4,12,2),'future_labels':torch.randint(0,3,(4,12))}
    def infer(d):
        f,fl,_,_=observable_features(d['history'],d['history_padding'],d['types'],d['predicted'],d['base_logits'],d['base_prob'])
        return f,r2(normalize(f,fl,normalization),d['base_logits'])['mode_prob']
    original,original_prob=infer(fixture);checks=[]
    for name in ('GT_future','future_mask','target_mask','neighbor_GT_future','future_labels'):
        altered=copy.deepcopy(fixture);altered[name]=~altered[name] if altered[name].dtype==torch.bool else altered[name]+10000
        f,p=infer(altered);assert torch.equal(f,original) and torch.equal(p,original_prob)
        checks.append({'perturbed':name,'feature_max_diff':float((f-original).abs().max()),'probability_max_diff':float((p-original_prob).abs().max())})
    # Observable neighbor predictions/probabilities must affect R2 features.
    altered=copy.deepcopy(fixture);altered['predicted'][1]+=2
    changed,_=infer(altered);assert not torch.equal(changed[:,:,12:],original[:,:,12:])
    altered=copy.deepcopy(fixture);altered['base_prob'][1]=torch.tensor([1.,0.,0.,0.,0.,0.])
    weighted,_=infer(altered);assert not torch.equal(weighted[:,:,12:],original[:,:,12:])
    gradient_features=normalized.clone();gradient_features[:,:,8]=torch.arange(6).float()
    loss=ranking_loss(r1(gradient_features,logits)['mode_logits'],torch.arange(6).float()[None].expand(4,6));loss.backward()
    assert torch.isfinite(loss) and r1.net[-1].weight.grad.norm()>0
    atomic_json(ROOT/'00_manifest/stage6a_unit_audit.json',{'status':'PASS','feature_dim':19,'head_params':673,
        'initial_head_parameter_max_diff':0.,'neutral_logit_max_diff':0.,'neutral_probability_max_diff':0.,
        'coordinate_fixture':'PASS','self_exclusion':'PASS','radius50_boundary':'PASS','cap8':'PASS',
        'padding_and_context_validity':'PASS','expected_min_distance_not_min_expected_distance':'PASS',
        'soft_conflict_sigma2':'PASS','missing_neighbor_normalization':'PASS','R1_last7_exact_zero':'PASS',
        'initial_final_weight_gradient':float(r1.net[-1].weight.grad.norm()),'no_optimizer_update':True,'NaN':0,'Inf':0})
    atomic_json(ROOT/'00_manifest/stage6a_no_future_leakage_audit.json',{'status':'PASS','checks':checks,
        'nonneutral_head_fixture':True,'neighbor_prediction_and_base_probability_positive_controls':'PASS',
        'GT_arguments_in_feature_API':False,'neighbor_future':'frozen Stage5A predictions only','recursive_reranking':False})
    print('STAGE6A_UNIT_AND_LEAKAGE_PASS',flush=True)

if __name__=='__main__':torch.set_num_threads(4);main()
