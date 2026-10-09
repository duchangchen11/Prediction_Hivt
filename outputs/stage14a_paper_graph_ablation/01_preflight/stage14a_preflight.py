"""Real-data GT poison and unchanged candidate/input/loss integrity checks."""
from pathlib import Path
import sys,copy
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'00_protocol'))
from stage14a_common import *
sys.path.insert(0,str(S8/'00_manifest'))
from stage8a_graph import observable_window,node_features,neighbors,interaction_edges

def raw_graph(w,targets):
    obs=observable_window(w,'',np.zeros(2),0.);node=node_features(obs);idx,keep=neighbors(obs)
    edge=interaction_edges(obs,idx,keep,targets)
    return torch.cat((node[targets,None],node[idx[targets]]),1),edge,keep[targets],w['mode_logits'][targets]

@torch.no_grad()
def main():
    seed();verify(history=True,data=True)
    assert read_json(ROOT/'01_preflight/stage14a_model_integrity.json')['Status']=='PASS'
    historical=read_json(S11B/'01_preflight/stage11b_data_and_poison_audit.json');assert historical['Status']=='PASS' and historical['GTpoisonWindows']==100
    f=frame();allowed=set(split(1)['InnerTrain']);lookup={(r.scene_token,r.sample_token,r.instance_token):i for i,r in enumerate(f.itertuples())}
    store=Store(1);cand=np.load(S11A/'01_identity_audit/cache/candidates.npy',mmap_mode='r');m=fresh(1,'cpu').eval()
    count=0;checked=[]
    for rec in read_json(S6/'01_cache/stage6a_cache_manifest.json')['batches']:
        if rec['split']!='train':continue
        block=torch.load(S6/rec['path'],map_location='cpu',weights_only=False)
        for w in block['windows']:
            if w['scene_token'] not in allowed:continue
            target=[(j,lookup[(w['scene_token'],w['sample_token'],t)]) for j,t in enumerate(w['instance_tokens']) if (w['scene_token'],w['sample_token'],t) in lookup]
            if not target:continue
            targets=torch.tensor([j for j,i in target]);ids=np.array([i for j,i in target]);assert np.isin(ids,indices(1,'InnerTrain')).all()
            original=raw_graph(w,targets);poison=copy.deepcopy(w)
            poison['GT'].fill_(float('nan'));poison['future_mask']=~poison['future_mask'];poison['target_mask']=~poison['target_mask']
            other=raw_graph(poison,targets);assert all(torch.equal(a,b) for a,b in zip(original,other))
            src=store.source[ids]
            assert np.array_equal(cand[src],w['ego_prediction'][targets].numpy())
            for j in range(3):assert np.array_equal(store.args[j][src],original[j].numpy())
            args,fd,ad=store.batch(ids,'cpu',part='InnerTrain');output=m(*args)
            n,e,keep=graph_normalize(*other[:3],store.norm['graph']);again=m(n,e,keep,None,None,None,other[3])
            assert all(torch.equal(output[k],again[k]) for k in output)
            checked.append(dict(SceneToken=w['scene_token'],SampleToken=w['sample_token'],Actors=len(ids),GraphPoisonMaxDiff=0,NoGraphPoisonMaxDiff=0,CandidateCoordinateMaxDiff=0))
            count+=1
            if count==10:break
        if count==10:break
    assert count==10
    initial=[]
    for fold in (1,2,3):
        model=fresh(fold,'cpu');g=SparseGraphReranker('G1',2022+100*(fold-1))
        assert all(torch.equal(v,g.state_dict()[k]) for k,v in model.state_dict().items())
        assert sum(p.numel() for p in model.parameters())==7425
        initial.append(dict(Fold=fold,Seed=2022+100*(fold-1),StateSHA256=state_sha(model),SharedG1Initialization='BITWISE_EQUAL',Params=7425))
    # The forward checks above run without gradients; this label-detachment
    # check needs a live scoring graph to distinguish detached GT labels.
    with torch.enable_grad():
        z=torch.tensor([[.1,.2,.3,.4,.5,.6],[1.,2.,3.,4.,5.,6.]],requires_grad=True)
        errors=torch.tensor([[1.,1.,2.,3.,4.,5.],[2.,3.,4.,5.,6.,7.]],requires_grad=True)
        c=errors.detach()-errors.detach().min(-1,keepdim=True).values;s=c.mean(-1).clamp(min=1.)
        expected={'A':-((-errors.detach()).softmax(-1)*z.log_softmax(-1)).sum(-1),'C':(z.softmax(-1)*(c/s[:,None])).sum(-1)}
        for name in VARIANTS:
            loss=objective(z,errors,name);assert torch.equal(loss,expected[name])
            assert torch.autograd.grad(loss.sum(),errors,allow_unused=True,retain_graph=True)[0] is None
    dump('01_preflight/stage14a_initialization.csv',initial);dump('01_preflight/stage14a_gt_poison_windows.csv',checked)
    atomic_json(ROOT/'01_preflight/stage14a_input_loss_integrity.json',dict(Status='PASS',HeadTrainScenes=630,Folds=3,InnerTrainDevOuterScenes=[378,42,210],
        GTpoisonWindows=10,HistoricalGTpoisonWindowsReused=100,PoisonScope='Fold1 InnerTrain only',CandidateIdentity='PASS',CandidateCoordinateMaxDiff=0,
        OriginalCachedGraphInputs='BITWISE_EQUAL',NoGraphGTpoisonMaxDiff=0,LabelsDetached=True,
        LossCode='exact original Stage11B objective AST, same1m floor and normalization; closed forms byte-equal',
        CurrentLossVariants=['A','C'],ClassWeightsAdded=False,CurrentInputArraysDestroyed=False,
        HeadDevOrVALOrOuterTestUsed=False,FormalTrainingAllowedOnlyAfterTiny=True))
    verify(history=True);print('STAGE14A_REAL_DATA_PREFLIGHT_PASS',flush=True)

if __name__=='__main__':main()
