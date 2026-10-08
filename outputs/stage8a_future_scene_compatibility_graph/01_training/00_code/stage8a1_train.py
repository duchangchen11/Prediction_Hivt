"""Three independent fixed runs. Dev selection is soft ranking loss only."""
from stage8a1_common import *
import argparse,copy

@torch.no_grad()
def assess(model,store,ids):
    model.eval();sums=np.zeros(4,dtype=np.float64)
    for start in range(0,len(ids),128):
        args,fde,ade=store.batch(ids[start:start+128]);out=model(*args)
        top=out['mode_logits'].argmax(-1);ii=torch.arange(len(top),device='cuda')
        sums+=np.array([float(loss(out['mode_logits'],fde))*len(top),float(ade[ii,top].double().sum()),
            float(fde[ii,top].double().sum()),float((fde[ii,top]-fde.min(-1).values).double().sum())])
    return dict(zip(('dev_loss','dev_Top1ADE','dev_Top1FDE','dev_OracleGap'),(sums/len(ids)).tolist()))

def gradient(model,predictor):
    grads=[p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    norm=float(torch.stack([g.float().square().sum() for g in grads]).sum().sqrt())
    assert norm>0 and not any(p.grad is not None or p.requires_grad for p in predictor.parameters())
    return {'predictor_gradient_count':0,'graph_gradient_norm':norm,'graph_finite':True,'graph_nonzero':True,'candidate_requires_grad':False}

def tiny(store,predictor,train):
    assert not FROZEN.exists();reg=read_json(ROOT/'00_manifest/stage8a1_registration.json')
    # Fixed seed sampling once shared across variants; all targets HeadTrain.
    ids=np.random.default_rng(2022).choice(train,128,replace=False)
    atomic_json(ROOT/'01_training/stage8a1_tiny_targets.json',{'indices':ids.tolist(),'partition':'HeadTrain','count':128,'seed':2022})
    neutral=np.random.default_rng(2022).choice(train,1024,replace=False);audit=[]
    for variant in PARAMS:
        model=fresh(variant);assert state_sha(model)==reg['initial_state_sha256'][variant]
        maxlog=maxprob=maxinstrument=0.
        for start in range(0,1024,128):
            args,_,_=store.batch(neutral[start:start+128]);
            with torch.no_grad():
                out=model(*args);hooked,_=capture(model,args)
                maxlog=max(maxlog,float((out['mode_logits']-args[6]).abs().max()))
                maxprob=max(maxprob,float((out['mode_prob']-args[6].softmax(-1)).abs().max()))
                maxinstrument=max(maxinstrument,float((out['mode_logits']-hooked['mode_logits']).abs().max()))
        assert maxlog<1e-6 and maxprob<1e-6 and maxinstrument==0.
        audit.append({'Variant':variant,'Targets':1024,'LogitMaxDiff':maxlog,'ProbabilityMaxDiff':maxprob,'CandidateMaxDiff':0.,'CaptureMaxDiff':maxinstrument,'Status':'PASS'})
        args,fde,_=store.batch(ids);assert not any(a.requires_grad for a in args)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
        model.train();initial=float(loss(model(*args)['mode_logits'],fde).detach());curve=[];grad=None
        for step in range(1,301):
            optimizer.zero_grad(set_to_none=True);out=model(*args);l=loss(out['mode_logits'],fde)
            assert torch.isfinite(l);l.backward();grad=gradient(model,predictor);optimizer.step()
            if step%10==0:curve.append({'Update':step,'Loss':float(l.detach())})
        model.eval()
        with torch.no_grad():out=model(*args);final=float(loss(out['mode_logits'],fde));delta=float(out['delta_logits'].abs().max())
        passed=final<.8*initial and delta>0 and bool(torch.isfinite(out['delta_logits']).all())
        root=ROOT/'01_training'/variant
        write_csv(root/'stage8a1_tiny_curve.csv',curve)
        atomic_json(root/'gradient_audit.json',grad)
        atomic_json(root/'stage8a1_tiny_audit.json',{'status':'PASS' if passed else 'FAIL','initial_loss':initial,'final_loss':final,
            'relative_decrease':1-final/initial,'delta_abs_max':delta,'updates':300,'precision':'FP32','predictor_gradient_count':0,
            'candidate_requires_grad':False,'full_training_initialization':'reset to frozen init; tiny weights discarded'})
        print('TINY',variant,initial,final,'PASS' if passed else 'FAIL',flush=True)
        del model,optimizer;torch.cuda.empty_cache()
        assert passed,variant+' tiny FAIL: full training prohibited'
    write_csv(ROOT/'06_tables/stage8a1_neutral_identity.csv',audit)

def train_one(variant,store,predictor,train,dev):
    assert not FROZEN.exists();root=ROOT/'01_training'/variant
    assert read_json(root/'stage8a1_tiny_audit.json')['status']=='PASS'
    regpath=ROOT/'00_manifest/stage8a1_registration.json';reg=read_json(regpath)
    seed();model=fresh(variant);assert state_sha(model)==reg['initial_state_sha256'][variant]
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    config={'variant':variant,'optimizer':'AdamW','lr':.001,'weight_decay':.0001,'seed':2022,'max_epoch':50,'patience':5,
        'selection':'minimum HeadDev ranking loss','microbatch':128,'accumulation':8,'effective_batch':1024,
        'normalization_sha256':sha256(NORM),'initial_state_sha256':state_sha(model),'AMP':True,
        'architecture_source_sha256':sha256(C/'03_model_audit/stage8a0c_model.py'),'registration_sha256':sha256(regpath)}
    atomic_json(root/'training_config.json',config);atomic_json(root/'normalization_manifest.json',read_json(NORM))
    amp=True;micro=128;best=float('inf');bad=0;bestepoch=None;curve=[];pending=np.empty(0,dtype=np.int64);first=1
    scaler=torch.cuda.amp.GradScaler(enabled=amp);elapsed=time.monotonic();fallbacks=[]
    last=root/'last.pt'
    if last.exists():
        saved=torch.load(last,map_location='cpu',weights_only=False);assert saved['registration_sha256']==sha256(regpath)
        if saved['complete']:return read_json(root/'stage8a1_summary.json')
        model.load_state_dict(saved['state_dict']);optimizer.load_state_dict(saved['optimizer_state']);scaler.load_state_dict(saved['scaler_state'])
        best=saved['best_loss'];bad=saved['bad'];bestepoch=saved['best_epoch'];curve=saved['curve'];pending=saved['pending'];first=saved['epoch']+1
        amp=saved['AMP'];micro=saved['microbatch'];fallbacks=saved['fallbacks']
    for epoch in range(first,51):
        order=np.random.default_rng(2022+epoch).permutation(train);order=np.concatenate((pending,order));used=len(order)//1024*1024;pending=order[used:]
        total=0.;seen=0;model.train();epochstart=time.monotonic()
        for pos in range(0,used,1024):
            batch=order[pos:pos+1024]
            # Retry only engineering OOM/AMP instability, same exact batch.
            while True:
                optimizer.zero_grad(set_to_none=True);batchloss=0.
                try:
                    for s in range(0,1024,micro):
                        args,fde,_=store.batch(batch[s:s+micro]);assert not any(a.requires_grad for a in args)
                        with torch.cuda.amp.autocast(enabled=amp):out=model(*args)
                        l=loss(out['mode_logits'].float(),fde);assert torch.isfinite(l),'nonfinite AMP loss'
                        scaler.scale(l*(micro/1024)).backward();batchloss+=float(l.detach())*micro
                    scaler.unscale_(optimizer);grad=gradient(model,predictor)
                    scaler.step(optimizer);scaler.update();break
                except torch.cuda.OutOfMemoryError:
                    if micro==32:raise
                    micro//=2;fallbacks.append({'epoch':epoch,'batch':pos,'reason':'OOM','microbatch':micro,'accumulation':1024//micro})
                    optimizer.zero_grad(set_to_none=True);torch.cuda.empty_cache()
                except AssertionError as error:
                    if not amp or 'predictor' in str(error):raise
                    amp=False;scaler=torch.cuda.amp.GradScaler(enabled=False)
                    fallbacks.append({'epoch':epoch,'batch':pos,'reason':'nonfinite AMP; retry same batch FP32','error':str(error)})
            total+=batchloss;seen+=1024
            if pos%65536==0:print('TRAIN',variant,epoch,pos,'/',used,flush=True)
        metrics=assess(model,store,dev);improved=metrics['dev_loss']<best
        if improved:best=metrics['dev_loss'];bestepoch=epoch;bad=0
        else:bad+=1
        row={'epoch':epoch,'train_loss':total/seen,**metrics,'best_flag':int(improved),'patience_count':bad,
            'optimizer_targets':seen,'carry_targets':len(pending),'seconds':time.monotonic()-epochstart};curve.append(row)
        write_csv(root/'training_curve.csv',curve);write_csv(ROOT/'06_tables'/f'{variant}_training_curve.csv',curve)
        complete=bad>=5 or epoch==50
        checkpoint={'state_dict':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},'variant':variant,'epoch':epoch,
            'best_epoch':bestepoch,'best_loss':best,'normalization_sha256':sha256(NORM),'registration_sha256':sha256(regpath),
            'parameters':PARAMS[variant],'selection':'minimum HeadDev ranking loss'}
        if improved:atomic_torch(root/'best_dev_loss.pt',checkpoint)
        atomic_torch(last,{**checkpoint,'optimizer_state':optimizer.state_dict(),'scaler_state':scaler.state_dict(),'curve':curve,
            'bad':bad,'pending':pending,'complete':complete,'AMP':amp,'microbatch':micro,'fallbacks':fallbacks})
        print('EPOCH',variant,epoch,metrics,'BEST',bestepoch,'PATIENCE',bad,flush=True)
        if complete:break
    selected=torch.load(root/'best_dev_loss.pt',map_location='cpu',weights_only=False);model.load_state_dict(selected['state_dict'])
    reloaded=assess(model,store,dev);assert abs(reloaded['dev_loss']-best)<1e-12
    summary={'status':'COMPLETE','variant':variant,'best_epoch':bestepoch,'executed_epochs':epoch,'selected':reloaded,
        'checkpoint_sha256':sha256(root/'best_dev_loss.pt'),'elapsed_seconds':time.monotonic()-elapsed,
        'checkpoint_selection':'HeadDev ranking loss only','official_VAL_used':False,'predictor_gradient_count':0,
        'candidate_requires_grad':False,'microbatch':micro,'accumulation':1024//micro,'effective_batch':1024,
        'AMP':amp,'fallbacks':fallbacks,'final_carry_discarded_without_optimizer_step':len(pending)}
    atomic_json(root/'stage8a1_summary.json',summary);atomic_json(root/'gradient_audit.json',grad)
    del model,optimizer;torch.cuda.empty_cache();return summary

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tiny-only',action='store_true');opt=parser.parse_args()
    seed();verify();assert not FROZEN.exists(),'All heads already frozen; training prohibited'
    registration=read_json(ROOT/'00_manifest/stage8a1_registration.json')
    for p,h in registration['new_sources_sha256'].items():assert sha256(Path(__file__).parent/p)==h,p
    assert read_json(ROOT/'01_training/stage8a1_future_leakage_audit.json')['status']=='PASS'
    store=Store('train');train=np.flatnonzero(store.partition==1);dev=np.flatnonzero(store.partition==0)
    sys.path.insert(0,str(S6/'00_manifest'));from stage6a_common import frozen_predictor
    predictor=frozen_predictor('cpu');predictorsha=state_sha(predictor)
    if not (ROOT/'06_tables/stage8a1_neutral_identity.csv').exists():tiny(store,predictor,train)
    if opt.tiny_only:return
    summaries=[train_one(v,store,predictor,train,dev) for v in ('G1','G2','G3')]
    assert state_sha(predictor)==predictorsha;verify()
    rows=[{'Variant':v,'Path':str((ROOT/'01_training'/v/'best_dev_loss.pt').relative_to(PROJECT)),
        'SHA256':sha256(ROOT/'01_training'/v/'best_dev_loss.pt'),'BestEpoch':s['best_epoch'],'HeadDevLoss':s['selected']['dev_loss'],
        'Params':PARAMS[v],'Selection':'minimum HeadDev ranking loss'} for v,s in zip(PARAMS,summaries)]
    write_csv(ROOT/'06_tables/stage8a1_checkpoint_manifest.csv',rows)
    atomic_json(FROZEN,{'status':'FROZEN_ALL_COMPLETE','training_prohibited_after_this_file':True,'checkpoints':rows,
        'normalization_sha256':sha256(NORM),'registration_sha256':sha256(ROOT/'00_manifest/stage8a1_registration.json'),
        'graph_spec_sha256':sha256(SPEC),'frozen_predictor_state_sha256':predictorsha,'official_VAL_used':False})
    print('ALL_CHECKPOINTS_FROZEN',rows,flush=True)

if __name__=='__main__':main()
