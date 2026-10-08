"""Fixed independent expert training; pure same-type loss and carried effective batches."""
from stage10a_common import *
import argparse

def gradient(model, predictor, r2):
    grads=[p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    norm=float(torch.stack([g.square().sum() for g in grads]).sum().sqrt()); assert norm>0
    for m in (predictor,r2): assert not m.training and not any(p.requires_grad or p.grad is not None for p in m.parameters())
    modules={}
    for name in ('node_encoder','interaction_encoder','interaction_attention','norm','head'):
        gg=[p.grad for p in getattr(model,name).parameters() if p.grad is not None]
        modules[name]=float(torch.stack([g.square().sum() for g in gg]).sum().sqrt()) if gg else 0.
    return {'status':'PASS','graph_gradient_norm':norm,'module_gradient_norms':modules,
        'gradients_finite_nonzero':True,'predictor_gradient_count':0,'R2_gradient_count':0,'candidate_requires_grad':False}

@torch.no_grad()
def assess(model, store, ids):
    model.eval(); sums=np.zeros(3)
    for start in range(0,len(ids),128):
        ix=ids[start:start+128]; args,fde,ade=store.batch(ix); out=model(*args)
        loss=per_actor_loss(out['mode_logits'],fde); top=out['mode_prob'].argmax(-1)
        ii=torch.arange(len(ix),device='cuda')
        sums+=np.array([float(loss.double().sum()),float(ade[ii,top].double().sum()),float(fde[ii,top].double().sum())])
    metrics=dict(zip(('DevSoftCE','DevTop1ADE','DevTop1FDE'),(sums/len(ids)).tolist()))
    metrics['DevCount']=len(ids); assert all(np.isfinite(x) for x in metrics.values()); return metrics

def tiny(store, predictor, r2):
    assert not FROZEN.exists(); reg=read_json(REG); audits=[]; choices={}
    for t,e in enumerate(EXPERTS):
        pool=np.flatnonzero((store.partition==1)&(store.types==t))
        ids=np.random.default_rng(SEEDS[e]).choice(pool,128,replace=False); choices[e]=ids.tolist()
        path=ROOT/'01_training'/e/'stage10a_tiny_audit.json'
        if path.exists():
            saved=read_json(path); assert saved['status']=='PASS', 'Failed tiny: STOP; no repeat attempt'
            audits.append(saved); continue
        seed(SEEDS[e]); model=fresh(e); assert state_sha(model)==reg['initial_state_sha256'][e]
        args,fde,_=store.batch(ids); assert not any(a.requires_grad for a in args)
        q=(-fde).softmax(-1); entropy=float(-(q*q.clamp_min(1e-30).log()).sum(-1).mean())
        initial=float(per_actor_loss(model(*args)['mode_logits'],fde).mean().detach()); assert initial>entropy
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001); curve=[]
        for step in range(1,301):
            optimizer.zero_grad(set_to_none=True); out=model(*args)
            loss=per_actor_loss(out['mode_logits'],fde).mean(); assert torch.isfinite(loss)
            assert all(torch.isfinite(x).all() for x in out.values())
            loss.backward(); grad=gradient(model,predictor,r2); optimizer.step()
            if step%10==0: curve.append({'Update':step,'SoftCE':float(loss.detach())})
        model.eval()
        with torch.no_grad():
            out=model(*args); final=float(per_actor_loss(out['mode_logits'],fde).mean())
        reduction=1-(final-entropy)/(initial-entropy)
        passed=reduction>=.9 and all(torch.isfinite(x).all() for x in out.values()) and all(v>0 for v in grad['module_gradient_norms'].values())
        row={'Expert':e,'status':'PASS' if passed else 'FAIL','InitialLoss':initial,'FinalLoss':final,
            'EntropyFloor':entropy,'InitialExcessLoss':initial-entropy,'FinalExcessLoss':final-entropy,
            'ExcessLossReduction':reduction,'Updates':300,'fixed_same_type_HeadTrain_targets':128,
            'seed':SEEDS[e],'tiny_weights_reused':False,**{k:v for k,v in grad.items() if k!='status'}}
        atomic_json(path,row); write_csv(ROOT/'01_training'/e/'stage10a_tiny_curve.csv',curve); audits.append(row)
        atomic_json(ROOT/'01_training/stage10a_tiny_targets.json',{'partition':'HeadTrain','choices':choices,'seeds':SEEDS})
        atomic_json(ROOT/'01_training/stage10a_tiny_gate.json',{'status':'PASS' if len(audits)==2 and passed else 'INCOMPLETE' if passed else 'FAIL',
            'formal_training_allowed':len(audits)==2 and all(a['status']=='PASS' for a in audits),'audits':audits})
        print('TINY',e,row,flush=True); del model,optimizer; torch.cuda.empty_cache()
        assert passed, e+' Tiny FAIL: STOP, no tuning'

def train_one(e, store, predictor, r2, train, dev):
    assert not FROZEN.exists(); root=ROOT/'01_training'/e/'formal'; root.mkdir(exist_ok=True)
    done=root/'stage10a_training_summary.json'
    if done.exists():
        s=read_json(done); assert s['status']=='COMPLETE' and sha256(ROOT/'02_checkpoints'/CPNAMES[e])==s['checkpoint_sha256']; return s
    assert read_json(ROOT/'01_training/stage10a_tiny_gate.json')['formal_training_allowed']
    reg=read_json(REG); seed(SEEDS[e]); model=fresh(e); assert state_sha(model)==reg['initial_state_sha256'][e]
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    best=float('inf'); bestepoch=None; bad=0; curve=[]; pending=np.empty(0,dtype=np.int64); first=1; micro=128
    coverage=np.zeros(len(store.types),dtype=np.int64); last=root/'last.pt'
    config={**read_json(PROTOCOL)['training'],'expert':e,'target_type':CLASSES[EXPERTS.index(e)],'seed':SEEDS[e],
        'initial_state_sha256':state_sha(model),'registration_sha256':sha256(REG),'normalization_sha256':sha256(NORM),
        'HeadTrainTargets':len(train),'HeadDevTargets':len(dev),'optimizer_parameter_ptrs_independent':True}
    atomic_json(root/'training_config.json',config)
    atomic_json(root/'normalization_manifest.json',{'source':str(NORM.relative_to(PROJECT)),'sha256':sha256(NORM),'refit':False})
    if last.exists():
        s=torch.load(last,map_location='cpu',weights_only=False); assert s['registration_sha256']==sha256(REG)
        model.load_state_dict(s['state_dict']); opt.load_state_dict(s['optimizer_state'])
        best=s['best_score']; bestepoch=s['best_epoch']; bad=s['bad']; curve=s['curve']; pending=s['pending']
        coverage=s['coverage']; first=s['epoch']+1; micro=s['microbatch']
    elapsed=time.monotonic(); t=EXPERTS.index(e)
    for epoch in range(first,51):
        order=np.concatenate((pending,np.random.default_rng(SEEDS[e]+epoch).permutation(train)))
        used=len(order)//1024*1024; pending=order[used:].copy(); assert used+len(pending)==len(order)
        total=0.; batches=0; model.train(); beg=time.monotonic()
        for pos in range(0,used,1024):
            batch=order[pos:pos+1024]; assert np.all(store.types[batch]==t) and np.all(store.partition[batch]==1)
            while True:
                opt.zero_grad(set_to_none=True); batchloss=0.
                try:
                    for start in range(0,1024,micro):
                        args,fde,_=store.batch(batch[start:start+micro]); assert not any(a.requires_grad for a in args)
                        out=model(*args); per=per_actor_loss(out['mode_logits'],fde)
                        loss=per.sum()/1024.; assert torch.isfinite(loss); loss.backward()
                        batchloss+=float(per.detach().double().sum())/1024.
                    grad=gradient(model,predictor,r2); opt.step(); break
                except torch.cuda.OutOfMemoryError:
                    if micro==32: raise
                    micro//=2; opt.zero_grad(set_to_none=True); torch.cuda.empty_cache()
                    print('OOM_SAME_EFFECTIVE_BATCH',e,micro,flush=True)
            np.add.at(coverage,batch,1); total+=batchloss; batches+=1
            if pos%65536==0: print('TRAIN',e,epoch,pos,'/',used,flush=True)
        metrics=assess(model,store,dev); score=metrics['DevSoftCE']; improved=score<best
        if improved: best=score; bestepoch=epoch; bad=0
        else: bad+=1
        row={'epoch':epoch,'TrainSoftCE':total/batches,**metrics,'best_flag':int(improved),'patience_count':bad,
            'optimizer_targets':used,'carry_targets':len(pending),'seconds':time.monotonic()-beg}
        curve.append(row); write_csv(root/'stage10a_training_curve.csv',curve)
        write_csv(ROOT/'06_tables'/f'stage10a_{e}_training_curve.csv',curve)
        complete=bad>=5 or epoch==50
        cp={'state_dict':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},'expert':e,'seed':SEEDS[e],
            'epoch':epoch,'best_epoch':bestepoch,'best_score':best,'registration_sha256':sha256(REG),
            'selection':'minimum own-type complete HeadDev SoftCE','parameters':PARAMS,'normalization_sha256':sha256(NORM)}
        if improved: atomic_torch(ROOT/'02_checkpoints'/CPNAMES[e],cp)
        atomic_torch(last,{**cp,'optimizer_state':opt.state_dict(),'bad':bad,'curve':curve,'pending':pending,
            'microbatch':micro,'coverage':coverage,'complete':complete})
        print('EPOCH',e,epoch,metrics,'BEST',bestepoch,'PATIENCE',bad,flush=True)
        if complete: break
    cp=torch.load(ROOT/'02_checkpoints'/CPNAMES[e],map_location='cpu',weights_only=False)
    model.load_state_dict(cp['state_dict']); selected=assess(model,store,dev)
    assert selected['DevSoftCE']==best and np.all(coverage[train]>0)
    assert not coverage[store.types!=t].any() and not coverage[store.partition==0].any()
    summary={'status':'COMPLETE','Expert':e,'selected_epoch':bestepoch,'executed_epochs':epoch,
        'HeadDevSoftCE':best,'selected':selected,'checkpoint_sha256':sha256(ROOT/'02_checkpoints'/CPNAMES[e]),
        'parameters':PARAMS,'precision':'FP32','microbatch':micro,'effective_batch':1024,'accumulation':1024//micro,
        'elapsed_seconds':time.monotonic()-elapsed,'official_VAL_used':False,'predictor_gradient_count':0,'R2_gradient_count':0,
        'candidate_requires_grad':False,'tiny_weights_reused':False,'all_own_type_HeadTrain_targets_optimized':True,
        'other_type_loss_targets':0,'HeadDev_loss_backprop_targets':0,'unique_train_targets':len(train),
        'final_unoptimized_carry_occurrences_retained_in_last_checkpoint':len(pending)}
    atomic_json(done,summary); atomic_json(root/'gradient_audit.json',grad)
    del model,opt; torch.cuda.empty_cache(); return summary

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--tiny-only',action='store_true'); args=ap.parse_args()
    seed(); verify(); assert not FROZEN.exists(), 'Frozen checkpoints: training prohibited'
    assert read_json(ROOT/'01_training/stage10a_engineering_audit.json')['status']=='PASS'
    store=Store(); predictor=frozen_predictor('cpu'); r2=frozen_r2('cpu'); states=[state_sha(m) for m in (predictor,r2)]
    tiny(store,predictor,r2)
    if args.tiny_only: return
    summaries=[]
    for t,e in enumerate(EXPERTS):
        train=np.flatnonzero((store.partition==1)&(store.types==t)); dev=np.flatnonzero((store.partition==0)&(store.types==t))
        summaries.append(train_one(e,store,predictor,r2,train,dev))
    assert states==[state_sha(m) for m in (predictor,r2)]; verify()
    rows=[{'Expert':e,'Path':str((ROOT/'02_checkpoints'/CPNAMES[e]).relative_to(PROJECT)),
        'SHA256':s['checkpoint_sha256'],'SelectedEpoch':s['selected_epoch'],'HeadDevSoftCE':s['HeadDevSoftCE'],
        'Params':PARAMS,'Seed':SEEDS[e],'Selection':'minimum own-type HeadDev SoftCE'} for e,s in zip(EXPERTS,summaries)]
    write_csv(ROOT/'06_tables/stage10a_checkpoint_manifest.csv',rows)
    write_csv(ROOT/'06_tables/stage10a_training_summary.csv',[{k:s[k] for k in (
        'Expert','selected_epoch','executed_epochs','HeadDevSoftCE','parameters','precision','microbatch','effective_batch',
        'accumulation','elapsed_seconds','unique_train_targets','final_unoptimized_carry_occurrences_retained_in_last_checkpoint')} for s in summaries])
    atomic_json(FROZEN,{'status':'FROZEN_ALL_COMPLETE','training_prohibited':True,'checkpoints':rows,
        'registration_sha256':sha256(REG),'normalization_sha256':sha256(NORM),'predictor_R2_state_sha256':states,
        'official_VAL_used':False,'test_used':False,'checkpoint_selection':'own-type HeadDev SoftCE only'})
    print('STAGE10_ALL_CHECKPOINTS_FROZEN',rows,flush=True)

if __name__=='__main__': main()
