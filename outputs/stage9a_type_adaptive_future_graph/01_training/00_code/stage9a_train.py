"""Fixed FP32 Stage9 tiny and T1->T2->T3 training; exact effective-batch macro."""
from stage9a_common import *
import argparse

def gradient(model,predictor,r2):
    g=[p.grad for p in model.parameters() if p.grad is not None]
    assert g and all(torch.isfinite(x).all() for x in g),'nonfinite graph gradient'
    norm=float(torch.stack([x.square().sum() for x in g]).sum().sqrt());assert norm>0
    for frozen in (predictor,r2):assert not any(p.requires_grad or p.grad is not None for p in frozen.parameters())
    return {'status':'PASS','graph_gradient_norm':norm,'graph_gradient_finite_nonzero':True,'predictor_gradient_count':0,'R2_gradient_count':0,'candidate_requires_grad':False}

@torch.no_grad()
def assess(model,store,ids):
    model.eval();loss_sums=np.zeros(3);fde_sums=np.zeros(3);counts=np.zeros(3,dtype=np.int64)
    for start in range(0,len(ids),128):
        ix=ids[start:start+128];args,fde,_=store.batch(ix);out=model(*args)
        values=per_actor_loss(out['mode_logits'],fde);top=out['mode_logits'].argmax(-1)
        selected=fde[torch.arange(len(ix),device='cuda'),top];ty=store.types[ix]
        for t in range(3):
            mask=torch.as_tensor(ty==t,device='cuda');counts[t]+=int(mask.sum())
            loss_sums[t]+=float(values[mask].double().sum());fde_sums[t]+=float(selected[mask].double().sum())
    losses=np.divide(loss_sums,counts,out=np.zeros(3),where=counts>0);macro=float(losses[counts>0].mean())
    overall=float(loss_sums.sum()/counts.sum());result={'OverallDevSoftCE':overall,'MacroTypeDevSoftCE':macro,'DevSelectionScore':.5*(overall+macro),'OverallDevTop1FDE':float(fde_sums.sum()/counts.sum())}
    for t,c in enumerate(CLASSES):result[c+'DevLoss']=float(losses[t]);result[c+'DevTop1FDE']=float(fde_sums[t]/counts[t]) if counts[t] else 0.;result[c+'DevCount']=int(counts[t])
    assert all(np.isfinite(v) for v in result.values());return result

def tiny(store,predictor,r2,train):
    assert not FROZEN.exists();reg=read_json(REG)
    ids=np.random.default_rng(2022).choice(train,128,replace=False);neutral=np.random.default_rng(2022).choice(train,1024,replace=False)
    atomic_json(ROOT/'01_training/stage9a_tiny_targets.json',{'indices':ids.tolist(),'seed':2022,'count':128,'partition':'HeadTrain','types':np.bincount(store.types[ids],minlength=3).tolist()})
    audits=[];neutrals=[]
    for v in VARIANTS:
        p=ROOT/'01_training'/v/'stage9a_tiny_audit.json'
        if p.exists():
            saved=read_json(p);assert saved['status']=='PASS','Tiny already failed; no new attempt authorized'
            audits.append(saved);neutrals.append(read_json(ROOT/'01_training'/v/'stage9a_neutral_audit.json'));continue
        seed();m=fresh(v);assert state_sha(m)==reg['initial_state_sha256'][v]
        maxlog=maxprob=0.;gatelo=1.;gatehi=0.
        with torch.no_grad():
            for start in range(0,1024,128):
                args,_,_=store.batch(neutral[start:start+128]);out=m(*args)
                maxlog=max(maxlog,float((out['mode_logits']-args[3]).abs().max()));maxprob=max(maxprob,float((out['mode_prob']-args[3].softmax(-1)).abs().max()))
                gatelo=min(gatelo,float(out['gate'].min()));gatehi=max(gatehi,float(out['gate'].max()))
                assert torch.isfinite(out['gate']).all() and 0<=gatelo<=gatehi<=1
        assert maxlog<1e-6 and maxprob<1e-6
        neutralrow={'Variant':v,'Targets':1024,'LogitMaxDiff':maxlog,'ProbabilityMaxDiff':maxprob,'CandidateMaxDiff':0.,'GateMin':gatelo,'GateMax':gatehi,'Status':'PASS'}
        atomic_json(ROOT/'01_training'/v/'stage9a_neutral_audit.json',neutralrow);neutrals.append(neutralrow)
        args,fde,_=store.batch(ids);ty=torch.as_tensor(store.types[ids],device='cuda');assert not any(a.requires_grad for a in args)
        balanced=v=='T3';q=(-fde).softmax(-1);ent=-(q*q.clamp_min(1e-30).log()).sum(-1)
        entropy=float(objective(ent,ty,balanced)[0]);initial=float(objective(per_actor_loss(m(*args)['mode_logits'],fde),ty,balanced)[0].detach())
        assert initial>entropy
        opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001);curve=[];macro_grad=None
        for step in range(1,301):
            opt.zero_grad(set_to_none=True);out=m(*args);value,micro,macro=objective(per_actor_loss(out['mode_logits'],fde),ty,balanced)
            assert all(torch.isfinite(x).all() for x in out.values()) and torch.isfinite(value)
            if balanced and step==2:
                gg=torch.autograd.grad(.5*macro,tuple(m.parameters()),retain_graph=True,allow_unused=True)
                valid=[g for g in gg if g is not None];macro_grad=float(torch.stack([g.square().sum() for g in valid]).sum().sqrt())
                assert macro_grad>0 and all(torch.isfinite(g).all() for g in valid)
            value.backward();grad=gradient(m,predictor,r2);opt.step()
            if step%10==0:curve.append({'Update':step,'Loss':float(value.detach()),'MicroLoss':float(micro.detach()),'MacroLoss':float(macro.detach())})
        m.eval()
        with torch.no_grad():out=m(*args);final=float(objective(per_actor_loss(out['mode_logits'],fde),ty,balanced)[0])
        reduction=1-(final-entropy)/(initial-entropy);delta=float(out['delta_logits'].abs().max())
        passed=reduction>=.9 and delta>0 and all(torch.isfinite(x).all() for x in out.values())
        row={'Variant':v,'status':'PASS' if passed else 'FAIL','InitialLoss':initial,'FinalLoss':final,'EntropyFloor':entropy,'InitialExcessLoss':initial-entropy,'FinalExcessLoss':final-entropy,'ExcessLossReduction':reduction,'Updates':300,'DeltaAbsMax':delta,'macro_half_gradient_norm':macro_grad,'tiny_weights_reused':False,**{k:val for k,val in grad.items() if k!='status'}}
        atomic_json(p,row);write_csv(ROOT/'01_training'/v/'stage9a_tiny_curve.csv',curve);audits.append(row)
        write_csv(ROOT/'06_tables/stage9a_neutral_identity.csv',neutrals)
        atomic_json(ROOT/'01_training/stage9a_tiny_gate.json',{'status':'PASS' if len(audits)==3 and all(a['status']=='PASS' for a in audits) else 'INCOMPLETE' if passed else 'FAIL','results':audits,'formal_training_allowed':len(audits)==3 and all(a['status']=='PASS' for a in audits)})
        print('TINY',v,row,flush=True);del m,opt;torch.cuda.empty_cache()
        assert passed,v+' TINY FAIL: STOP; no tuning/full training'

def train_one(v,store,predictor,r2,train,dev):
    assert not FROZEN.exists();root=ROOT/'01_training'/v/'formal';root.mkdir(exist_ok=True)
    done=root/'stage9a_training_summary.json'
    if done.exists():
        s=read_json(done);assert s['status']=='COMPLETE' and sha256(ROOT/'02_checkpoints'/f'{v}_best.pt')==s['checkpoint_sha256'];return s
    assert read_json(ROOT/'01_training/stage9a_tiny_gate.json')['formal_training_allowed']
    seed();m=fresh(v);reg=read_json(REG);assert state_sha(m)==reg['initial_state_sha256'][v]
    opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001);best=float('inf');bad=0;bestepoch=None;curve=[];pending=np.empty(0,dtype=np.int64);first=1;micro=128
    config={**read_json(ROOT/'00_manifest/stage9a_protocol.json')['training'],'variant':v,'objective':'0.5 micro+0.5 present-type macro' if v=='T3' else 'micro SoftCE','registration_sha256':sha256(REG),'normalization_sha256':sha256(NORM),'initial_state_sha256':state_sha(m)}
    atomic_json(root/'training_config.json',config);atomic_json(root/'normalization_manifest.json',{'source':str(NORM.relative_to(PROJECT)),'sha256':sha256(NORM),'gate_history_columns':[5,6,7],'context_not_zscored':True,'HeadTrain_only':True})
    last=root/'last.pt'
    if last.exists():
        s=torch.load(last,map_location='cpu',weights_only=False);assert s['registration_sha256']==sha256(REG)
        m.load_state_dict(s['state_dict']);opt.load_state_dict(s['optimizer_state']);best=s['best_score'];bad=s['bad'];bestepoch=s['best_epoch'];curve=s['curve'];pending=s['pending'];first=s['epoch']+1;micro=s['microbatch']
    elapsed=time.monotonic()
    for epoch in range(first,51):
        order=np.concatenate((pending,np.random.default_rng(2022+epoch).permutation(train)));used=len(order)//1024*1024;pending=order[used:]
        sums=np.zeros(3);counts=np.zeros(3,dtype=np.int64);contributions=np.zeros(3);micro_sum=macro_sum=total=0.;batches=0;m.train();beg=time.monotonic()
        for pos in range(0,used,1024):
            batch=order[pos:pos+1024];bc=np.bincount(store.types[batch],minlength=3);present=int((bc>0).sum())
            weights=np.full(3,1/1024.) if v!='T3' else .5/1024.+np.divide(.5,bc*present,out=np.zeros(3),where=bc>0)
            while True:
                opt.zero_grad(set_to_none=True);ls=np.zeros(3)
                try:
                    for start in range(0,1024,micro):
                        ix=batch[start:start+micro];args,fde,_=store.batch(ix);ty=torch.as_tensor(store.types[ix],device='cuda')
                        assert not any(a.requires_grad for a in args)
                        out=m(*args);per=per_actor_loss(out['mode_logits'],fde);weight=torch.as_tensor(weights,dtype=per.dtype,device='cuda')[ty]
                        l=(per*weight).sum();assert torch.isfinite(l);l.backward()
                        for t in range(3):ls[t]+=float(per[ty==t].detach().double().sum())
                    grad=gradient(m,predictor,r2);opt.step();break
                except torch.cuda.OutOfMemoryError:
                    if micro==32:raise
                    micro//=2;opt.zero_grad(set_to_none=True);torch.cuda.empty_cache();print('OOM_SAME_BATCH',v,micro,flush=True)
            sums+=ls;counts+=bc;contributions+=ls*weights;micro_sum+=ls.sum()/1024;macro_sum+=float((ls[bc>0]/bc[bc>0]).mean());total+=float((ls*weights).sum());batches+=1
            if pos%65536==0:print('TRAIN',v,epoch,pos,'/',used,flush=True)
        metrics=assess(m,store,dev);score=metrics['DevSelectionScore'];improved=score<best
        if improved:best=score;bestepoch=epoch;bad=0
        else:bad+=1
        row={'epoch':epoch,'TrainObjective':total/batches,'TrainMicroLoss':micro_sum/batches,'TrainMacroLoss':macro_sum/batches,**metrics,'best_flag':int(improved),'patience_count':bad,'optimizer_targets':used,'carry_targets':len(pending),'seconds':time.monotonic()-beg}
        for t,c in enumerate(CLASSES):row[c+'TrainLoss']=float(sums[t]/counts[t]);row[c+'TrainCount']=int(counts[t]);row[c+'ObjectiveContribution']=float(contributions[t]/batches)
        curve.append(row);write_csv(root/'stage9a_training_curve.csv',curve);write_csv(ROOT/'06_tables'/f'stage9a_{v}_training_curve.csv',curve)
        complete=bad>=5 or epoch==50
        cp={'state_dict':{k:x.detach().cpu().clone() for k,x in m.state_dict().items()},'variant':v,'epoch':epoch,'best_epoch':bestepoch,'best_score':best,'registration_sha256':sha256(REG),'selection':'minimum HeadDev DevSelectionScore','parameters':PARAMS[v],'normalization_sha256':sha256(NORM)}
        if improved:atomic_torch(ROOT/'02_checkpoints'/f'{v}_best.pt',cp)
        atomic_torch(last,{**cp,'optimizer_state':opt.state_dict(),'bad':bad,'curve':curve,'pending':pending,'microbatch':micro,'complete':complete})
        print('EPOCH',v,epoch,metrics,'BEST',bestepoch,'PATIENCE',bad,flush=True)
        if complete:break
    cp=torch.load(ROOT/'02_checkpoints'/f'{v}_best.pt',map_location='cpu',weights_only=False);m.load_state_dict(cp['state_dict']);selected=assess(m,store,dev);assert abs(selected['DevSelectionScore']-best)<1e-12
    s={'status':'COMPLETE','Variant':v,'selected_epoch':bestepoch,'executed_epochs':epoch,'HeadDevSelectionScore':best,'selected':selected,'checkpoint_sha256':sha256(ROOT/'02_checkpoints'/f'{v}_best.pt'),'precision':'FP32','microbatch':micro,'effective_batch':1024,'accumulation':1024//micro,'elapsed_seconds':time.monotonic()-elapsed,'official_VAL_used':False,'predictor_gradient_count':0,'R2_gradient_count':0,'candidate_requires_grad':False,'tiny_weights_reused':False,'final_unoptimized_carry':len(pending)}
    atomic_json(done,s);atomic_json(root/'gradient_audit.json',grad);del m,opt;torch.cuda.empty_cache();return s

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tiny-only',action='store_true');args=ap.parse_args()
    seed();verify();assert not FROZEN.exists(),'All frozen: training prohibited'
    reg=read_json(REG)
    for p,h in reg['training_sources'].items():assert sha256(Path(__file__).parent/p)==h,p
    assert read_json(ROOT/'01_training/stage9a_no_leakage_audit.json')['status']=='PASS'
    store=Store();train=np.flatnonzero(store.partition==1);dev=np.flatnonzero(store.partition==0)
    predictor=frozen_predictor('cpu');r2=frozen_r2('cpu');before=[state_sha(m) for m in (predictor,r2)]
    tiny(store,predictor,r2,train)
    if args.tiny_only:return
    summaries=[train_one(v,store,predictor,r2,train,dev) for v in VARIANTS]
    assert before==[state_sha(m) for m in (predictor,r2)];verify()
    rows=[{'Variant':v,'Path':str((ROOT/'02_checkpoints'/f'{v}_best.pt').relative_to(PROJECT)),'SHA256':s['checkpoint_sha256'],'SelectedEpoch':s['selected_epoch'],'HeadDevSelectionScore':s['HeadDevSelectionScore'],'Params':PARAMS[v]} for v,s in zip(VARIANTS,summaries)]
    write_csv(ROOT/'06_tables/stage9a_checkpoint_manifest.csv',rows)
    write_csv(ROOT/'06_tables/stage9a_training_summary.csv',[{k:s[k] for k in ('Variant','selected_epoch','executed_epochs','HeadDevSelectionScore','precision','microbatch','effective_batch','accumulation','elapsed_seconds')} for s in summaries])
    atomic_json(FROZEN,{'status':'FROZEN_ALL_COMPLETE','training_prohibited':True,'checkpoints':rows,'registration_sha256':sha256(REG),'normalization_sha256':sha256(NORM),'official_VAL_used':False,'frozen_predictor_R2_state_sha256':before})
    print('STAGE9_ALL_CHECKPOINTS_FROZEN',flush=True)

if __name__=='__main__':main()
