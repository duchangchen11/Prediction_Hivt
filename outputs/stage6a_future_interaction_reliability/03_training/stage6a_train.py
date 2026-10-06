"""Two fixed ranking heads; official VAL is never loaded by this trainer."""
from pathlib import Path
import sys,time,copy
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'00_manifest'))
from stage6a_common import *
from stage6a_head import ReliabilityHead,ranking_loss

SOURCES=('00_manifest/stage6a_common.py','00_manifest/stage6a_features.py','00_manifest/stage6a_head.py',
    '00_manifest/stage6a_tests.py','01_cache/stage6a_cache.py','02_features/stage6a_build_features.py',
    '03_training/stage6a_train.py')

def input_features(features,variant):
    if variant=='R2':return features
    out=features.clone();out[:,:,12:]=0.;return out

@torch.no_grad()
def assess(model,data,variant):
    model.eval();fde=ade=loss=0.;hit=0;n=len(data['base_logits'])
    for start in range(0,n,4096):
        features=input_features(data['features_R2'][start:start+4096],variant)
        base=data['base_logits'][start:start+4096];errors=data['FDE_by_mode'][start:start+4096]
        out=model(features,base);top=out['mode_prob'].argmax(-1);idx=torch.arange(len(top),device=top.device)
        fde+=float(errors[idx,top].double().sum());ade+=float(data['ADE_by_mode'][start:start+4096][idx,top].double().sum())
        loss+=float(ranking_loss(out['mode_logits'],errors))*len(top);hit+=int((top==errors.argmin(-1)).sum())
    return {'Count':n,'Top1FDE':fde/n,'Top1ADE':ade/n,'HitRate':hit/n,'rank_loss':loss/n}

def train_one(variant,cpu_train,cpu_dev,initial,registration):
    c=config();seed_all(2022);model=ReliabilityHead().cuda();model.load_state_dict(initial,strict=True)
    assert state_digest(model.state_dict())==registration['initial_head_state_sha256']
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    train={k:v.cuda() for k,v in cpu_train.items() if torch.is_tensor(v)}
    dev={k:v.cuda() for k,v in cpu_dev.items() if torch.is_tensor(v)}
    best=float('inf');bad=0;best_epoch=None;curve=[];start_time=time.monotonic()
    zero=assess(model,dev,variant);atomic_json(ROOT/'03_training'/f'stage6a_{variant.lower()}_step0.json',zero)
    assert abs(zero['Top1FDE']-registration['headdev_base_Top1FDE'])<1e-12
    last=ROOT/'07_checkpoints'/f'stage6a_{variant.lower()}_last.pt';start_epoch=1
    if last.exists():
        saved=torch.load(last,map_location='cpu',weights_only=False)
        assert saved['registration_sha256']==sha256(ROOT/'00_manifest/stage6a_training_registration.json')
        model.load_state_dict(saved['state_dict']);optimizer.load_state_dict(saved['optimizer_state'])
        best=saved['best_Top1FDE'];bad=saved['bad_epochs'];best_epoch=saved['best_epoch'];curve=saved['curve'];start_epoch=saved['epoch']+1
        if saved['complete']:return read_json(ROOT/'03_training'/f'stage6a_{variant.lower()}_summary.json')
    for epoch in range(start_epoch,51):
        model.train();generator=torch.Generator().manual_seed(2022+epoch)
        order=torch.randperm(len(train['base_logits']),generator=generator).cuda();total_loss=0.;seen=0
        for start in range(0,len(order),1024):
            idx=order[start:start+1024]
            x=input_features(train['features_R2'][idx],variant);out=model(x,train['base_logits'][idx])
            loss=ranking_loss(out['mode_logits'],train['FDE_by_mode'][idx]);assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True);loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            optimizer.step();total_loss+=float(loss.detach())*len(idx);seen+=len(idx)
        metrics=assess(model,dev,variant);improved=metrics['Top1FDE']<best
        if improved:best=metrics['Top1FDE'];best_epoch=epoch;bad=0
        else:bad+=1
        row={'Variant':variant,'Epoch':epoch,'Train_rank_loss':total_loss/seen,**metrics,
            'Strict_improvement':int(improved),'Best_epoch':best_epoch,'Bad_epochs':bad}
        curve.append(row);write_csv(ROOT/'03_training'/f'stage6a_{variant.lower()}_training_curve.csv',curve)
        metadata={'variant':variant,'epoch':epoch,'best_epoch':best_epoch,'selection':'HeadDev70 scenes from TRAIN700 Top1FDE strict improvement',
            'predictor_sha256':PREDICTOR_SHA,'split_sha256':sha256(SPLIT),'normalization_sha256':sha256(NORM),
            'config_sha256':sha256(CONFIG),'training_sources':registration['source_sha256'],'parameters':673,'seed':2022}
        checkpoint={'state_dict':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},'metadata':metadata}
        if improved:atomic_torch(head_path(variant),checkpoint)
        complete=bad>=5 or epoch==50
        atomic_torch(last,{**checkpoint,'optimizer_state':optimizer.state_dict(),'registration_sha256':sha256(ROOT/'00_manifest/stage6a_training_registration.json'),
            'epoch':epoch,'best_Top1FDE':best,'bad_epochs':bad,'best_epoch':best_epoch,'curve':curve,'complete':complete})
        print('HEAD_EPOCH',variant,epoch,'Dev_Top1FDE',metrics['Top1FDE'],'best',best_epoch,best,'bad',bad,flush=True)
        if complete:break
    best_saved=torch.load(head_path(variant),map_location='cpu',weights_only=False)
    model.load_state_dict(best_saved['state_dict']);reloaded=assess(model,dev,variant)
    assert abs(reloaded['Top1FDE']-best)<1e-12
    summary={'status':'COMPLETE','variant':variant,'best_epoch':best_epoch,'executed_epochs':epoch,
        'stop_reason':'patience_5' if bad>=5 else 'maximum_epochs_50','checkpoint_sha256':sha256(head_path(variant)),
        'head_params':673,'selected_HeadDev_metrics':reloaded,'headdev_step0':zero,'elapsed_seconds':time.monotonic()-start_time,
        'split_sha256':sha256(SPLIT),'normalization_sha256':sha256(NORM),'config_sha256':sha256(CONFIG),
        'predictor_sha256':PREDICTOR_SHA,'seed':2022,'official_VAL_used':False,'full_horizon_labels_only':True,'NaN':0,'Inf':0}
    atomic_json(ROOT/'03_training'/f'stage6a_{variant.lower()}_summary.json',summary)
    del model,optimizer,train,dev;torch.cuda.empty_cache();return summary

def main():
    verify_frozen();f=read_json(ROOT/'02_features/stage6a_feature_manifest.json');assert f['status']=='PASS'
    assert read_json(ROOT/'00_manifest/stage6a_unit_audit.json')['status']=='PASS'
    assert read_json(ROOT/'00_manifest/stage6a_real_batch_audit.json')['status']=='PASS'
    assert read_json(ROOT/'00_manifest/stage6a_no_future_leakage_audit.json')['status']=='PASS'
    train=load_features('headtrain');dev=load_features('headdev')
    assert train['normalization_sha256']==dev['normalization_sha256']==sha256(NORM)
    assert {r['scene_token'] for r in train['rows']}.isdisjoint({r['scene_token'] for r in dev['rows']})
    seed_all(2022);a=ReliabilityHead();seed_all(2022);b=ReliabilityHead()
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
    initial=copy.deepcopy(a.state_dict());idx=torch.arange(len(dev['base_logits']));base_top=dev['base_logits'].argmax(-1)
    base_fde=float(dev['FDE_by_mode'][idx,base_top].double().mean())
    regpath=ROOT/'00_manifest/stage6a_training_registration.json'
    registration={'status':'REGISTERED_BEFORE_BOTH_HEADS','source_sha256':{p:sha256(ROOT/p) for p in SOURCES},
        'initial_head_state_sha256':state_digest(initial),'R1_R2_shared_parameter_max_diff':0.,'headdev_base_Top1FDE':base_fde,
        'config_sha256':sha256(CONFIG),'normalization_sha256':sha256(NORM),'split_sha256':sha256(SPLIT),
        'formal_heads':2,'seed':2022,'predictor_backprop':False,'official_VAL_used':False}
    if regpath.exists():assert read_json(regpath)==registration
    else:atomic_json(regpath,registration)
    results=[train_one(v,train,dev,initial,registration) for v in ('R1','R2')]
    verify_frozen(shards=True)
    atomic_json(ROOT/'03_training/stage6a_training_summary.json',{'status':'COMPLETE','heads':results,
        'formal_heads':2,'same_initial_weights':True,'same_batch_orders':True,'official_VAL_used':False,'predictor_frozen':True})
    atomic_json(ROOT/'07_checkpoints/stage6a_checkpoint_manifest.json',{'status':'PASS','heads':results,
        'predictor_sha256':PREDICTOR_SHA,'config_sha256':sha256(CONFIG),'normalization_sha256':sha256(NORM),'split_sha256':sha256(SPLIT)})
    print('STAGE6A_HEAD_TRAINING_COMPLETE',results,flush=True)

if __name__=='__main__':torch.set_num_threads(4);main()
