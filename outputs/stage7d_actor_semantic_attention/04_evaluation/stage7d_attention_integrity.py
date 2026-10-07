"""Reuse Stage7C hook instrumentation, fail-closed on 100 random VAL batches."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from stage7d_analysis_common import *
sys.path.insert(0,str(STAGE7C/'01_attention_capture'))
from stage7c_hooks import LaneCapture
from torch_geometric.data import Batch

@torch.no_grad()
def main():
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    ds=Stage7DSemanticDataset('val');model=model_new()
    cp=torch.load(BEST,map_location='cpu',weights_only=False);model.load_state_dict(cp['state_dict'],strict=True)
    model.eval().requires_grad_(False);before=state_digest(model.state_dict())
    batches=sorted(np.random.default_rng(2022).choice((len(ds)+15)//16,100,replace=False).tolist());rows=[]
    for j,b in enumerate(batches):
        data=Batch.from_data_list([ds[i] for i in range(b*16,min((b+1)*16,len(ds)))]).cuda()
        normal=model(model_input(data));capture=LaneCapture(model,representations=False)
        observed=model(model_input(data));capture.close()
        diff={k:float((normal[k]-observed[k]).abs().max()) for k in ('raw_prediction','mode_logits','mode_prob')}
        assert all(v<1e-6 for v in diff.values());assert capture.data['alpha'].shape[1]==8
        rows.append({'batch_index':b,'graphs':data.num_graphs,**diff})
        if (j+1)%20==0:print('ATTENTION_INTEGRITY',j+1,'/100',flush=True)
    assert state_digest(model.state_dict())==before;ds.clear()
    write_csv(ROOT/'06_tables/stage7d_capture_integrity_batches.csv',rows)
    atomic_json(ROOT/'02_model_audit/stage7d_attention_capture_integrity.json',{'status':'PASS','VAL_batches':100,'seed':2022,
        'max_differences':{k:max(r[k] for r in rows) for k in ('raw_prediction','mode_logits','mode_prob')},
        'checkpoint_sha256':sha256(BEST),'post_softmax':True,'eval_dropout_disabled':True,'model_state_unchanged':True,
        'baseline_integrity_reused':str(STAGE7C.relative_to(ROOT.parent))+'/00_manifest/stage7c_attention_capture_integrity.json',
        'kernel_policy':'controlled deterministic CUDA, CUBLAS :4096:8','training':False,'test_used':False})
    print('ATTENTION_INTEGRITY_PASS',flush=True)

if __name__=='__main__':main()
