"""Fail-closed 100 random official VAL batch observer-equivalence audit."""
from pathlib import Path
import sys,os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'00_manifest'))
from stage7c_common import *
from stage7c_hooks import LaneCapture
from torch_geometric.data import Batch

@torch.no_grad()
def main():
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    ds=Stage7ASemanticDataset('val',cache_scenes=4)
    rng=np.random.default_rng(2022)
    # Choose 100 distinct canonical batches without replacement, then read in scene order.
    batches=sorted(rng.choice((len(ds)+15)//16,size=100,replace=False).tolist())
    rows=[];summary={}
    for name in ['Stage3B','Stage7A']:
        model=load_model(name);before=state_digest(model.state_dict())
        maxima={k:0. for k in ('raw_prediction','mode_logits','mode_prob')}
        for j,b in enumerate(batches):
            indices=list(range(b*16,min((b+1)*16,len(ds))))
            data=Batch.from_data_list([ds[i] for i in indices]).cuda()
            normal=model(input_for(data,name))
            capture=LaneCapture(model)
            observed=model(input_for(data,name));capture.close()
            assert capture.data['alpha'].shape[1]==8
            for k in maxima:
                diff=float((normal[k]-observed[k]).abs().max())
                maxima[k]=max(maxima[k],diff)
                assert diff<1e-6,(name,b,k,diff)
            rows.append({'model':name,'batch_index':b,'windows':len(indices),
                **{k+'_max_diff':float((normal[k]-observed[k]).abs().max()) for k in maxima}})
            if (j+1)%20==0:print('INTEGRITY',name,j+1,'/100',maxima,flush=True)
        assert state_digest(model.state_dict())==before
        summary[name]={'max_differences':maxima,'model_state_unchanged':True,'batches':100}
        del model;torch.cuda.empty_cache()
    ds.clear()
    pd.DataFrame(rows).to_csv(ROOT/'06_tables/stage7c_capture_integrity_batches.csv',index=False)
    atomic_json(ROOT/'00_manifest/stage7c_attention_capture_integrity.json',{
        'status':'PASS','models':summary,'seed':2022,'batch_indices':batches,
        'distinct_VAL_batches':100,'batch_size':16,'dropout_disabled':True,
        'observation':'forward hooks return None; post-softmax input to eval dropout',
        'kernel_policy':'deterministic CUDA for controlled equality; :4096:8 CUBLAS workspace',
        'GT_relevance_in_model_input':False,'training':False,'test_used':False})
    print('ATTENTION_CAPTURE_INTEGRITY PASS',summary,flush=True)

if __name__=='__main__':main()
