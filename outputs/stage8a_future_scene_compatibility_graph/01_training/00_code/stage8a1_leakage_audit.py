"""New cache adapter future-poison audit; frozen selector has its own200-window gate."""
from stage8a1_common import *

def main():
    seed();verify();index=SparseSemanticIndex();records=[r for r in read_json(C/'02_graph_cache/stage8a0c_selector_manifest.json')['batches'] if r['split']=='train']
    records=np.random.default_rng(2022).choice(records,8,replace=False).tolist();count=0
    for rec in records:
        source=torch.load(S6/rec['source_path'],map_location='cpu',weights_only=False)
        with np.load(C/rec['path']) as sel,np.load(ROOT/'02_graph_cache'/rec['frame_path']) as frame:
            for j,w in enumerate(source['windows'][:2]):
                a=graph_window(w,rec,j,index,sel,frame);poison=dict(w)
                poison['GT']=torch.full_like(w['GT'],float('nan'));poison['future_mask']=~w['future_mask'];poison['target_mask']=~w['target_mask']
                b=graph_window(poison,rec,j,index,sel,frame)
                for k in (1,2,3,4):
                    assert np.array_equal(np.asarray(a[k]),np.asarray(b[k]))
                for k in a[5]:assert np.array_equal(a[5][k],b[5][k]),k
                assert torch.equal(a[0].predicted,b[0].predicted) and torch.equal(a[0].original_logits,b[0].original_logits)
                count+=1
    atomic_json(ROOT/'01_training/stage8a1_future_leakage_audit.json',{'status':'PASS','TRAIN_windows':count,
        'GT_nan_and_future_target_masks_inverted':True,'observable_graphs_bitwise_identical':True,
        'frozen_0C_additional_audit':'200 source windows, including100TRAIN/100VAL; Stage8A-1 current audit never opensVAL',
        'labels_joined_only_after_observable_graph':True})
    print('FUTURE_POISON_PASS',count,flush=True)

if __name__=='__main__':main()
