"""Replay stored secondary SoftCE at FP32/FP64; never modify metrics or heads."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'00_manifest'))
from stage11b_common import *

def main():
    verify();torch.set_num_threads(4);f=frame();src=ROOT/'05_oof_evaluation/cache'
    assert read_json(src/'complete.json')['Status']=='PASS'
    z=np.load(src/'stage11b_oof_predictions_logits.npy');v=np.load(src/'stage11b_oof_metrics.npy',mmap_mode='r')
    fd=np.load(S8/'01_training/cache/fde.npy',mmap_mode='r')[f.source_index.to_numpy()]
    q=np.exp(-fd.astype(np.float64)+fd.min(-1,keepdims=True));q/=q.sum(-1,keepdims=True);rows=[]
    for j,name in enumerate(MODELS):
        logp=z[:,j].astype(np.float64)-z[:,j].max(-1,keepdims=True)
        logp-=np.log(np.exp(logp).sum(-1,keepdims=True));truth=-(q*logp).sum(-1)
        actual=np.asarray(v[:,j,5]);err=np.abs(actual-truth)
        replay=objective(torch.from_numpy(z[:,j].copy()),torch.from_numpy(fd.copy()),'A').numpy().astype(np.float64)
        rows.append({'Model':name,'MaxAbsoluteFP64Difference':float(err.max()),
            'MaxRelativeFP64DifferenceWithUnitFloor':float((err/np.maximum(1,np.abs(truth))).max()),
            'CountExceedingOriginalAbsoluteTolerance':int((err>2e-5).sum()),
            'MaxCPUFP32ReplayDifference':float(np.abs(actual-replay).max()),
            'FP32RelativeTolerancePASS':bool(np.allclose(actual,truth,atol=2e-5,rtol=2e-6))})
    assert all(r['FP32RelativeTolerancePASS'] for r in rows)
    result={'Status':'PASS','Scope':'Secondary SoftCE numerical replay only; primary metrics remain bitwise',
        'StoredComputation':'CUDA FP32 softmax/log_softmax and reduction',
        'IndependentComputation':'CPU float64 stable softmax/logsumexp','AbsoluteTolerance':2e-5,'RelativeTolerance':2e-6,
        'StoredMetricsOrCheckpointsChanged':False,'Rows':rows}
    atomic_json(ROOT/'09_reports/stage11b_softce_numeric_audit.json',result);print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
