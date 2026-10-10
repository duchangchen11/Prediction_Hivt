"""Seal implementation of registered analyses before new external Outer metrics."""
from pathlib import Path
import sys,datetime,ast
sys.path.insert(0,str(Path(__file__).resolve().parent))
from stage17_common import *
def main():
 assert not (ROOT/'06_evaluation/cache/complete.json').exists()
 assert not (ROOT/'06_evaluation/cache/metrics.npy').exists()
 files=['06_evaluation/stage17_evaluate.py','07_efficiency/stage17_benchmark.py','08_figures/stage17_figures.py','08_figures/stage17_figure_contract.md']
 for name in files:
  if name.endswith('.py'):ast.parse((ROOT/name).read_text())
 atomic_json(ROOT/'00_manifest/stage17_analysis_code_freeze.json',{'Status':'FROZEN_BEFORE_NEW_TNT_OUTER_RESULTS','UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'RegistrationSHA256':sha256(REG),'ProtocolChange':False,'AnalysisSHA256':{f:sha256(ROOT/f) for f in files},'OutcomeMetricsViewed':False})
 print('ANALYSIS_CODE_FROZEN',flush=True)
if __name__=='__main__':main()
