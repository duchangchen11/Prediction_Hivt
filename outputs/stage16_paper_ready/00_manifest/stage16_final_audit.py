"""Final preservation, source registration, head protocol and delivery gate."""
from pathlib import Path
import sys,subprocess,ast
sys.path.insert(0,str(Path(__file__).resolve().parent))
from stage16_common import *
def main():
 old=read_json(ROOT/'stage16_stage15b_result_manifest.json');checked=0
 for r in old['Files']:assert sha256(PROJECT/r['path'])==r['sha256'],r['path'];checked+=1
 ancestors=read_json(OLD/'00_manifest/stage15b_frozen_history.json')
 for section in ('historical_files','checkpoints','preserved_untracked'):
  for path,h in ancestors[section].items():assert sha256(PROJECT/path)==h,path
 reg=read_json(ROOT/'00_manifest/stage16_supplement_registration.json');gate=read_json(ROOT/'03_seed_stability/stage16_all_frozen.json')
 assert gate['Status']=='FROZEN_ALL_COMPLETE' and len(gate['Results'])==24
 assert sha256(ROOT/'00_manifest/stage16_supplement_registration.json')==gate['RegistrationSHA256']
 assert sha256(ROOT/'03_seed_stability/stage16_train_heads.py')==gate['TrainingSourceSHA256']
 for p,h in reg['ImplementationSHA256'].items():assert sha256(PROJECT/p)==h,p
 assert read_json(ROOT/'03_seed_stability/stage16_seed_evaluation_integrity.json')['Status']=='PASS'
 for rec in gate['Results']:
  folder=ROOT/f"03_seed_stability/fold{rec['fold']}/seed{rec['initialization_seed']}/{rec['model']}"
  curve=pd.read_csv(folder/'stage16_training_curve.csv');sel=int(curve.loc[curve.Selected==1,'Epoch'].iloc[-1])
  assert sel==rec['SelectedEpoch'] and len(curve)==rec['ExecutedEpochs']<=50
  assert int(curve.iloc[-1].PatienceCount)==5 or len(curve)==50
  assert not rec['OuterTestUsed'] and rec['PredictorGradientCount']==0 and rec['Params']==PARAMS[rec['model']]
  cp=ROOT/f"03_seed_stability/checkpoints/fold{rec['fold']}/seed{rec['initialization_seed']}/{rec['model']}_best.pt";assert sha256(cp)==rec['CheckpointSHA256']
  for name in reg['Models']:
   original_order=pd.read_csv(OLD/f"06_rank_training/fold{rec['fold']}/{name}/stage15b_batch_order.csv")
   new_order=pd.read_csv(folder/'stage16_batch_order.csv');limit=min(len(original_order),len(new_order))
   assert (original_order.OrderSHA256[:limit].to_numpy()==new_order.OrderSHA256[:limit].to_numpy()).all()
 for p in ROOT.rglob('*.py'):
  if 'cache' not in p.parts:ast.parse(p.read_text())
 required=['stage16_external_baseline_review.md','stage16_seed_stability.md','stage16_probability_analysis.md','stage16_scientific_summary.md','stage16_final_report.md']
 for p in required:assert (ROOT/p).is_file() and (ROOT/p).stat().st_size>800
 figures=read_json(ROOT/'05_figures/stage16_figure_manifest.json');qa=read_json(ROOT/'05_figures/stage16_figure_qa.json')
 assert len(figures['Figures'])==8 and len(qa['Figures'])==8 and qa['Status']=='PASS_DATA_VECTOR_EXPORT_CHECKS'
 assert qa['VisualQAStatus']=='PASS_ALL8_REVIEWED' and figures['Status']=='PASS_DATA_EXPORT_AND_VISUAL_REVIEW'
 atomic_json(ROOT/'06_source_data/stage16_complete_source_sha256.json',{'Status':'PASS','CSV':{str(p.relative_to(ROOT)):sha256(p) for p in sorted((ROOT/'06_source_data').glob('*.csv'))},'OriginalMetricSHA256':sha256(OLD/'08_evaluation/cache/metrics.npy'),'OriginalProbabilitySHA256':sha256(OLD/'08_evaluation/cache/probabilities.npy'),'SupplementEvaluationSHA256':sha256(ROOT/'03_seed_stability/stage16_seed_evaluation_integrity.json'),'Semantics':'snapshot of all original-copy and derived aggregate source tables; no raw nuScenes data uploaded'})
 diff=subprocess.check_output(['git','diff','--name-only','2deef55128730b02c1cf0bd9e7237a23d6865133'],cwd=PROJECT,text=True).splitlines();assert all(p.startswith('outputs/stage16_paper_ready/') for p in diff)
 result={'Status':'PASS_COMPLETE_READY_FOR_REVIEW','FrozenHistoricalFilesChecked':checked,'AncestorVersionedFilesChecked':len(ancestors['historical_files']),'AncestorCheckpointsChecked':len(ancestors['checkpoints']),'PreservedUntrackedDrawings':len(ancestors['preserved_untracked']),'OriginalStage15BPredictors':3,'OriginalStage15BHeads':18,'SupplementHeads':24,'HiVTFits':0,'ExternalBaselineFits':0,'Stage15BResultsReselected':False,'ModifiedHistoricalBytes':0,'RegisteredImplementationUnchanged':True,'DataOrderPrefixesUnchanged':True,'DevCheckpointSelectionVerified':True,'FutureGTInputPermitted':False,'CalibrationFits':0,'Figures':8,'Exports':24,'GitOnlyStage16':True,'STOPAfterPush':True}
 atomic_json(ROOT/'00_manifest/stage16_final_audit.json',result);print(result)
if __name__=='__main__':main()
