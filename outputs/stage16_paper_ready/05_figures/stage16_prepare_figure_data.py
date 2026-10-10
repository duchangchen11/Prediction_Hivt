"""Exact frozen tables and four actual local case-coordinate exports."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'00_manifest'))
from stage16_common import *
def main():
 sources={
 'stage16_seven_model_metrics.csv':OLD/'stage15b_end_to_end_metrics.csv',
 'stage16_primary_bootstrap.csv':OLD/'stage15b_bootstrap_ci.csv',
 'stage16_head_compute.csv':OLD/'09_statistics/stage15b_head_compute.csv',
 'stage16_system_compute.csv':OLD/'09_statistics/stage15b_system_compute.csv',
 'stage16_candidate_distribution.csv':OLD/'09_statistics/stage15b_candidate_error_distributions.csv',
 'stage16_switch_harm.csv':OLD/'09_statistics/stage15b_switch_harm.csv'}
 for dest,source in sources.items():
  (ROOT/'06_source_data'/dest).write_bytes(source.read_bytes())
 f,m,p,z=frozen_eval();seen=set();records=[]
 for group in ('MovingVehicle','Pedestrian'):
  delta=m[:,MODELS.index('G-C'),0]-m[:,MODELS.index('NG-C'),0];eligible=np.flatnonzero(groups(f)[group])
  for outcome in ('Improvement','Failure'):
   order=eligible[np.argsort(delta[eligible],kind='stable')]
   if outcome=='Failure':order=order[::-1]
   i=next(int(i) for i in order if int(i) not in seen);seen.add(i);row=f.iloc[i];fold=int(row.fold)
   path=OLD/f'05_candidate_interface/cache/fold{fold}/OuterTest/{row.scene_token}.pt'
   payload=torch.load(path,map_location='cpu',weights_only=False)
   w=next(x for x in payload if x['observable']['sample_token']==row.sample_token);obs=w['observable'];lab=w['labels'];node=int(row.node_index)
   assert obs['instance_tokens'][node]==row.instance_token
   pred=obs['ego_prediction'][node].numpy();gt=lab['future_xy'][node].numpy();history=obs['history'][node].numpy();pad=obs['history_padding'][node].numpy()
   current=history[4];dist=np.linalg.norm(pred[:,-1]-gt[-1],axis=-1)
   top={name:int(p[i,j].argmax()) for j,name in enumerate(MODELS)}
   for name in MODELS:assert np.isclose(dist[top[name]],m[i,MODELS.index(name),0],atol=1e-5,rtol=1e-5)
   assert np.isclose(dist.min(),m[i,0,13],atol=1e-5,rtol=1e-5)
   coords=[]
   for entity,xy,times in [('GT',gt,obs['future_sample_times_metadata'].numpy()),*[(f'candidate{k}',pred[k],obs['future_sample_times_metadata'].numpy()) for k in range(6)],('history',history[~pad],obs['history_times'].numpy()[~pad])]:
    for t,(xyv,tm) in enumerate(zip(xy,times)):coords.append({'Entity':entity,'Point':t,'ActualTimeSeconds':float(tm),'x_m':float(xyv[0]),'y_m':float(xyv[1])})
   cid=f'case{len(records)+1}';csv=ROOT/f'07_cases/{cid}_coordinates.csv';dump(csv,coords)
   rec={'Case':cid,'Group':group,'Outcome':outcome,'Selection':'unique maximum gain/harm in frozen G-C minus NG-C within registered group; illustrative extremum','ActorRecordIndex':i,'Fold':fold,'SceneToken':row.scene_token,'SampleToken':row.sample_token,'InstanceToken':row.instance_token,'NodeIndex':node,'DeltaTop1FDE':float(delta[i]),'OracleMinFDE6':float(dist.min()),'SelectedModes':top,'ModeErrors':dist.tolist(),'OriginalModeProbabilities':{name:p[i,j].tolist() for j,name in enumerate(MODELS)},'CurrentXY':current.tolist(),'CoordinateFrame':'unmodified t0 ego xy meters; identical axis across compared methods in each case','Times':'exact original cached history/future timestamps','LocalCoordinateFile':csv.relative_to(ROOT).as_posix(),'CoordinateFileSHA256':sha256(csv),'SourceContextPath':path.relative_to(PROJECT).as_posix(),'SourceContextSHA256':sha256(path),'PredictorCheckpointSHA256':row.predictor_checkpoint_sha256,'FDECoordinateCheck':'PASS max difference tolerance1e-5 + rtol1e-5','ModelInputGT':False,'GTUse':'offline illustration only'}
   atomic_json(ROOT/f'07_cases/{cid}.json',rec);records.append(rec)
 atomic_json(ROOT/'07_cases/stage16_case_manifest.json',{'Status':'PASS_ACTUAL_FROZEN_COORDINATES','Cases':records,'FabricatedTrajectories':False,'CoordinatesLocalOnly':True})
 atomic_json(ROOT/'06_source_data/stage16_source_data_manifest.json',{'Status':'PASS','OriginalTables':{k:{'source':v.relative_to(PROJECT).as_posix(),'sha256':sha256(v),'export_sha256':sha256(ROOT/'06_source_data'/k)} for k,v in sources.items()},'FrozenEvaluationSources':{k:sha256(OLD/'08_evaluation/cache'/k) for k in ('metrics.npy','logits.npy','probabilities.npy','actor_records.csv')},'CaseManifestSHA256':sha256(ROOT/'07_cases/stage16_case_manifest.json'),'PlotContractSHA256':sha256(ROOT/'05_figures/stage16_figure_contract.md')})
 print('PASS exact table copies and four actual case exports',[(r['Group'],r['Outcome'],r['DeltaTop1FDE']) for r in records])
if __name__=='__main__':main()
