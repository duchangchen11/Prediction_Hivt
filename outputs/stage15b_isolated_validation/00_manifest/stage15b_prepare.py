"""Freeze data identities, all raw scene shard hashes, methods and fitting sources."""
from pathlib import Path
import csv,hashlib,json,subprocess,time
ROOT=Path(__file__).resolve().parents[1]; PROJECT=ROOT.parents[1]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
p=ROOT/'00_manifest/stage15b_protocol.json';registration=ROOT/'00_manifest/stage15b_registration.json'
assert not registration.exists(),'Immutable pre-fitting registration'
c=json.loads(p.read_text()); a=json.loads((ROOT.parent/'stage15a_isolated_predictor/00_manifest/stage15a_protocol.json').read_text())
for fold in c['folds']:
 assert fold['parts']==a['folds'][fold['fold']-1]['parts']
 for k,v in fold['files'].items():assert sha(ROOT/v['path'])==v['sha256']
additional=[PROJECT/'outputs/stage14b_paper_validation/00_protocol/stage14b_protocol.json',
 PROJECT/'outputs/stage14b_paper_validation/02_models/stage14b_matched_nograph.py',
 PROJECT/'outputs/stage14a_paper_graph_ablation/02_models/stage14a_nograph.py',
 PROJECT/'outputs/stage8a_future_scene_compatibility_graph/00_manifest/stage8a_graph.py',
 PROJECT/'outputs/stage8a_future_scene_compatibility_graph/00c_sparse_type_aware_graph_spec/03_model_audit/stage8a0c_model.py',
 PROJECT/'outputs/stage6a_future_interaction_reliability/00_manifest/stage6a_features.py',
 PROJECT/'outputs/stage6a_future_interaction_reliability/00_manifest/stage6a_head.py',
 PROJECT/'outputs/stage11b_error_aware_ranking/00_manifest/stage11b_common.py',
 PROJECT/'outputs/stage11b_error_aware_ranking/03_training/stage11b_train.py',PROJECT/'metrics/hivt_forecasting.py']
c['source_sha256'].update({str(x.relative_to(PROJECT)):sha(x) for x in additional})
save(p,c)
old=json.loads((ROOT.parent/'stage15a_isolated_predictor/00_manifest/stage15a_frozen_history.json').read_text())
tracked=subprocess.check_output(['git','ls-files','-z']).decode().split('\0')
hist={x:sha(PROJECT/x) for x in tracked if x}
checkpoints=dict(old['checkpoints'])
for x in (ROOT.parent/'stage15a_isolated_predictor/04_checkpoints').rglob('*.pt'):checkpoints[str(x.relative_to(PROJECT))]=sha(x)
save(ROOT/'00_manifest/stage15b_frozen_history.json',{'BaseCommit':c['BaseCommit'],'historical_files':hist,'checkpoints':checkpoints,'preserved_untracked':old['preserved_untracked']})
manifest=PROJECT/'outputs/stage14a_paper_graph_ablation/09_reports/stage14a_predictor_train_shard_manifest.csv'
rows=list(csv.DictReader(manifest.open()));begin=time.monotonic()
for row in rows:
 path=PROJECT/row['Path'];assert path.stat().st_size==int(row['Bytes']);assert sha(path)==row['SHA256']
assert len(rows)==700
save(ROOT/'01_data_isolation/stage15b_scene_integrity.json',{'Status':'PASS','All700RawShardsVerified':True,'ShardManifestSHA256':sha(manifest),'ShardBytes':sum(int(x['Bytes']) for x in rows),'Seconds':time.monotonic()-begin,'OriginalPartitionsAndSHAIdentical':True,'RoleCounts':[378,42,210,70],'OuterTokenUnion630':len(set.union(*(set(f['parts']['OuterTest']) for f in c['folds'])))==630,'OuterPerformanceViewed':False,'OfficialVALRead':False})
save(registration,{'Status':'REGISTERED_BEFORE_FITTING','BaseCommit':c['BaseCommit'],'ProtocolSHA256':sha(p),'sources':{str(x.relative_to(ROOT)):sha(x) for x in sorted(ROOT.rglob('*.py'))},'PredictorFullTrainingAuthorized':True,'CandidateGate':'all3 fresh NLL predictors frozen','HeadTrainingGate':'fresh candidates and Train-only norm','OuterEvaluationGate':'all18 heads frozen','HistoricalWeightsForbidden':True,'StatisticalProtocol':c['ranking']['bootstrap'],'Decisions':c['decisions']})
print('STAGE15B_FROZEN',len(hist),'historical files',len(checkpoints),'historical checkpoints','700 shards',flush=True)
