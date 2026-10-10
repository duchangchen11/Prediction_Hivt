"""Freeze historical bytes and verify selected Stage15B artifacts; no fitting."""
from pathlib import Path
import hashlib,json,subprocess,time
PROJECT=Path(__file__).resolve().parents[3]
ROOT=PROJECT/'outputs/stage16_paper_ready'
OLD=PROJECT/'outputs/stage15b_isolated_validation'
BASE='2deef55128730b02c1cf0bd9e7237a23d6865133'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,obj):
 p=Path(p);assert p.resolve().is_relative_to(ROOT);p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def main():
 for d in ('01_freeze','02_literature','03_seed_stability','04_probability','05_figures','06_source_data','07_cases','08_logs','09_reports'):(ROOT/d).mkdir(parents=True,exist_ok=True)
 tracked=subprocess.check_output(['git','ls-tree','-r','--name-only',BASE],cwd=PROJECT,text=True).splitlines()
 paths={p for p in tracked if p.startswith('outputs/')}
 paths.update(p.relative_to(PROJECT).as_posix() for p in (PROJECT/'outputs/stage2c_trainval_vehicle_baseline/04_evaluation').glob('stage2c_qualitative_main_case_new*'))
 predictors=read(OLD/'04_predictor_checkpoints/stage15b_all_frozen.json');heads=read(OLD/'07_rank_checkpoints/stage15b_all_frozen.json')
 assert predictors['Status']==heads['Status']=='FROZEN_ALL_COMPLETE'
 assert len(predictors['Checkpoints'])==3 and len(heads['Checkpoints'])==18
 selected=[]
 for c in predictors['Checkpoints']:
  p=OLD/c['path'];assert sha(p)==c['sha256'];paths.add(p.relative_to(PROJECT).as_posix());selected.append({'kind':'predictor','fold':c['fold'],'path':p.relative_to(PROJECT).as_posix(),'sha256':c['sha256'],'selected_global_step':c['selected_global_step']})
 for c in heads['Checkpoints']:
  p=OLD/f"07_rank_checkpoints/fold{c['Fold']}/{c['Model']}_best.pt";assert sha(p)==c['CheckpointSHA256'];paths.add(p.relative_to(PROJECT).as_posix());selected.append({'kind':'ranking_head','fold':c['Fold'],'model':c['Model'],'path':p.relative_to(PROJECT).as_posix(),'sha256':c['CheckpointSHA256'],'selected_epoch':c['SelectedEpoch']})
 for p in OLD.rglob('*.pt'):
  if 'checkpoint' in str(p.parent) or p.name=='last.pt':paths.add(p.relative_to(PROJECT).as_posix())
 for p in (OLD/'08_evaluation/cache').glob('*'): 
  if p.is_file():paths.add(p.relative_to(PROJECT).as_posix())
 paths.add((OLD/'09_statistics/cache/bootstrap_scene_weights.npy').relative_to(PROJECT).as_posix())
 for p in (OLD/'05_candidate_interface/cache').glob('fold*/*/manifest.json'):
  paths.add(p.relative_to(PROJECT).as_posix())
  paths.add((p.parent/'targets.csv').relative_to(PROJECT).as_posix())
 entries=[{'path':p,'bytes':(PROJECT/p).stat().st_size,'sha256':sha(PROJECT/p)} for p in sorted(paths)]
 protocol=read(OLD/'00_manifest/stage15b_protocol.json');folds=[]
 for f in protocol['folds']:
  parts=f['parts'];assert [len(parts[k]) for k in ('InnerTrain','InnerDev','OuterTest','QuarantinedHeadDev')]==[378,42,210,70]
  assert sum(map(len,parts.values()))==len(set().union(*map(set,parts.values())))==700
  for v in f['files'].values():assert sha(OLD/v['path'])==v['sha256']
  folds.append({'fold':f['fold'],'seed':f['seed'],'counts':{k:len(v) for k,v in parts.items()},'source_lists':f['files']})
 assert len(set().union(*(set(f['parts']['OuterTest']) for f in protocol['folds'])))==630
 manifest={'Status':'PASS_FROZEN','Stage':'Stage16','BaseCommit':BASE,'FrozenAtUTC':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'SelectedCheckpointCount':{'Predictors':3,'RankingHeads':18},'NoCheckpointReselection':True,'SelectedCheckpoints':selected,'Folds':folds,'Files':entries,'CachePolicy':'Candidate arrays are referenced read-only through their pre-existing signed manifests; no duplication. Frozen evaluation arrays are independently SHA256 hashed here.','HistoricalDataPolicy':'All historical versioned output files and existing Stage2C untracked drawings are byte protected; checkpoints remain local.'}
 write(ROOT/'stage16_stage15b_result_manifest.json',manifest)
 print('FROZEN',len(entries),'files; 3 predictors + 18 heads; folds PASS',flush=True)
if __name__=='__main__':main()
