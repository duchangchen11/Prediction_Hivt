"""Freeze the public V1 and all prior project files before semantic analysis."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'00_manifest'))
from stage7a_common import *
def main():
    assert git('rev-parse','HEAD')==BASE and git('branch','--show-current')=='stage7a/semantic-map-enhancement'
    assert not git('diff','--name-only') and not git('diff','--cached','--name-only')
    s=read_json(STAGE6/'03_training/stage6a_r2_summary.json');a=read_json(STAGE6/'00_manifest/stage6a_final_audit.json')
    assert s['checkpoint_sha256']==sha256(R2) and a['RecommendedFinalVariant']=='R2'
    paths={'Stage5A_checkpoint':PREDICTOR,'R2_checkpoint':R2,'Stage6A_config':STAGE6/'00_manifest/stage6a_config.json',
        'Stage6A_normalization':STAGE6/'02_features/stage6a_normalization.json',
        'Stage6A_report':STAGE6/'09_reports/stage6a_final_report.md','Stage6A_actor_results':ACTORS}
    ref={'status':'FROZEN','git_commit':BASE,'git_tag':TAG,'annotated_tag_object':git('rev-parse',TAG),
        'tag_message':'Freeze paper V1: Stage5A + future interaction reliability R2',
        'files':{k:{'path':str(p.relative_to(PROJECT)),'sha256':sha256(p)} for k,p in paths.items()},
        'scientific_conclusions':{k:a[k] for k in ('ReliabilityHead','FutureInteractionContribution','PaperUsableReliability','RecommendedFinalVariant')},
        'checkpoint_copied':False,'training_authorized':False,'tag_remote_status':'PENDING_TRANSPORT_VERIFICATION'}
    assert ref['files']['Stage5A_checkpoint']['sha256']=='88fe3feb59b7e830ec1e40aa917484386e0d2799d0f6116f410adda2d97fdce7'
    atomic_json(ROOT/'00_manifest/stage7a_v1_frozen_reference.json',ref)
    files={n:sha256(PROJECT/n) for n in git('ls-files').splitlines() if not n.startswith(str(ROOT.relative_to(PROJECT))+'/')}
    prior=read_json(STAGE6/'00_manifest/stage6a_frozen_references.json')
    for n,d in prior['files'].items():assert sha256(PROJECT/n)==d;files[n]=d
    for value in paths.values():files[str(value.relative_to(PROJECT))]=sha256(value)
    atomic_json(ROOT/'00_manifest/stage7a_frozen_previous.json',{'status':'PASS','base_commit':BASE,'files':files,
        'scene_shards':prior['scene_shards'],'all_old_code_and_results_read_only':True})
    atomic_json(ROOT/'00_manifest/stage7a_figure_contract.json',{'status':'REGISTERED_BEFORE_PLOT_CODE','backend':'python',
        'core_question':'Do real connector geometry and topology support static turn semantics and valid semantic exposure audits?',
        'archetype':'quantitative histogram and map geometry grids','exports':['PNG300dpi','PDF editable','SVG editable'],
        'turn_examples':'at least20 random connectors per left/straight/right; all if fewer; actual centerline and entrance/exit arrows',
        'statistics':'map record counts and descriptive actor-window exposure; no performance tuning or causal inference',
        'integrity':'no smoothing, invented maps, dynamic signal states or trained models',
        'review_risks':['angle sign in global and ego frames','geometric control association is derived, not source lane label',
            'token-level semantics broadcast to all segments may differ from direct object proximity','official raw trainval mount currently unavailable; use frozen official scene shards and available original expansion JSON']})
    print('STAGE7A_V1_FREEZE_PASS',ref['files']['R2_checkpoint']['sha256'],flush=True)
if __name__=='__main__':main()
