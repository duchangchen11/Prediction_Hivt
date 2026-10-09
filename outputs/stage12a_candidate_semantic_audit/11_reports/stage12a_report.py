"""Evidence-grounded Stage12A decision; no new prediction or ranking model."""
from pathlib import Path
import sys
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'00_manifest'),str(Path(__file__).resolve().parents[1]/'03_feature_statistics')]
from stage12a_analysis_common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def table(frame,columns=None):
    frame=frame[columns] if columns else frame
    def fmt(x):
        if x is None or (not isinstance(x,str) and pd.isna(x)):return 'not estimable'
        if isinstance(x,(float,np.floating)):return f'{x:.6g}'
        return str(x).replace('|',' / ').replace('\n',' ')
    return '| '+' | '.join(frame.columns)+' |\n| '+' | '.join(['---']*len(frame.columns))+' |\n'+'\n'.join('| '+' | '.join(fmt(x) for x in row)+' |' for row in frame.itertuples(index=False,name=None))
def csv(path):return pd.read_csv(ROOT/path)
def emit_figures(cov,controls,matched,switch):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'svg.fonttype':'none','pdf.fonttype':42,
        'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(13,4.2));colors=['#246c8a','#ca7731','#4b8d67']
    for j,t in enumerate(ACTOR_TYPES):
        a=cov[(cov.Scope=='Pooled')&(cov.Group==t)].iloc[0]
        axes[0].bar(j,a.CandidateMapCoverage,color=colors[j]);axes[0].text(j,a.CandidateMapCoverage+.008,f'{a.CandidateMapCoverage:.2%}',ha='center',fontsize=8)
    axes[0].set(xticks=range(3),xticklabels=ACTOR_TYPES,ylim=(0,1.12),ylabel='Fraction of six candidates',title='Map coverage (all actors retained)')
    for ax,typ in zip(axes[1:],['Vehicle','Pedestrian']):
        a=controls[(controls.Scope=='Pooled')&(controls.ActorType==typ)];positions=np.arange(len(a))
        for j,(tag,color) in enumerate(zip(['Real','NC1','NC2'],['#246c8a','#9a9a9a','#ca7731'])):
            vals=a[tag+'_SceneMeanRho'].to_numpy();ax.bar(positions+(j-1)*.23,vals,width=.23,color=color,label=tag)
        ax.axhline(0,color='black',lw=.7);ax.axhline(.05,color='black',lw=.8,ls=':')
        labels=[str(s).replace('centerline_','lane ').replace('drivable_inside_fraction','−drivable inside').replace('walkway_inside_fraction','−walkway inside').replace('crosswalk_intersects','−crossing intersects').replace('crosswalk_distance','crossing distance').replace('_',' ') for s in a.Feature]
        ax.set(xticks=positions,xticklabels=labels,ylabel='Scene mean within-actor Spearman',title=typ+' primary associations')
        ax.tick_params(axis='x',labelrotation=20);ax.legend(frameon=False,fontsize=8)
    fig.suptitle('Stage12A: HeadTrain630 OOF development diagnosis; descriptive associations',fontsize=11);fig.tight_layout()
    stems=[]
    for ext in ['png','svg','pdf']:
        path=ROOT/'10_figures'/f'stage12a_coverage_and_controls.{ext}';fig.savefig(path,dpi=180,bbox_inches='tight');stems.append(path)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.8))
    for ax,typ in zip(axes,['Vehicle','Pedestrian']):
        a=matched[(matched.Scope=='Pooled')&(matched.ActorType==typ)].copy();y=np.arange(len(a));ax.axvline(0,c='black',lw=.7)
        for j,(tag,color) in enumerate([('NC1','#246c8a'),('NC2','#ca7731')]):
            mean=a['DeltaRealMinus'+tag].to_numpy();lo=a['Delta'+tag+'_CI95Low'].to_numpy();hi=a['Delta'+tag+'_CI95High'].to_numpy()
            ok=np.isfinite(mean)&np.isfinite(lo)&np.isfinite(hi)
            ax.errorbar(mean[ok],y[ok]+(j-.5)*.18,xerr=np.vstack([np.maximum(0,mean[ok]-lo[ok]),np.maximum(0,hi[ok]-mean[ok])]),fmt='o',color=color,capsize=3,label='Real − '+tag)
        signed_labels=[('−'+s.replace('_',' ') if s in ['drivable_inside_fraction','walkway_inside_fraction','crosswalk_intersects'] else s.replace('_',' ')) for s in a.Feature]
        ax.set(yticks=y,yticklabels=signed_labels,xlabel='Paired signed-rho contrast (95% CI)',title=typ+' / same actor & displacement bin');ax.legend(frameon=False,fontsize=8)
    fig.suptitle('Matched motion: descriptive 2000 scene bootstrap; independent prediction gain unresolved',fontsize=10);fig.tight_layout()
    for ext in ['png','svg','pdf']:
        path=ROOT/'10_figures'/f'stage12a_matched_motion_controls.{ext}';fig.savefig(path,dpi=180,bbox_inches='tight');stems.append(path)
    plt.close(fig)
    for path in stems:
        if path.suffix=='.svg':path.write_text('\n'.join(s.rstrip() for s in path.read_text().splitlines())+'\n')
    atomic_json(ROOT/'10_figures/stage12a_figure_audit.json',{'Conclusion':'coverage and prespecified control contrasts, no predictive performance gain claim',
        'Evidence':'unchanged OOF actors; paired descriptive scene associations; no trained semantic model',
        'Files':{p.name:sha256(p) for p in stems},'ExportFormats':['PNG','SVG','PDF']})
def main():
    verification=read_json(ROOT/'11_reports/stage12a_verification.json');assert verification['Status']=='PASS'
    nc=read_json(ROOT/'08_negative_controls/stage12a_negative_control_audit.json');cache=read_json(ROOT/'02_semantic_cache/stage12a_semantic_cache_manifest.json')
    cov=csv('03_feature_statistics/stage12a_semantic_coverage.csv');controls=csv('08_negative_controls/stage12a_negative_controls.csv')
    matched=csv('07_incremental_information/stage12a_incremental_information.csv');switch=csv('03_feature_statistics/stage12a_switch_semantic_analysis.csv')
    v=csv('04_vehicle_analysis/stage12a_vehicle_semantic_error.csv');ped=csv('05_pedestrian_analysis/stage12a_pedestrian_semantic_error.csv')
    features=csv('03_feature_statistics/stage12a_candidate_feature_statistics.csv');switchfeatures=csv('03_feature_statistics/stage12a_switch_feature_differences.csv')
    motion=csv('03_feature_statistics/stage12a_motion_semantic_associations.csv');corr=csv('07_incremental_information/stage12a_semantic_geometry_interaction_correlations.csv')
    f,fd,ad,top,best,p,z,calibrated=load_labels();data=load_semantics();ii=np.arange(len(f));metrics=[]
    for scope,sm in scopes(f):
        for group,gm in groups(f).items():
            mask=sm&gm
            if not mask.any():continue
            for name,j in [('R0',0),('FoldR2',1),('A SoftCE',2),('C Raw',4),('C Calibrated',4)]:
                chosen=top[:,j];pp=calibrated if name=='C Calibrated' else p[:,j]
                metrics.append({'Scope':scope,'Group':group,'Model':name,'Count':int(mask.sum()),
                    'Top1FDE':float(fd[ii[mask],chosen[mask]].astype(float).mean()),'Top1ADE':float(ad[ii[mask],chosen[mask]].astype(float).mean()),
                    'minFDE6':float(fd[mask].min(-1).astype(float).mean()),'OracleGap':float((fd[ii[mask],chosen[mask]]-fd[mask].min(-1)).astype(float).mean()),
                    'MeanTop1Confidence':float(pp[ii[mask],chosen[mask]].mean()),'NewModelOrCandidate':False})
    dump('03_feature_statistics/stage12a_frozen_ranking_results.csv',metrics)
    signal=[nc['VehicleSemanticSignal'],nc['PedestrianSemanticSignal']]
    result='GO' if 'PROMISING' in signal else ('CONDITIONAL_GO' if 'WEAK' in signal else 'STOP')
    decision={k:verification[k] for k in ['MapIntegrity','CoordinateAudit','GTLeakage','CandidateIdentity','SemanticCoverage','NegativeControl']}
    decision.update({'Stage':'Stage12A','BaseCommit':BASE,'OOFScenes':630,'OOFActors':260151,'BicycleAnalyzed':'YES',
        **{k:nc[k] for k in ['VehicleSemanticSignal','PedestrianSemanticSignal','SemanticIncrementalValue']},
        'Stage12A':result,'ReadyForStage12B':'YES' if result=='GO' else 'NO',
        'FrozenCandidatesAndTop1':'CONFIRMED','SemanticFeatureExtraction':'CONFIRMED',
        'ReproducibleDescriptiveAssociation':'STRONGLY_SUGGESTED' if 'PROMISING' in signal else 'NOT_ESTABLISHED',
        'GenuineIndependentPredictionGain':'UNRESOLVED','TrainingExecuted':False,'NewCheckpointCreated':False,
        'OfficialVALOrTestUsed':False,'HeadDevEvaluationUsed':False,'Action':'STOP; wait for 大脑AI review; no Stage12B implementation or training authorized',
        'EvidenceScope':'previously inspected OOF development exploration; scene-cluster CIs descriptive and unadjusted',
        'ProtocolSHA256':sha256(PROTOCOL),'SemanticManifestSHA256':sha256(ROOT/'02_semantic_cache/stage12a_semantic_cache_manifest.json'),
        'VerificationSHA256':sha256(ROOT/'11_reports/stage12a_verification.json'),'FeatureCriteria':nc['FeatureCriteria'],
        'CandidateCoverage':{t:float(cov[(cov.Scope=='Pooled')&(cov.Group==t)].iloc[0].CandidateMapCoverage) for t in ACTOR_TYPES}})
    atomic_json(ROOT/'11_reports/stage12a_scientific_decision.json',decision);emit_figures(cov,controls,matched,switch)
    coord=read_json(ROOT/'01_map_integrity/stage12a_coordinate_audit.json');stat=read_json(ROOT/'03_feature_statistics/stage12a_statistics_audit.json')
    criteria=pd.DataFrame(nc['FeatureCriteria']);cases=csv('09_cases/stage12a_case_manifest.csv');comparison=csv('03_feature_statistics/stage12a_stage8_coverage_comparison.csv')
    pr=controls[(controls.Scope=='Pooled')&(controls.ActorType=='Pedestrian')&(controls.Feature=='walkway_inside_fraction')].iloc[0]
    pm=matched[(matched.Scope=='Pooled')&(matched.ActorType=='Pedestrian')&(matched.Feature=='walkway_inside_fraction')].iloc[0]
    motionfocus=motion[(motion.Scope=='Pooled')&motion.Group.isin(['MovingVehicle','Pedestrian5-8m'])]
    mainfields=['centerline_mean_distance','centerline_heading_error','drivable_inside_fraction','crosswalk_distance','crosswalk_intersects','walkway_inside_fraction']
    errorcols=['Group','Feature','WrongModeActors','PairedMapFeatureActors','Mean','Median','P90','P95','MeanOracleGap']
    vc=v[(v.Scope=='Pooled')&v.Group.isin(['Vehicle','MovingVehicle','StoppedVehicle','ParkedVehicle','Vehicle>5m'])&v.Feature.isin(mainfields[:3])]
    pc=ped[(ped.Scope=='Pooled')&ped.Group.isin(['Pedestrian','Pedestrian<5m','Pedestrian>5m','Pedestrian5-8m'])&ped.Feature.isin(mainfields[3:])]
    casecols=['Category','Ordinal','Fold','CFDE','R2FDE','OracleFDE','DeltaFDE','CMapValid','Figure']
    report=f'''# Stage12A Candidate Trajectory Semantic Consistency Audit

## Scientific decision and evidence boundary

Stage12A = **{result}**; VehicleSemanticSignal = **{nc['VehicleSemanticSignal']}**; PedestrianSemanticSignal = **{nc['PedestrianSemanticSignal']}**; SemanticIncrementalValue = **{nc['SemanticIncrementalValue']}**. ReadyForStage12B = **{decision['ReadyForStage12B']}** is a review recommendation only. Execution stops here. No model was trained, no score/candidate was changed, and no new checkpoint was created.

**CONFIRMED:** static map extraction, correct ego/global coordinates, complete actor/candidate identity, and frozen R0/FoldR2/A/C Raw/C Calibrated Top1 results. **STRONGLY_SUGGESTED** applies only to features meeting the registered cross-fold and both-control screen shown below. **UNRESOLVED:** independent prediction gain beyond existing motion/interaction features. A descriptive semantic association, even after coarse displacement matching, does not establish causality or useful predictive gain.

This is HeadTrain630 development exploration using previously inspected Stage11B/11C OOF results, not an independent confirmatory test. No historical HeadDev70, official VAL150 or test evaluation was performed. All 2000 scene-cluster intervals are descriptive, unadjusted for the exploratory feature family. The registered protocol was hashed before extraction and label analysis; all630 semantic scene caches were frozen before FDE/ADE/GT analysis.

The qualifying signal is Pedestrian **−walkway inside fraction**: pooled oriented rho={pr.Real_SceneMeanRho:.6f} on {int(pr.CommonActors)} paired eligible actors/{int(pr.Scenes)} scenes, positive in all3folds, with real-minus-both-control contrasts passing the registered screen. After same-actor displacement-bin matching, rho={pm.Real_SceneMeanRho:.6f} on {int(pm.CommonActors)} actors; real−NC1={pm.DeltaRealMinusNC1:.6f} [95% descriptiveCI {pm.DeltaNC1_CI95Low:.6f},{pm.DeltaNC1_CI95High:.6f}], real−NC2={pm.DeltaRealMinusNC2:.6f} [{pm.DeltaNC2_CI95Low:.6f},{pm.DeltaNC2_CI95High:.6f}]. This supports a development-stage semantic screen, not a rule to force all pedestrians onto walkways.

Vehicle lane distance and −drivable occupancy correlate with six-mode errors, but their C-minus-oracle signed differences are positive in only onefold each, so they remain **WEAK** under the predeclared rule. Vehicle heading is **NONE**. Pedestrian crossing-distance/intersection hypotheses are also **NONE** and often show the opposite orientation. The primary walkway relation is nearzero in the important Pedestrian5–8m subgroup; the pooled Pedestrian result cannot be extrapolated to these moving targets. No new threshold or semantic reranker was derived from these findings.

## Frozen history and source identity

Base commit: `{BASE}`. Branch: `stage12a/candidate-semantic-consistency-audit`. All new files reside in `outputs/stage12a_candidate_semantic_audit/`. The source arrays and20 checkpoint hashes are recorded in `00_manifest/stage12a_frozen_manifest.json`; final verification rehashed {verification['HistoryFilesVerified']} historical tracked files and {verification['PreservedUntrackedVerified']} preserved Stage2C files.

Historical conclusions remain unchanged: Stage8 AgentGraph=SUPPORTED, SemanticGraph=NOT_SUPPORTED, SemanticContributionToJoint=NOT_SUPPORTED; Stage9 TAFIG=NOT_SUPPORTED; Stage10 DualExpert=STOP; Stage11A LossRankingMismatch=CONFIRMED, CostlyModeSwitchProblem=CONFIRMED, CandidateGeometryBottleneck=PARTIAL; Stage11B ErrorAwareRanking=STRONG_SUPPORTED, VehicleImproved=YES, PedestrianImproved=YES, BicyclePreserved=YES; Stage11C CalibrationEngineering=PASS, CalibrationUseful=YES, Top1Identity=PASS, Stage11C=GO.

Frozen Stage5 predictor SHA256: `88fe3feb59b7e830ec1e40aa917484386e0d2799d0f6116f410adda2d97fdce7`. K=6, Th=5, Tf=12. Population:630 scenes,260151 actor-window targets, including191026 Vehicle,66145 Pedestrian and2980 Bicycle. Three outer folds retain210 scenes each; no resplit. All1,560,906 geometry hashes and per-mode FP32 FDE/ADE were independently checked; maxdiff FDE={verification['FDEMaxDiff']}, ADE={verification['ADEMaxDiff']}. Positive-temperature C Calibrated has exactly the C Raw Top1 for all actors. Probability changes from calibration are auxiliary and provide no new trajectory improvement.

Frozen ranking metrics, pooled:

{table(pd.DataFrame(metrics).query('Scope == "Pooled" and Group in ["Overall", "Vehicle", "Pedestrian", "Bicycle", "MovingVehicle", "Pedestrian5-8m"]'),['Group','Model','Count','Top1FDE','Top1ADE','minFDE6','OracleGap','MeanTop1Confidence'])}

## Map integrity, coordinate audit and GT isolation

All engineering gates PASS. Reused the original Stage8A-0C SparseSemanticIndex, exact six static entities, 3/3/1/1 Vehicle quotas, 0/0/1/0/2/3 Pedestrian quotas and3/3/1/0 Bicycle quotas, 10m centerlines and2m polygons. No nearest fallback, buffer, geometry repair or GT selection was added. Four original nuScenes expansion maps retain the frozen Stage8 hashes; the local source directory name contains `nuscenes-mini` but includes all four full-region expansion maps. The existing single invalid Boston walkway component is excluded by unchanged Stage8 policy, retaining other valid components of that entity.

The 114 fixed preflight actors/684 candidates, selected without error labels, reproduce source entity IDs/types and raw FP32 map_node/map_edge bitwise. Ego x-forward/y-left, positive yaw counterclockwise; each sample uses its original t0 origin and yaw. Roundtrip maximum={coord['RoundtripMaxDiffM']:.6g} m <1e-5m. Polygon strict inside, polyline intersection, original-component boundary distance/crossing, true segment-interior distance and multipart OR tests pass. Poisoning GT toNaN, future/target masks and future motion labels changes neither entity selection nor semantic features.

Source caches include opaque future labels; the extractor uses an explicit observation/prediction allowlist and never uses those fields to choose entities or features. Standalone future-label array reads are blocked during extraction. Label access starts only after `FROZEN_ALL_630`. Three fork CPU workers write disjoint per-scene caches; three sequential-cache replay scenes pass, and spatial-index optimization maximum replay difference={cache['MaxOptimizationReplayDiff']:.6g}. Spatial pruning changes no definition. Stationary predicted headings have an explicit missing mask; original Stage8 degenerate-line warnings do not change finite cached values. Case visual clipping affects display size only.

## Feature definitions and completeness

Every mode stores scene/sample/actor/type/fold/mode, original geometry SHA256, traceable selected entity IDs and feature-valid bitmask. The unchanged source entity dictionary maps IDs to type, token and map region. The local full CSV `{stat['LocalFullModeCSV']}` contains1,560,906 rows; SHA256 `{stat['LocalFullModeCSVSHA256']}`. Its small versioned summary is `03_feature_statistics/stage12a_mode_semantics.csv`. Large arrays/CSV remain local and ignored by Git.

The nearest selected lane/connector is chosen by whole-polyline minimum distance; its cached mean12-point distance, endpoint distance and acos clipped heading cosine are retained. These are distances to true centerline polylines, not just start points. Candidate map matching uses12 predicted future points, without adding the observed-t0-to-first-future segment. Polygon distance is minimum over selected entities; strict sampled-point containment is OR over original valid components, not polygon union. Drivable boundary crossing and distance use original component boundaries. These details can differ from containment in a repaired/unioned polygon and are frozen here.

All26 semantic fields and24 observable candidate/history/interaction fields are finite. Missing features are zero with an explicit validity mask, and all empty-map actors remain. Centerline headings use the frozen final1s secant; undefined stationary headings are masked. History groups use only observed speed, net displacement and heading. Future displacement groups are offline labels only. SemanticCoverage=PASS means complete finite/masked extraction and retention; it does not claim100% map coverage.

Candidate endpoints/current positions are in the original t0 ego frame. `trajectory_curvature` is explicitly a turn-magnitude proxy: sum of absolute wrapped heading increments over consecutive nonzero candidate steps, in radians, not inverse-meter differential curvature. Frozen interaction summaries retain Stage8 raw physical units (distances in meters and closing distance/6s) and mask unavailable neighbors; they are not recomputed from future GT.

{table(cov[cov.Group.isin(ACTOR_TYPES)],['Scope','Group','Actors','Candidates','CandidateMapCoverage','ActorAnyModeCoverage','ActorAllModesCoverage','RouteCenterlineCoverage','PedestrianSpecificCoverage','EmptyCandidates'])}

Stage8 comparison uses the identical selector but a different population (TRAIN700 includes historical HeadDev70; Stage12 analyzes only HeadTrain630). This is a comparison of existing audit summaries, not a new HeadDev evaluation:

{table(comparison,['Group','Stage8Actors','Stage12Actors','Stage8CandidateCoverage','Stage12CandidateCoverage','DifferenceDueToPopulation'])}

## Prespecified semantic associations and negative controls

Within each actor, signed feature vs frozen six-mode FDE is tied-rank Spearman over at least3 valid varying modes. Undefined correlations remain marked ineligible; all actors/modes stay in coverage and identity outputs. Primary signs encode a screening hypothesis, not a rule that off-lane vehicles or road-entering pedestrians are wrong. Means average eligible actors within each scene, then scenes equally. Continuous feature/error tables also include actor-level mean,median,P90/P95/P99; association tables include scene median/P90/P95.

NC1 permutes entire semantic+mask records among each actor's six modes. NC2 permutes candidate semantic+mask records within exactly the same static map region and observed actor type. Independent fixed seed2022 generators preserve complete record distributions. Candidates, FDE, ADE, logits, probabilities and mode choices never change. All comparisons use the intersection of eligible actors across REAL/NC1/NC2, with explicit denominators. NegativeControl=PASS certifies this construction, not the existence of a semantic effect.

Scene bootstrap resamples whole scenes within each frozen210-scene fold,2000 repetitions, seed2022; the exact historical weight matrix was independently regenerated. The same weights and eligible actor intersection are used for real-minus-control paired contrasts. No independent-actor bootstrap is used.

{table(controls,['Scope','ActorType','Feature','CommonActors','Scenes','Real_SceneMeanRho','NC1_SceneMeanRho','NC2_SceneMeanRho','DeltaRealMinusNC1','DeltaNC1_CI95Low','DeltaNC1_CI95High','DeltaRealMinusNC2','DeltaNC2_CI95Low','DeltaNC2_CI95High','CMinusOracleSignedSceneMean'])}

Registered PROMISING requires pooled signedrho>=.05,500 eligible actors/50scenes, positive direction in at least2 adequately sized folds, real-minus-each-control>=.03 with descriptive95%CI lower>0, and positive C-minus-oracle signed feature difference in at least2folds. WEAK retains the rho/reproducibility requirements but fails remaining controls/selection checks. NONE includes no registered signal or insufficient estimability; absent estimates are shown as not estimable, never filled as evidence of zero.

{table(criteria)}

## Vehicle error modes, motion groups and high-cost switches

Differences below are raw feature(C Top1)−feature(oracle), among wrong-mode actors with both feature masks valid. Positive distance/heading and negative occupancy differences can be compatible with a map-consistency problem, but reasonable stopping, parking, lane changes and turns can also deviate from lane centerlines.

{table(vc,errorcols)}

MovingVehicle is reported separately from Stopped/ParkedVehicle and Vehicle>5m; it is not averaged away by stationary targets. The motion state/GT displacement labels are used only for this offline stratification. All fourfold scopes and feature quantiles are in the source CSVs.

Focused raw-real subgroup correlations below use each subgroup's real-only eligible actor set. They are exploratory, have no subgroup control-adjusted decision gate, and are not directly comparable to the primary three-way common eligibility denominator:

{table(motionfocus,['Group','Feature','Count','Scenes','SceneMean','SceneMedian','SceneP90','SceneP95'])}

## Pedestrian error modes and observable history groups

Pedestrian<5m,>5m and5–8m are offline future-displacement groups; those labels never define an inference feature. Drivable entry can be valid crossing behavior, and a nearby crossing does not establish intent. Feature validity/variation and scene denominators are especially important for small displacement subgroups.

{table(pc,errorcols)}

Recent speed bins [0,.5,1.5,3,inf), history net-displacement bins [0,.5,2,5,inf), and eight observed heading sectors plus missing are evaluated separately in feature/error, switch and association tables. These use no future trajectory. Pooled observable subgroup associations:

{table(motion[(motion.Scope=='Pooled')&motion.Group.str.contains('Pedestrian_')],['Group','Feature','Count','Scenes','SceneMean','SceneMedian','SceneP90','SceneP95'])}

## FoldR2 to C Raw switches

Improved/Worsened/Unchanged use the exact frozen ΔFDE sign; high-cost harm is the registered fixed ΔFDE>5m. Counts below retain unchanged and empty-map actors. Gross gain and harm refer to all improvements/worsenings in the named population; repeated outcome rows share those totals. High-cost examples are descriptive development cases.

{table(switch[(switch.Scope=='Pooled')&switch.Group.isin(['Vehicle','MovingVehicle','Pedestrian','Pedestrian5-8m','Bicycle'])],['Group','Outcome','Count','ModeSwitches','TotalModeSwitches','Mean','Median','P90','P95','GrossGain','GrossHarm','HarmP90','HarmP95','HarmP99'])}

Paired semantic changes for high-cost and ordinary harms, compared identically with improvements:

{table(switchfeatures[(switchfeatures.Scope=='Pooled')&switchfeatures.Group.isin(['Vehicle','MovingVehicle','Pedestrian','Pedestrian5-8m'])&switchfeatures.Outcome.isin(['Improved','Worsened','HighCostWorsened'])&switchfeatures.Feature.isin(mainfields)],['Group','Outcome','Feature','OutcomeActors','Count','Mean','Median','P90','P95','MeanFDEDeltaPaired'])}

## Semantic versus motion/interaction information

Matching is within the same actor, hence type,region and observed history are identical, and within fixed predicted-displacement bins [0,1,5,10,20,inf). At least3 valid varying modes are needed in a bin. The actor's eligible-bin correlations are averaged before equal-scene aggregation. This is a coarse no-training diagnostic, not full adjustment for endpoint,heading,curvature or interactions. Matching can substantially reduce estimable coverage; its denominators are reported.

{table(matched,['Scope','ActorType','Feature','CommonActors','Scenes','Real_SceneMeanRho','NC1_SceneMeanRho','NC2_SceneMeanRho','DeltaRealMinusNC1','DeltaNC1_CI95Low','DeltaNC1_CI95High','DeltaRealMinusNC2','DeltaNC2_CI95Low','DeltaNC2_CI95High'])}

Registered SemanticIncrementalValue={nc['SemanticIncrementalValue']}; this label concerns the semantic screening relation after coarse matching. Independent true prediction gain is **UNRESOLVED**. Semantic correlations with frozen endpoint, displacement,heading,length,curvature and interaction summaries show potential redundancy; they do not isolate map information causally. Heading correlations use wrapped angle ranks, a descriptive limitation.

`07_incremental_information/stage12a_region_displacement_associations.csv` additionally reports each type/map-region/predicted-displacement-bin combination in all3folds and pooled, retaining explicit unestimable rows; the region order is the frozen sorted dictionary order. This exploratory table is not used to change the six-feature primary screen.

{table(corr[corr.Scope=='Pooled'],['ActorType','SemanticFeature','ObservableControl','Count','Scenes','SceneMean'])}

## Bicycle preservation

All2980 Bicycle targets and17880 candidates participate in coverage, feature distributions and switch diagnostics. No semantic reranking is performed. C Raw logits/probabilities are bitwise FoldR2 for Bicycle; C Calibrated is the existing passthrough, so every Bicycle Top1 remains unchanged. No Vehicle/Pedestrian signal is extrapolated to Bicycle.

## Thirty registered BEV cases

Exactly10 Vehicle errors,10 Pedestrian errors,5 Vehicle successes and5 Pedestrian successes. Errors rank descending C oracle-gap and successes descending FDE gain, actor_id ties; one actor per scene is preferred. Map coverage/semantic values never filter or order cases. Each PNG/SVG/PDF overlays original HD Map,GT,all6 candidates,R2/C/oracle in the same t0 ego axes, includes scene+actor+type+ΔFDE and six-mode feature values with masks. GT and oracle are offline figure references only. Full traceable IDs and original entity tokens are available through case JSON and frozen dictionary.

{table(cases,casecols)}

## Reproduction and stop

Use the existing `ped_intent` environment and `PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1`; no package/environment upgrades. Ordered commands are recorded in `00_manifest/stage12a_run_commands.txt`. Registration, preflight, complete semantic freeze, statistics, negative controls, cases, independent verification and report run in that order. Feature extraction supports per-scene resume with SHA256 and source code hashes. Numerical failures abort instead of creating substitute data.

All required small tables,protocols,verification,figures and reports are versioned on the Stage12A branch; raw arrays/full actor CSV remain local. No merge is performed. Final action: **STOP** and wait for 大脑AI review, even when ReadyForStage12B=YES. No Stage12B training, semantic GNN, official VAL/test, HiVT retraining or loss-parameter search was executed.
'''
    (ROOT/'11_reports/stage12a_final_report.md').write_text(report)
    print('STAGE12A_SCIENTIFIC_DECISION',json.dumps(decision,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
