"""Write evidence-grounded report and freeze verification after the complete audit."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'00_manifest'))
from stage8a0b_common import *

def rows(name):return list(csv.DictReader((ROOT/'06_tables'/f'stage8a0b_{name}.csv').open()))
def table(rs,keys,percent=()):
    output=['| '+' | '.join(keys)+' |','| '+' | '.join(['---']*len(keys))+' |']
    for r in rs:
        values=[]
        for key in keys:
            value=r[key]
            if value is None or value=='':value='NA'
            elif key in percent:value=f'{float(value)*100:.4f}%'
            values.append(str(value))
        output.append('| '+' | '.join(values)+' |')
    return '\n'.join(output)

def main():
    decision=read_json(ROOT/'00_manifest/stage8a0b_decision.json')
    integrity=read_json(ROOT/'04_coverage/stage8a0b_integrity_audit.json')
    schema=read_json(ROOT/'01_schema_audit/stage8a0b_schema_audit.json')
    assert schema['raw_layer_polygon_geometry_audit']=='PASS'
    geom=read_json(ROOT/'03_entity_retrieval/stage8a0b_geometry_audit.json');assert geom['status']=='PASS'
    coverage=rows('coverage');fallback=rows('vehicle_fallback_decomposition');ped=rows('pedestrian_semantic_coverage');bike=rows('bicycle_semantic_coverage')
    def full(rs):return [r for r in rs if r['Population']=='full_horizon_ranking_targets' and r['Split']!='combined']
    f=full(fallback);cov=full(coverage);p=full(ped);bi=full(bike);g=full(rows('graph_estimate'))
    full_combined=[r for r in coverage if r['Population']=='full_horizon_ranking_targets' and r['Split']=='combined']
    overall=next(r for r in full_combined if r['ActorGroup']=='Overall')
    sections=[]
    def add(title,text):sections.append(f'## 【{title}】\n\n{text}\n')
    add('Why Stage8A-0 Failed','Stage8A-0 stays **NOT_READY**, with lane-only fallback 25.170734% TRAIN / 26.539193% VAL in current-valid Vehicle modes. This audit introduces a separate actor-type coverage definition and does not rewrite that result. Predictions are the frozen Stage5A candidates; R2 weights and historical rankings are untouched. All six modes have equal diagnostic weight.\n\n'+
        table([r for r in rows('fallback_contributions') if r['Split']!='combined'],['Split','Population','ActorGroup','Candidates','LaneFallbackCandidates','ShareOfVehicleLaneFallback'],['ShareOfVehicleLaneFallback']))
    add('Lane-only Fallback Decomposition','Complete centerlines, fixed 10 m and top8, with exact token/distance reproduction of the historical nearest-one rule. Every coverage count is **before** nearest fallback. All four observed vehicle motion states are reported. Missing groups have undefined rates, never implicit success. The complete tables include current-valid and full-horizon populations.\n\n'+
        table(f,['Split','MotionState','Actors','Candidates','LaneOnlyFallbackRate','TypeAwareFallbackRate','RecoveredByDrivableRate','RecoveredByCarparkRate'],['LaneOnlyFallbackRate','TypeAwareFallbackRate','RecoveredByDrivableRate','RecoveredByCarparkRate'])+
        '\n\nRecovery denominators are lane-fallback candidates, with possible drivable/carpark overlap. The CSV also includes union, intersection and unconditional recovery. Candidate displacement is last minus first of the twelve predicted points, with bins [0,1], (1,5], >5 m; it never uses GT. Actor-level all6/any1/all6-failure counts and rates are in `stage8a0b_actor_fallback.csv`.\n\n'+
        table([r for r in full(rows('candidate_displacement')) if r['ActorGroup']=='Vehicle'],['Split','PredictedNetDisplacementBin','BinCandidates','LaneOnlyFallbackRate','TypeAwareFallbackRate'],['LaneOnlyFallbackRate','TypeAwareFallbackRate']))
    add('Map Schema','All four actual local raw JSON maps and the installed NuScenesMap API were read. All seven requested layers exist. Lane and connector retrieval geometry is the frozen complete LineString; area layers use actual polygon references, including plural `polygon_tokens` for multipart drivable-area records. Stop_line is audited and matched diagnostically but excluded from every primary graph selector.\n\n'+
        table(rows('map_schema'),['MapRegion','Layer','Records','ValidGeometry','InvalidGeometry','UsableGeometry','InvalidReference']))
    add('Polygon Geometry Integrity',f"Finite source coordinates and closed API polygon rings were verified. Invalid components: {schema['invalid_components']}; invalid audited references: {schema['invalid_references']}; partially usable multipart records: {schema['partial_entities']}. Boston's one invalid walkway polygon is excluded exactly as preregistered. No buffer, topology repair, invented association or new label was used. Multipart tokens are deduplicated without unioning their geometry.\n\nA polygon matches if a predicted future point lies inside, the future polyline intersects, or its true boundary distance is <=2 m. Interior intersection yields filled-geometry distance zero even when the boundary is far away. Sorting uses filled geometry distance, then type ID/token. The independent brute-force check evaluated 12 candidates across all four maps against every usable entity, and reproduced counts and top8 IDs/distances. Synthetic geometry checks cover segment interiors, crossing with both endpoints outside, interior far from boundary, and the exact fixed 2 m inclusion boundary.")
    add('Vehicle Type-Aware Context','Vehicle relevance is lane/connector/drivable/carpark. Coverage remains uncapped and before any fallback. Full-horizon TRAIN and VAL gates are applied independently.\n\n'+table([r for r in cov if r['ActorGroup']=='Vehicle'],['Split','Actors','RouteCenterlineCoverage','AreaContextCoverage','CarparkCoverage','TypeAwareAnyCoverage','GenericOnlyRate'],['RouteCenterlineCoverage','AreaContextCoverage','CarparkCoverage','TypeAwareAnyCoverage','GenericOnlyRate']))
    for title,name in [('Moving Vehicle','MovingVehicle'),('Stopped Vehicle','StoppedVehicle'),('Parked Vehicle','ParkedVehicle')]:
        add(title,table([r for r in cov if r['ActorGroup']==name],['Split','Actors','RouteCenterlineCoverage','AreaContextCoverage','TypeAwareAnyCoverage','TypeAwareFallbackRate'],['RouteCenterlineCoverage','AreaContextCoverage','TypeAwareAnyCoverage','TypeAwareFallbackRate'])+
            ('\n\nMoving vehicles retain the independent >=95% route-centerline gate; area coverage cannot replace it.' if name=='MovingVehicle' else '\n\nThe >=95% any-context gate uses only the four authorized vehicle entity types.'))
    add('Pedestrian Semantic Context','Primary: walkway, ped_crossing or drivable_area. Lane/connector are secondary and cannot rescue pedestrian primary coverage. SemanticSpecific is walkway OR crossing; it has no independent 95% gate. Primary-only groups and generic-only all-six-entity groups have different definitions.\n\n'+
        table(p,['Split','Candidates','WalkwayCoverage','CrosswalkCoverage','DrivableCoverage','SemanticSpecificCoverage','TypeAwareAnyCoverage','GenericOnlyRate'],['WalkwayCoverage','CrosswalkCoverage','DrivableCoverage','SemanticSpecificCoverage','TypeAwareAnyCoverage','GenericOnlyRate'])+
        '\n\n'+table(p,['Split','WalkwayOnlyRate','CrosswalkOnlyRate','DrivableOnlyPrimaryRate','MultiplePrimaryRate','NoPrimaryContextRate'],['WalkwayOnlyRate','CrosswalkOnlyRate','DrivableOnlyPrimaryRate','MultiplePrimaryRate','NoPrimaryContextRate']))
    add('Bicycle Semantic Context','Primary: lane, connector or drivable_area. Carpark coverage is descriptive only.\n\n'+table(bi,['Split','Candidates','LaneCoverage','ConnectorCoverage','DrivableCoverage','TypeAwareAnyCoverage','GenericOnlyRate'],['LaneCoverage','ConnectorCoverage','DrivableCoverage','TypeAwareAnyCoverage','GenericOnlyRate']))
    add('Generic Drivable Dominance','GenericOnly requires drivable_area and **no** lane, connector, carpark, crossing or walkway, including secondary context. The denominator is all modes in the stated population. Conditional-on-covered rates are also available in the coverage CSV. The preregistered full-horizon limits are Vehicle <=30%, Pedestrian <=50%, separately in each split. This narrow engineering classification does not demonstrate information gain, forecast accuracy or semantic usefulness.\n\n'+
        table([r for r in cov if r['ActorGroup'] in ['Vehicle','Pedestrian','Bicycle']],['Split','ActorGroup','GenericOnlyRate','GenericOnlyAmongCoveredRate'],['GenericOnlyRate','GenericOnlyAmongCoveredRate']))
    add('Top8 Capacity','All-entity overflow includes all six authorized types; relevant overflow restricts to actor-type primary context. Neither changes TopM=8. Selection is purely exact geometry distance, type ID, token; no GT, error or semantic priority. Uncapped counts and capped graph counts are reported separately.\n\n'+
        table([r for r in g if r['Group'] in ['Overall','Vehicle','Pedestrian','Bicycle']],['Split','Group','UncappedMeanRelevantEntitiesPerMode','MeanMapEntitiesPerMode','OverflowRateTop8','RelevantOverflowRateTop8'],['OverflowRateTop8','RelevantOverflowRateTop8'])+
        '\n\nThe entity statistics CSV gives each type’s mean/median/p95/max and matched-candidate contribution. Stop-line counts are separated and never consume capacity.\n\n'+
        table([r for r in full(rows('entity_statistics')) if r['ActorGroup']=='Overall' and r['EntityType'] in TYPES[:6]],
            ['Split','EntityType','Coverage','MeanCount','MedianCount','P95Count','MaxCount'],['Coverage']))
    add('Graph Size','This is a resource estimate, with no formal new G2/G3 implementation. Six modes each have <=8 relevant map entities, allowing genuinely empty selectors. Thus there are <=48 mode-map edges per target. Future map-node schema is only a proposal: six-type onehot, unchanged nine lane attributes (zero for non-lanes), relative geometry and polygon/centerline attributes. The byte estimate budgets 64 float32 values per map-node record and 64 per map-edge record without defining those as model features.\n\n'+
        table([r for r in g if r['Group']=='Overall'],['Split','Targets','MeanModeMapEdges','P95ModeMapEdges','MaxModeMapEdges','EstimatedBytesPerTarget','Batch128DenseMaxInputBytes','EstimatedPeakCUDAMiB'])+
        f"\n\nConservative inference peak budget: {decision['inference_memory_budget_MiB']:.3f} MiB, derived from fourfold the historical measured non-input peak plus expanded dense inputs and 32 MiB margin. Device capacity: {decision['GPU_total_bytes']/2**30:.3f} GiB. This estimate is unmeasured for any future model; training activation/optimizer memory and future latency are unknown. No training feasibility or predictive performance claim follows.")
    add('No-Future-Leakage','ObservableWindow is an explicit allowlist of history, actor type, frozen candidate/logits/probabilities, identities and observed coordinate frames. Worker jobs carry only predicted candidate geometry, actor type and map location. GT/future/target access is guarded. In the same seed2022 100 TRAIN +100 VAL windows, GT becomes NaN and both masks are inverted. Entity IDs, distances, counts, coverage and relevant/all top8 selectors remain bitwise identical. Offline masks only determine reporting populations after retrieval. Candidate identity inherits the exact frozen predictor replay and revalidates every source cache SHA; predictions are aliased with unchanged data pointers, and all historical lane token IDs/distances are reproduced bitwise. No predictor forward, optimizer, backward, checkpoint, tiny run or test data was used.')
    add('Coordinate Integrity',f"PASS on {integrity['windows']} windows and {integrity['polygons_checked']} polygon checks. Candidate roundtrip max {integrity['candidate_roundtrip_max_m']:.12g} m; polygon roundtrip max {integrity['polygon_roundtrip_max_m']:.12g} m; true geometry distance discrepancy max {integrity['geometry_distance_max_m']:.12g} m; boundary distance discrepancy max {integrity['boundary_distance_max_m']:.12g} m. All are below 1e-5 m. Both actual matched multipart polygons and random source polygons were transformed. Native shared-geometry threading failed during an early attempt; the final complete run used independent GEOS processes and discarded partial results.")
    add('Scientific Decision',table(decision['coverage_gates'],['Split','Group','Metric','Value','Threshold','PASS'])+
        f"\n\nLaneOnlyFailureCause = **{decision['LaneOnlyFailureCause']}**\n\nTypeAwareSemanticCoverage = **{decision['TypeAwareSemanticCoverage']}**\n\nSemanticContextQuality = **{decision['SemanticContextQuality']}**\n\nStage8A_0B = **{decision['Stage8A_0B']}**\n\nThe failure-cause label follows the preregistered descriptive evidence rule. Low-motion share evidence: {decision['low_motion_evidence']}; residual authorized map-coverage limitation: {decision['map_coverage_limitation_evidence']}. Map limitation here includes geometry/threshold coverage gaps and does not assert that a map layer is missing. No causal training conclusion is inferred. No failed gate is rescued by pooling TRAIN/VAL or adding layers, radius or capacity. Historical Stage8A-0 remains NOT_READY. **STOP. No G1/G2/G3 training. Await scientific review.**")
    (ROOT/'09_reports/stage8a0b_type_aware_semantic_entity_audit.md').write_text('# Stage8A-0B actor-type-aware semantic entity coverage audit\n\n'+ '\n'.join(sections))
    frozen=read_json(ROOT/'00_manifest/stage8a0b_frozen_inputs.json')
    for key in ['files','unrelated_untracked']:
        for name,h in frozen[key].items():assert sha256(PROJECT/name)==h,name
    manifest=read_json(ROOT/'03_entity_retrieval/stage8a0b_retrieval_manifest.json')
    for b in manifest['batches']:assert sha256(ROOT/b['path'])==b['sha256']
    assert git('branch','--show-current')==BRANCH
    assert not list(ROOT.rglob('*.pt'))
    assert not git('diff','--name-only')
    atomic_json(ROOT/'00_manifest/stage8a0b_final_integrity.json',{'status':'PASS','historical_files_checked':len(frozen['files']),
        'unrelated_files_unchanged':len(frozen['unrelated_untracked']),'retrieval_batches_checked':len(manifest['batches']),
        'source_batches_SHA_verified':manifest['source_sha_verified'],'cache_contains_GT':False,'new_checkpoints':0,
        'optimizer_steps':0,'backward_calls':0,'test_used':False,'historical_Stage8A_0':'NOT_READY',
        'preregistration_unchanged':sha256(ROOT/'00_manifest/stage8a0b_execution_plan.md')==frozen['preregistration_sha256'],
        'decision':decision['Stage8A_0B'],'STOP':True})
    print('FINAL INTEGRITY PASS',decision['Stage8A_0B'],flush=True)

if __name__=='__main__':main()
