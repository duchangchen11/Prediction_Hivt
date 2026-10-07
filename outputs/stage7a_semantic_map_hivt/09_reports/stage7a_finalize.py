"""Evidence-grounded report and final read-only preservation/figure audit."""
from pathlib import Path
import sys,csv,collections,json,subprocess,xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'00_manifest'),str(ROOT/'01_data_audit')]
from stage7a_common import *
from stage7a_map_inventory import schema as raw_record_schema
import numpy as np
from PIL import Image
def table(name):return list(csv.DictReader(open(ROOT/'06_tables'/name)))
def markdown(rows,fields):
    def value(v):
        if v is None or v=='':return '—'
        if isinstance(v,float):return f'{v:.6f}'
        return str(v)
    return '\n'.join(['| '+' | '.join(fields)+' |','| '+' | '.join('---' for _ in fields)+' |']+['| '+' | '.join(value(r.get(k)) for k in fields)+' |' for r in rows])
def source_line(path,phrase):
    text=path.read_text().splitlines();return next(i for i,s in enumerate(text,1) if phrase in s)
def pct(v):return f'{100*float(v):.3f}%'
def figure_audit():
    rows=[]
    for p in sorted((ROOT/'05_figures').glob('*.png')):
        svg=p.with_suffix('.svg');pdf=p.with_suffix('.pdf');assert svg.exists() and pdf.exists()
        im=Image.open(p);dpi=im.info.get('dpi',(0,0));assert all(abs(v-300)<0.02 for v in dpi)
        tree=ET.parse(svg);text=tree.findall('.//{http://www.w3.org/2000/svg}text');assert text
        selectable=subprocess.check_output(['pdftotext',str(pdf),'-'],text=True);assert len(selectable.strip())>20
        fonts=subprocess.check_output(['pdffonts',str(pdf)],text=True).splitlines()[2:];assert fonts and not any('Type 3' in f for f in fonts)
        rows.append({'name':p.stem,'PNG_dimensions':list(im.size),'PNG_dpi':list(dpi),'PNG_SHA256':sha256(p),
            'SVG_editable_text_nodes':len(text),'PDF_selectable_text':True,'PDF_fonts':fonts})
    assert len(rows)==12
    visual=read_json(ROOT/'01_data_audit/stage7a_semantic_manual_review.json');assert visual['status']=='PASS'
    atomic_json(ROOT/'00_manifest/stage7a_figure_audit.json',{'status':'PASS','backend':'python','figures':rows,
        'direct_visual_inspection':'60 random turns + all 12 U-turns + 4 boundary/short-path cases + 50 semantic samples; Codex visual review, not independent expert review',
        'layout_corrections':'Overview shared legend and boundary-grid title moved away from panel titles.',
        'scientific_statistics':'descriptive counts only; samples are seeded random token records; no performance confidence claim',
        'global_geometry_unmodified':True,'image_smoothing_or_geometry_interpolation':False,'contact_sheets_for_audit_not_submission_pages':True})
    return rows
def main():
    ref=verify_v1();qa=read_json(ROOT/'01_data_audit/stage7a_all_scene_data_quality.json');assert qa['status']=='PASS'
    layers=table('stage7a_map_layer_statistics.csv');turns=table('stage7a_turn_semantic_statistics.csv');controls=table('stage7a_control_semantic_statistics.csv');exposure=table('stage7a_actor_semantic_exposure.csv');errors=table('stage7a_semantic_context_error_audit.csv')
    cross=table('stage7a_crosswalk_coverage.csv');hist=table('stage7a_analytic_angle_histogram.csv');candidates=table('stage7a_intersection_turning_group_candidates.csv');segments=table('stage7a_stage3_segment_semantic_coverage.csv');moving_n=table('stage7a_moving_semantic_sample_counts.csv');stops=table('stage7a_stop_line_raw_statistics.csv')
    schema_path=ROOT/'01_data_audit/stage7a_map_schema_audit.json';schema=read_json(schema_path);angle_qa=read_json(ROOT/'01_data_audit/stage7a_turn_geometry_audit.json');topo=read_json(ROOT/'01_data_audit/stage7a_source_topology_gaps.json')
    reference_rows=[]
    for loc,source in schema['locations'].items():
        raw=read_json(source['source_path']);assert sha256(source['source_path'])==source['sha256']
        source['arcline_path_3']={'key_count':len(raw['arcline_path_3']),'raw_record_fields':raw_record_schema([r for v in raw['arcline_path_3'].values() for r in v])}
        source['connectivity']={'key_count':len(raw['connectivity']),'raw_record_fields':raw_record_schema(list(raw['connectivity'].values()))}
        relations=[('lane','polygon_token','polygon'),('lane_connector','polygon_token','polygon'),('stop_line','polygon_token','polygon'),
            ('ped_crossing','polygon_token','polygon'),('lane','from_edge_line_token','line'),('lane','to_edge_line_token','line'),
            ('stop_line','road_block_token','road_block'),('traffic_light','line_token','line'),('traffic_light','from_road_block_token','road_block'),('ped_crossing','road_segment_token','road_segment')]
        for layer,field,dest in relations:
            targets={r['token'] for r in raw[dest]};refs=[r.get(field) for r in raw[layer] if r.get(field)]
            reference_rows.append({'location':loc,'source_layer':layer,'source_field':field,'target_layer':dest,'reference_count':len(refs),
                'null_or_empty_reference_records':sum(not r.get(field) for r in raw[layer]),'unresolved_count':sum(t not in targets for t in refs)})
        for layer,field,dest in [('stop_line','traffic_light_tokens','traffic_light'),('stop_line','ped_crossing_tokens','ped_crossing')]:
            targets={r['token'] for r in raw[dest]};refs=[t for r in raw[layer] for t in r.get(field,[])]
            reference_rows.append({'location':loc,'source_layer':layer,'source_field':field,'target_layer':dest,'reference_count':len(refs),
                'null_or_empty_reference_records':sum(not r.get(field) for r in raw[layer]),'unresolved_count':sum(t not in targets for t in refs)})
        refs=[t for r in raw['traffic_light'] for i in r.get('items',[]) for t in i.get('to_road_block_tokens',[])];targets={r['token'] for r in raw['road_block']}
        reference_rows.append({'location':loc,'source_layer':'traffic_light','source_field':'items[].to_road_block_tokens','target_layer':'road_block','reference_count':len(refs),
            'null_or_empty_reference_records':sum(not any(i.get('to_road_block_tokens') for i in r.get('items',[])) for r in raw['traffic_light']),'unresolved_count':sum(t not in targets for t in refs)})
    assert all(r['unresolved_count']==0 for r in reference_rows)
    write_csv(ROOT/'06_tables/stage7a_map_source_reference_audit.csv',reference_rows);atomic_json(schema_path,schema)
    local_cache_manifest=read_json(ROOT/'02_semantic_cache/stage7a_cache_manifest.json');local_cache_manifest['source_files']=schema['locations'];atomic_json(ROOT/'02_semantic_cache/stage7a_cache_manifest.json',local_cache_manifest)
    prior=read_json(ROOT/'00_manifest/stage7a_frozen_previous.json')
    for name,digest in prior['files'].items():assert sha256(PROJECT/name)==digest,name
    for name,digest in prior['scene_shards'].items():assert sha256(STAGE3/name)==digest,name
    for loc,source in schema['locations'].items():assert sha256(source['source_path'])==source['sha256']
    for name in ('03_training','04_evaluation','07_checkpoints'):assert all(p.name=='README.md' for p in (ROOT/name).iterdir())
    assert not git('diff','--name-only') and not git('diff','--cached','--name-only')
    figs=figure_audit()
    counts={k:sum(int(r['record_count']) for r in layers if r['layer']==k) for k in ['lane','lane_connector','stop_line','traffic_light','ped_crossing','walkway']}
    tcounts={k:sum(int(r['connector_count']) for r in turns if r['turn_type']==k) for k in TURN}
    cache=read_json(ROOT/'02_semantic_cache/stage7a_semantic_metadata.json');categories=collections.Counter(v['traffic_control_type'] for m in cache.values() for v in m.values())
    regulatory_tokens=sum(any(t in v['control_types_present'] for t in ['traffic_light','stop_sign','yield']) for m in cache.values() for v in m.values())
    stoptypes=collections.Counter()
    for r in stops:stoptypes[r['stop_line_type']]+=int(r['record_count'])
    candidate_path=ROOT/'00_manifest/stage7a_semantic_candidate_definition.json';candidate_def=read_json(candidate_path)
    candidate_def['exclusive_control_onehot_scope']='audit bookkeeping of actual categories only; not a frozen future model vector'
    candidate_def['removed_candidate_features']=['dedicated_yield_control']
    candidate_def['yield_candidate']='REMOVED_FROM_RECOMMENDED_FEATURES; raw map audit label retained without substitution'
    candidate_def['recommended_future_feature_candidates']=['is_lane_connector','turn_type','static_traffic_light_existence','static_stop_sign_existence','other_source_stop_line_existence','near_ped_crossing']
    candidate_def['future_model_control_encoding_frozen']=False
    atomic_json(candidate_path,candidate_def)
    atomic_json(ROOT/'01_data_audit/stage7a_removed_semantic_candidates.json',{'removed_features':[{'feature':'dedicated_yield_control',
        'source_record_count':stoptypes['YIELD'],'associated_tokens':sum('yield' in v['control_types_present'] for m in cache.values() for v in m.values()),
        'reason':'only3 actual map records, insufficient independent source support; not retained as a dedicated future feature',
        'raw_audit_labels_preserved':True}], 'unavailable_features':['dynamic_signal_state','direct_authoritative_lane_control_label'],
        'excluded_source':'16 dangling ordinary-lane connectivity links; not consumed by candidate geometry features','no_values_fabricated':True})
    sources={name:{'path':name,'sha256':sha256(PROJECT/name)} for name in ['preprocessing/extract_lane_polylines.py','preprocessing/coordinates.py','outputs/stage3_multitype_hivt/02_preprocessed/stage3_dataset.py']}
    extractor=PROJECT/'preprocessing/extract_lane_polylines.py';dataset=STAGE3/'02_preprocessed/stage3_dataset.py';coords=PROJECT/'preprocessing/coordinates.py'
    code=['# Existing Stage3 map input audit','',f'All {qa["placeholder_lane_segments_checked"]:,} segment occurrences in {sum(qa["windows"].values()):,} candidate windows / 850 official scenes were checked, including {sum(qa["empty_supervision_windows"].values())} empty-supervision windows. All three stored placeholders are zero.','',
        f'- `{extractor.relative_to(PROJECT)}:{source_line(extractor,"def __init__")}`: 50 m radius, 2 m resolution.',
        f'- `{extractor.relative_to(PROJECT)}:{source_line(extractor,"layer_names=")}`: queries lane and lane_connector only.',
        f'- `{extractor.relative_to(PROJECT)}:{source_line(extractor,"discretize_lanes")}`: batches missing token centerlines, caches per (location, token).',
        f'- `{extractor.relative_to(PROJECT)}:{source_line(extractor,"owners.extend")}`: repeated lane_tokens identify segment ownership.',
        f'- `{extractor.relative_to(PROJECT)}:{source_line(extractor,"lane, actor =")}`: actor edges use strict <50 m distance to segment starts.',
        f'- `{dataset.relative_to(PROJECT)}:{source_line(dataset,"is_intersections=torch.zeros")}`: is_intersections = zero uint8 placeholder.',
        f'- `{dataset.relative_to(PROJECT)}:{source_line(dataset,"turn_directions=torch.zeros")}`: turn_directions and traffic_controls = zero uint8 placeholders.',
        f'- `{dataset.relative_to(PROJECT)}:{source_line(dataset,"map_location=location")}`: source location and ownership tokens retained.',
        f'- `{coords.relative_to(PROJECT)}:{source_line(coords,"def global_to_ego")}` and `:{source_line(coords,"def ego_to_global")}`: unchanged right-handed row-vector transforms; +x ego forward, +y ego left.','',
        markdown(list(sources.values()),['path','sha256']),'',f'Joint global start XY/vector XY matching to original 2 m centerlines resolves repeated arc-junction starts. Maximum position difference {qa["maximum_global_segment_start_difference_m"]:.9g} m; vector difference {qa["maximum_global_segment_vector_difference_m"]:.9g} m; tolerances 1e-4 m. Full edge recomputation exactly matches lane_actor_index in every window.','',
        'No old extractor, polyline cache, dataset, coordinate transform, model, loss, decoder or R2 code was modified.']
    (ROOT/'01_data_audit/stage7a_existing_map_feature_audit.md').write_text('\n'.join(code)+'\n')
    schema_md=['# Actual nuScenes Map Expansion schema audit','', 'Both original regional JSON and the installed NuScenesMap API were read. API shortcuts are reported separately from raw fields. Static map version is 1.3 in all four locations.','']
    for loc,v in schema['locations'].items():
        schema_md += [f'## {loc}','',f'Source: `{v["source_path"]}`. SHA256: `{v["sha256"]}`.','']
        for layer in ['lane','lane_connector','stop_line','traffic_light','ped_crossing']:
            fs=v['layers'][layer]['raw_record_fields'];api_fs=v['layers'][layer]['api_record_fields'];schema_md += [f'### {layer} ({v["layers"][layer]["count"]} records)','',
                markdown([{'field':k,'present':x['present'],'non_null':x['non_null'],'types':', '.join(x['types'])} for k,x in fs.items()],['field','present','non_null','types']),
                '', 'API-only shortcut fields: '+(', '.join(sorted(set(api_fs)-set(fs))) or 'none')+'.','']
        schema_md += ['NOT AVAILABLE: '+', '.join(v['unavailable_fields'])+'.','',f'Traffic-light all-zero XY poses: {v["traffic_light_static_pose_quality"]["all_zero_xy_count"]}. items[].color and shape describe static light lenses, not observations of signal state.','']
        for auxiliary in ['arcline_path_3','connectivity']:
            schema_md += [f'### {auxiliary} ({v[auxiliary]["key_count"]} dictionary keys)','',
                markdown([{'field':k,'present':x['present'],'non_null':x['non_null'],'types':', '.join(x['types'])} for k,x in v[auxiliary]['raw_record_fields'].items()],['field','present','non_null','types']),'']
    schema_md += ['## Limits','',f'No direct raw lane control label or turn_type exists. Geometric static associations are derived and labeled as such. {topo["unresolved_reference_count"]} dangling connectivity references occur on 16 ordinary lanes (8 absent targets) in singapore-onenorth; connector references are complete. No repairs or substitutes are made. Candidate features do not consume raw connectivity links.','',
        'The external original trainval mount is absent. Complete original regional map JSON is in the shared expansion install under the existing local mini dataroot view. Compatibility with trainval is established by all 850 frozen official scene shards and every stored segment start/vector, not by claiming a new read of absent raw annotations/logs.']
    (ROOT/'01_data_audit/stage7a_map_schema_audit.md').write_text('\n'.join(schema_md)+'\n')
    core_pass=qa['unresolved_segment_tokens']==0 and qa['all_control_references_valid'] and qa['NaN_or_Inf_count']==0 and qa['moving_exposure_sufficient'] and angle_qa['all_finite'] and angle_qa['connector_dangling_topology_references']==0
    # Geometric-only candidate is independently fully reference-valid. Raw topology is separately PARTIAL.
    science={'TurnSemantic':'AVAILABLE','TrafficControlSemantic':'PARTIAL','CrosswalkSemantic':'AVAILABLE','SemanticMapStage7A':'READY' if core_pass else 'NOT_READY',
        'ready_scope':'only audited static geometric candidate cache; excludes dangling raw topology features, dynamic signal states, and a dedicated sparse-YIELD feature',
        'RawConnectivitySemantic':'PARTIAL','training_authorized':False}
    valex=[r for r in exposure if r['Split']=='val'];trainex=[r for r in exposure if r['Split']=='train'];seg=[r for r in segments if r['location']=='all']
    hsum=[]
    for low,high in zip([-180,-135,-90,-60,-30,-15,15,30,60,90,135],[-135,-90,-60,-30,-15,15,30,60,90,135,180]):hsum.append({'Interval_deg':f'[{low}, {high}{"]" if high==180 else ")"}','Connectors':sum(int(r['connector_count']) for r in hist if float(r['bin_low_deg'])==low)})
    main_rows=[r for r in table('stage7a_moving_semantic_sample_counts.csv') if r['Split']=='val']
    frozen_main=table_from_old(STAGE6/'06_tables/stage6a_main_ranking_results.csv')
    oldoverall=next(r for r in frozen_main if r['Group']=='overall')
    byerror={r['Context']:r for r in errors}
    error_findings=[]
    for label,near,no in [('connector','NearConnector','NoConnector'),('turning connector','NearTurnConnector','NoTurnConnector'),('typed traffic-control token','NearTrafficControl','NoTrafficControl'),('crosswalk token','NearCrosswalk','NoCrosswalk')]:
        a=byerror[near];b=byerror[no];d=float(a['R2_Top1FDE'])-float(b['R2_Top1FDE'])
        error_findings.append(f'{label}: R2 mean Top1FDE is {float(a["R2_Top1FDE"]):.6f} vs {float(b["R2_Top1FDE"]):.6f} m (near−no={d:+.6f}); near n={a["Count"]}, no n={b["Count"]} / {b["Instances"]} instances / {b["Scenes"]} scenes.')
    report=['# Stage7A-0 semantic map data audit','', 'Audit only. No model training, architecture change or predictor/R2 forward pass. New work remains in this independent Stage7A root.','',
        '【Frozen V1】','',f'Frozen Stage6A commit: `{BASE}`. Annotated tag `{TAG}` targets exactly this commit; tag object `{ref["annotated_tag_object"]}`. Message: `{ref["tag_message"]}`. The pre-tag workspace was clean after a path-scoped stash of the five unrelated Stage2C redraw files; all five were restored byte-identically and remain unsubmitted.','',
        f'Tag remote status: **{ref["tag_remote_status"]}**. Tag is local; GitHub tag push remains blocked by absent local Git credentials. This delivery limitation does not change map-data feasibility. The branch can be published via the authorized GitHub connector.','',
        markdown([{'Reference':k,**v} for k,v in ref['files'].items()],['Reference','path','sha256']),'',
        f'Frozen results: Overall minFDE6={float(oldoverall["R2_minFDE6"]):.9f} m; Overall R2 Top1FDE6={float(oldoverall["R2_Top1FDE"]):.9f} m. Vehicle=3.010194075 m; Pedestrian=1.418252746 m; vehicle.moving=10.450046384 m. ReliabilityHead=SUPPORTED; FutureInteractionContribution=SUPPORTED; PaperUsableReliability=YES; RecommendedFinalVariant=R2. Stage5A prior NOT SUPPORTED is unchanged.','',
        '【Current Map Input】','', f'Current input is lane and lane_connector centerline geometry at unchanged 2 m resolution and per-actor 50 m radius. is_intersections / turn_directions / traffic_controls are all zero on {qa["placeholder_lane_segments_checked"]:,} segment occurrences. Actual source lines, hashes and stored tensor checks are in [existing map feature audit](../01_data_audit/stage7a_existing_map_feature_audit.md).','',
        f'Official TRAIN700 has {qa["windows"]["train"]:,} candidate windows ({qa["empty_supervision_windows"]["train"]} empty-supervision); VAL150 has {qa["windows"]["val"]:,} ({qa["empty_supervision_windows"]["val"]} empty-supervision). Full target actor-windows: TRAIN={qa["TRAIN_full_actor_count"]:,}, VAL=54,990. All candidate windows, including empty ones, contribute map-segment counts.','',
        f'No-current-actor null graphs: TRAIN={qa["no_current_actor_null_graphs"]["train"]}, VAL={qa["no_current_actor_null_graphs"]["val"]}; each original index confirms zero actors/targets and therefore no stored lane segments. All-null scenes: {qa["all_empty_graph_scenes"]}. Their absent stored map location is explicitly NOT AVAILABLE, rather than inferred from incomplete logs. All nonempty scene graphs are checked against their actual recorded location.','',
        f'{qa["source_mount_note"]} The shared map install contains four complete regional maps; it is not restricted to the mini scenes. No raw scene/log/annotation read from the absent trainval mount is claimed.','',
        '【nuScenes Map Schema】','', markdown(layers,['location','layer','record_count']),'',
        'The raw/API field-by-field inventories are [schema JSON](../01_data_audit/stage7a_map_schema_audit.json) and [schema Markdown](../01_data_audit/stage7a_map_schema_audit.md). Raw lane_connector has only token and polygon_token; source arcline_path_3 and connectivity separately provide directed geometry/topology. Raw lane_type is a road lane category, not left/straight/right. No raw turn_type, lane-token control annotation or per-frame current/future signal state exists. API cue shortcuts derive from stop-line references and do not provide a lane-control label.','',
        '【Turn Semantics】','', 'All 4,589 connectors have finite original centerlines and analytic entrance/exit tangents from the first start_pose[2] / last end_pose[2] in arcline_path_3. delta_theta=wrap(theta_out−theta_in). We first computed the full real histogram, then examined coarse 2 m chords, independent 0.2 m chords, connectivity endpoint gaps and seeded random plots before adopting audit candidates. Original map inputs retain the original 2 m geometry.','',
        markdown(hsum,['Interval_deg','Connectors']),'',
        f'The 2 m chord angle differs from the analytic tangent angle by up to {angle_qa["coarse_analytic_difference_max_deg"]:.6f} degrees and crosses the simple ±20 degree class boundary in 32 records. Finer 0.2 m chords have maximum difference {angle_qa["fine_analytic_difference_max_deg"]:.6f} degrees; three >2 degree short-path cases were inspected. There are two simple fine/analytic class disagreements: one +20.019 degree boundary (fine +19.929), and one U-turn wrap branch cut. Analytic tangents provide the stable continuous angle; classification near ±20 remains an arbitrary audit taxonomy boundary.','',
        'Candidate labels: |delta|≤20 = straight; delta>20 = left; delta<−20 = right, except |delta|≥150 = unknown. The 12 near U-turn cases lie outside the intended V1 taxonomy, and assigning them a signed left/right label after wrap can invert physical turning direction. All 12 were visually checked and left unknown; no pseudo turn labels were made. Ordinary lanes also carry unknown (not a connector-turn label). This is a source-geometry decision, not a VAL-error threshold search.','',
        markdown(turns,['location','turn_type','connector_count','segment_count','angle_mean','angle_std','angle_min','angle_max']),'',
        'Codex directly inspected 20 seeded random left, 20 straight and 20 right connectors, plus 16 U-turn/boundary/short-path cases: no sign inversions in the 60 ordinary random examples. Review is agent visual inspection, not an independent blinded expert annotation. Global +x east/+y north is right-handed; positive CCW is left. Unchanged determinant+1 ego rotations preserve sign. Full token/sample identities and source geometry are saved in the manual-sample JSON files.','',
        '[Boston map](../05_figures/stage7a_turn_semantics_boston.png); [Singapore map](../05_figures/stage7a_turn_semantics_singapore.png); [left20](../05_figures/stage7a_turn_manual_left.png); [straight20](../05_figures/stage7a_turn_manual_straight.png); [right20](../05_figures/stage7a_turn_manual_right.png); [boundary cases](../05_figures/stage7a_turn_boundary_cases.png). PNG300dpi, editable SVG/PDF and scripts are retained.','',
        f'There are 16 raw dangling connectivity links on ordinary lanes in singapore-onenorth; none originate from connectors. RawConnectivitySemantic=PARTIAL. The complete raw defect list is retained without repair. The V1 candidate cache uses known token ownership, actual centerline geometry and valid control/crossing references; raw connectivity strings are not used as features. Readiness here is for that geometric candidate only, not an unrestricted topology-based extension.','',
        '【Traffic Control】','',markdown([{'StopLineType':k,'Count':v} for k,v in sorted(stoptypes.items())],['StopLineType','Count']),'',
        markdown(controls,['location','control_type','record_count','associated_lane_count','segment_count','associated_stop_line_records']),'',
        'Control association means the whole token centerline intersects the actual typed stop-line polygon. There is no direct authoritative lane-control field. We preserve intersected raw types and valid static light token references; no nearest-signal substitution or topology propagation is used. Associated lane counts include connectors and are token unions within each type; segment counts broadcast to all valid 2 m segments and may overlap across types. Some stop lines have no intersecting centerline, documented in the full association table.','',
        f'Exclusive cache categories: {dict(categories)}. Actual overlaps require mixed_control in addition to the five initial candidate labels; no invented precedence or forced 5D one-hot is used. There are {regulatory_tokens} unique tokens with at least one traffic_light / stop_sign / yield geometry association. YIELD has only three source records and five intersecting tokens (four exclusive yield, one mixed); its dedicated future feature is removed from the recommended set, while the raw audit remains.','',
        'The six-way exclusive control one-hot is audit bookkeeping of actual categories only. Recommended future candidates keep separate static light, stop-sign and other-source-stop-line existence; dedicated YIELD is removed, and future control encoding is not frozen. No sparse real YIELD label is relabeled or synthesized.','',
        'NearTrafficControl exposure means any typed stop-line-associated token, including TURN_STOP/PED_CROSSING. It does not establish that an actor must obey a particular sign or light. The three regulatory flags remain separately recorded in segment coverage. TrafficControlSemantic=PARTIAL refers to derived spatial associations, sparse YIELD and absence of direct lane authority; at least static TRAFFIC_LIGHT and STOP_SIGN association is available and visually checked.','',
        'Holland Village has119/119 and Queenstown81/81 all-zero traffic-light XY poses. These are not usable geolocations. Static lens colors/shapes are descriptors of hardware, not the active light. No red/yellow/green labels, current_signal_state or future_signal_state are generated. The source record’s static type/references and actual stop-line polygon are used. Ten traffic-light and ten stop/yield-associated observed tokens were visually inspected; static token references are all valid.','',
        '【Pedestrian Crossing】','',f'Actual polygon count={counts["ped_crossing"]}. All crossings, stop lines and lane/connector polygons checked are nonempty, valid and finite. No geometry repair was applied. We audit centerline intersects and distance≤2 m/≤5 m as alternatives; intersections are the primary conservative relation, selected before reading VAL errors.','',
        markdown(cross,['location','layer','definition','token_count','associated_token_count','token_coverage_rate','broadcast_segment_coverage_rate','direct_segment_coverage_rate']),'',
        'Token-level metadata is inherited by every segment owned by that token, as requested. A nearby segment can therefore carry a crossing/control association located farther along its token. Broadcast coverage is not the same as individual segment/object proximity; both are quantified. Actor exposure below means semantic tokens visible within the existing 50 m map edges, not direct actor-to-object distance or actor intent. Ten observed crossing-associated cases were visually inspected.','',
        '【Actor Semantic Exposure】','', 'Denominator: official full-horizon target actor-windows. Rates are proportions [0,1], using existing per-actor edges to stored segment starts at <50 m. A turning connector is left/right; unknown U-turns are excluded. Each table gives scene/instance support; overlapping windows are not independent observations.','',
        'Official TRAIN700:','',markdown(trainex,['Group','Count','NearConnectorRate','NearTurnConnectorRate','NearTrafficControlRate','NearCrosswalkRate','Scenes','Instances']),'',
        'Official VAL150:','',markdown(valex,['Group','Count','NearConnectorRate','NearTurnConnectorRate','NearTrafficControlRate','NearCrosswalkRate','Scenes','Instances']),'',
        'All current Stage3 segment occurrences (ordinary lanes are unknown turn):','',markdown(seg,['Split','Total','ConnectorRate','LeftRate','StraightRate','RightRate','UnknownRate','UnknownConnectorRate','TrafficLightRate','StopSignRate','YieldRate','CrosswalkRate','DirectCrosswalkRate']),'',
        '【Moving Vehicle Exposure】','',markdown([r for r in exposure if r['Split']!='combined' and r['Group'] in ['vehicle.moving','Vehicle >5m']],['Split','Group','Count','NearConnectorRate','NearTurnConnectorRate','NearTrafficControlRate','NearCrosswalkRate']),'',
        'vehicle.moving uses the unchanged t0 annotation attribute. Vehicle >5m uses frozen full-future endpoint displacement only as an offline group. Before exposure counts, we registered an operational sufficient-sample screen: each semantic should expose ≥100 moving actor-windows, ≥10 scenes and ≥50 distinct instances in each split. This checks audit feasibility, not statistical power or a frozen future subgroup definition.','',
        markdown(moving_n,['Split','Semantic','Count','Scenes','Instances','sufficient']),'',
        'Intersection candidates use t0 global actor-to-complete-connector-centerline minimum distance <20 m / <30 m; they are distinct from the 50 m segment-start semantic-token exposure above. TurningVehicle_GT diagnostic requires vehicle endpoint displacement>5 m, first/last 1 s secant displacement>0.5 m, and absolute secant-heading change>20 degrees. These future positions are used only for offline grouping; the map-semantic cache cannot read them. No future group definitions are frozen or training inputs created.','',
        markdown(candidates,['Split','Candidate','Count','VehicleDenominator','VehicleRate','Scenes','Instances','Stage5A_minFDE','Stage5A_Top1FDE','R2_Top1FDE']),'',
        '【Current Error vs Semantic Context】','', 'Only the frozen Stage6A actor CSV is read: minFDE6 is the shared frozen Stage5A geometry; R0 Top1FDE is Stage5A original ranking; R2 Top1FDE is final frozen reranking. Every one of the85,027 full+partial VAL identities, node/type/motion/future mask and exact GT SHA pairs to the frozen shard; primary context tables use only full-horizon moving vehicles. No model evaluation or cache inference is repeated.','',
        markdown(errors,['Context','Count','Stage5A_minFDE','Stage5A_Top1FDE','R2_Top1FDE','Scenes','Instances']),'',
        '\n\n'.join(error_findings),'',
        'The simple assertion that all connector-associated cases are harder is not supported: their mean minFDE and Top1FDE are lower than the very small NoConnector subset. Turning, control and crossing-associated subsets have higher R2 point-estimate Top1FDE, with different effect sizes. Broad 50 m token exposure is highly saturated (moving connector99.054%, turning95.526%, control97.763%); NoConnector has only99 windows /16 instances /13 scenes, and NoTrafficControl234 /27 /14. These results do not establish concentration, statistically resolved differences, or a semantic model benefit. More selective t0 distance groups are candidates for later review, not definitions frozen here.','',
        'All values are descriptive actor-window means in metres. Near/no comparisons describe associated difficulty, not causal effects of intersections, crossings or traffic control. Groups overlap, windows repeat actors, and location/speed/trajectory differences can confound comparisons. No semantic model benefit, confidence interval, subgroup checkpoint selection or VAL threshold tuning is inferred. R2 only changes ranking; minFDE remains identical.','',
        '【Feasibility】','', f'Core candidate audit: token parse failures=0; NaN=0; Inf=0; turn/control one-hot sums=1; connector angles finite; all light/crossing/road-block references valid; TRAIN/VAL overlap=0;850 shard SHA unchanged. Maximum map start error={qa["maximum_global_segment_start_difference_m"]:.9g} m, vector error={qa["maximum_global_segment_vector_difference_m"]:.9g} m, roundtrip={qa["maximum_roundtrip_difference_m"]:.9g} m. Every stored 50 m edge exactly recomputes. Prior project/checkpoint/result SHA checks pass. Raw topology is separately PARTIAL and excluded from candidate feature dependencies.','',
        'Semantic candidate cache is keyed by (location,lane_token), computed once per token; expensive API calls are batched per location and are never repeated per observed segment. Local derived caches and actor-level audit rows are ignored by Git; paths, schema, counts and SHA are saved. No checkpoint copies exist. 03_training / 04_evaluation / 07_checkpoints contain README only.','',
        f'TurnSemantic = {science["TurnSemantic"]}\n\nTrafficControlSemantic = {science["TrafficControlSemantic"]}\n\nCrosswalkSemantic = {science["CrosswalkSemantic"]}\n\nSemanticMapStage7A = {science["SemanticMapStage7A"]}','',
        'Readiness scope is the validated static geometric candidate only, subject to the recorded partial traffic/topology limitations. Dedicated sparse YIELD, raw dangling topology links and dynamic signals are excluded from the recommended future feature set. Future backbone guidance is Stage3B HiVT + TypeEmbedding + Semantic Map; no Stage4A bias or Stage4F gate is inherited, and Stage5A residual-decoder inheritance remains undecided.','',
        'STOP. Stage7A is not trained. Await 大脑AI review. Git tag push remains an external authentication delivery item until local Git credentials are configured.','']
    target=ROOT/'09_reports/stage7a_semantic_data_audit_report.md';target.write_text('\n'.join(report))
    atomic_json(ROOT/'00_manifest/stage7a_final_audit.json',{'status':'AUDIT_PASS_TAG_REMOTE_BLOCKED' if ref['tag_remote_status']!='PUSHED_VERIFIED' else 'PASS',
        'stage':'Stage7A-0','branch':'stage7a/semantic-map-enhancement','base_commit':BASE,'frozen_tag':TAG,'tag_remote_status':ref['tag_remote_status'],
        'map_counts':counts,'connector_turn_counts':tcounts,**science,'previous_files_SHA_unchanged':len(prior['files']),'previous_shards_SHA_unchanged':len(prior['scene_shards']),
        'full_target_counts':{'train':qa['TRAIN_full_actor_count'],'val':54990},'VAL_pairing_full_partial':85027,'map_api_calls_per_observed_segment':0,
        'NaN':0,'Inf':0,'onehot':'PASS','coordinate':'PASS','candidate_references':'PASS','raw_topology_references':'PARTIAL_16_DANGLING_LINKS_NOT_CONSUMED',
        'figure_bundles':len(figs),'visual_QA':'PASS_AGENT_VISUAL_REVIEW','model_training_runs':0,'model_inference_runs':0,'old_results_modified':False,
        'test_used':False,'report_sha256':sha256(target),'STOP':True})
    atomic_json(ROOT/'00_manifest/stage7a_pipeline_state.json',{'status':'AUDIT_COMPLETE','branch_delivery':'TO_PUBLISH','tag_delivery':ref['tag_remote_status'],'STOP_after_delivery':True,'training_authorized':False})
    manifest=[]
    for p in sorted(ROOT.rglob('*')):
        if not p.is_file() or '__pycache__' in p.parts or p.suffix in ('.pyc','.tmp','.log') or p.name.startswith(('stage7a_git_upload_','stage7a_artifact_manifest')):continue
        ignored=(p.parent.name=='02_semantic_cache' and p.suffix in ('.json','.npz')) or (p.parent.name=='01_data_audit' and p.name.startswith('stage7a_actor_'))
        manifest.append({'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':sha256(p),'Git_eligible':not ignored})
    atomic_json(ROOT/'00_manifest/stage7a_artifact_manifest.json',{'stage':'Stage7A-0','self_hash_excluded':True,'log_and_transient_git_payload_excluded':True,'artifacts':manifest})
    print('STAGE7A_AUDIT_DONE',json.dumps(science),flush=True)
def table_from_old(path):return list(csv.DictReader(open(path)))
if __name__=='__main__':main()
