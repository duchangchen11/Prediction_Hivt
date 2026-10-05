"""Official-VAL only; actor-paired vehicle retention and immutable case exports."""
import csv
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'00_manifest'))
from stage3_common import (PREVIOUS,CLASSES,GROUPS,HORIZONS,SceneDataset,atomic_json,read_json,
                          config,evaluate,errors_with_top1,model_new,model_input,sha256,verify_frozen,update_manifest,write_csv)
import numpy as np
import torch
from torch_geometric.data import Batch


@torch.no_grad()
def main():
    assert read_json(ROOT/'03_no_type_baseline/stage3_nll_summary.json')['status']=='COMPLETE'
    checkpoint=ROOT/'07_checkpoints/stage3_best_overall_minfde.pt'
    saved=torch.load(checkpoint,weights_only=False,map_location='cpu')
    assert saved['metadata']['phase']=='original_nll' and saved['metadata']['selection_metric']=='overall minFDE6'
    model=model_new();model.load_state_dict(saved['state_dict']);model.eval()
    ds=SceneDataset('val');actor_path=ROOT/'04_evaluation/stage3_no_type_actor_errors.csv'
    measured=evaluate(ds,model,actor_path=actor_path,progress=True)
    assert abs(measured['metrics']['full_horizon']['overall']['minFDE6']-saved['metadata']['validation_FDE'])<1e-4
    measured.update(primary_checkpoint=str(checkpoint.relative_to(ROOT)),checkpoint_sha256=sha256(checkpoint),checkpoint_metadata=saved['metadata'])
    atomic_json(ROOT/'03_no_type_baseline/stage3_no_type_val_metrics.json',measured)
    main_fields=('minADE6','minFDE6','MR6','Top1ADE6','Top1FDE6','NLL')
    group_labels={'overall':'Overall','vehicle':'Vehicle','pedestrian':'Pedestrian','bicycle':'Bicycle'}
    rows=[{'Method':'Multi-Type HiVT (No Type)','Group':group_labels.get(g,g),'Count':values['count'],**{k:values[k] for k in main_fields}}
          for g,values in measured['metrics']['full_horizon'].items()]
    write_csv(ROOT/'06_tables/stage3_no_type_main_results.csv',rows)
    write_csv(ROOT/'06_tables/stage3_no_type_partial_results.csv',
              [{'Method':'Multi-Type HiVT (No Type)','Group':group_labels.get(g,g),'Count':values['count'],**{k:values[k] for k in main_fields}} for g,values in measured['metrics']['partial_future'].items()])
    scene_rows=[{'scene_token':scene,'Horizon':h,'Group':g,**values} for scene,summaries in measured['scenes'].items()
                for h,groups in summaries.items() for g,values in groups.items()]
    write_csv(ROOT/'04_evaluation/stage3_no_type_scene_metrics.csv',scene_rows)
    with open(actor_path) as f:actors=list(csv.DictReader(f))
    with open(PREVIOUS/'04_evaluation/stage2c_val_actor_errors.csv') as f:old=list(csv.DictReader(f))
    key=lambda r:(r['scene_token'],r['sample_token'],r['instance_token'],r['horizon'])
    vehicles={key(r):r for r in actors if r['agent_type']=='vehicle'}
    assert len(vehicles)==len(old)
    for r in old:
        new=vehicles[key(r)]
        assert r['motion_state']==new['motion_state'] and r['valid_future_steps']==new['valid_future_steps']
    reference=read_json(PREVIOUS/'04_evaluation/stage2c_val_metrics.json')['metrics']['full_horizon']
    retention=[]
    for group,old_group in (('vehicle','overall'),('vehicle.moving','vehicle.moving'),('vehicle.stopped','vehicle.stopped'),('vehicle.parked','vehicle.parked'),('unknown','unknown')):
        a=reference[old_group];b=measured['metrics']['full_horizon'][group];assert a['count']==b['count']
        for label,v in (('Stage2C Vehicle-only HiVT',a),('Stage3A Multi-Type No-Type HiVT',b)):
            retention.append({'Method':label,'Group':group,'Count':v['count'],'minADE6':v['minADE6'],'minFDE6':v['minFDE6'],'MR6':v['MR6']})
    write_csv(ROOT/'06_tables/stage3_vehicle_retention.csv',retention)
    atomic_json(ROOT/'04_evaluation/stage3_vehicle_retention_audit.json',
                {'status':'PASS','exactly_paired_vehicle_actor_windows':len(old),'official_val_vehicle_identities_horizons_masks_motion_states_identical':True,
                 'reference_checkpoint_sha256':read_json(PREVIOUS/'04_evaluation/stage2c_val_metrics.json')['checkpoint_sha256'],
                 'stage3_checkpoint_sha256':sha256(checkpoint),'test_used':False})
    by_sample={(r['scene_token'],r['sample_token']):i for i,r in enumerate(ds.rows)}
    cases=[];coverage={};folder=ROOT/'04_evaluation/cases';folder.mkdir(exist_ok=True)
    for cls,minimum in zip(CLASSES,(5.,1.,2.)):
        full=[r for r in actors if r['agent_type']==cls and r['horizon']=='full_horizon']
        meaningful=[r for r in full if float(r['GT_endpoint_displacement_m'])>=minimum and (cls!='vehicle' or r['motion_state']=='vehicle.moving')]
        assert len(meaningful)>=2
        pools={'success':sorted(meaningful,key=lambda r:(float(r['minFDE6']),float(r['minADE6']))),
               'failure':sorted(full,key=lambda r:(-float(r['minFDE6']),-float(r['minADE6'])))}
        for kind,pool in pools.items():
            selected=[];instances=set()
            for row in pool:
                if row['instance_token'] in instances:continue
                instances.add(row['instance_token']);selected.append(row)
                if len(selected)==2:break
            assert len(selected)==2
            coverage[cls+'_'+kind]=len(selected)
            for number,row in enumerate(selected,1):
                dataset_index=by_sample[(row['scene_token'],row['sample_token'])]
                graph=ds[dataset_index];node=int(row['node_in_graph'])
                assert graph.instance_tokens[node]==row['instance_token']
                # Reproduce the full-VAL batch partition for numerical traceability.
                # The actor's graph/coordinates are unchanged; context graphs remain disjoint.
                batch_size=config()['batch_size'];batch_start=(dataset_index//batch_size)*batch_size
                graphs=[ds[i] for i in range(batch_start,min(batch_start+batch_size,len(ds)))]
                data=Batch.from_data_list(graphs).cuda();output_node=int(data.ptr[dataset_index-batch_start])+node
                output=model(model_input(data));prediction,errors=errors_with_top1(model,output,data)
                source={**row,'history_trajectory_m':graph.positions[node,:5].tolist(),'GT_trajectory_m':graph.positions[node,5:].tolist(),
                        'history_mask':graph.history_mask[node].tolist(),'future_mask':graph.future_mask[node].tolist(),
                        'history_times_seconds':graph.history_times.tolist(),'future_times_seconds':graph.future_times.tolist(),
                        'HiVT_trajectories_m':prediction[output_node].cpu().tolist(),'mode_probabilities':output['mode_prob'][output_node].cpu().tolist(),
                        'best_FDE_mode_zero_based':int(errors['best_mode'][output_node]),'top1_mode_zero_based':int(errors['top1_mode'][output_node]),
                        'reproduced_VAL_batch_start_index':batch_start,'reproduced_VAL_batch_size':len(graphs),
                        'lane_positions_m':graph.lane_positions.tolist(),'lane_vectors_m':graph.lane_vectors.tolist(),
                        'origin_global_m':graph.origin.tolist(),'ego_yaw_global_rad':float(graph.ego_yaw),
                        'coordinate_frame':'t0 ego +x forward, +y left, meters','checkpoint_sha256':sha256(checkpoint),
                        'case_selection':('success: low-FDE rank within class; full horizon; GT displacement >= vehicle5m/pedestrian1m/bicycle2m; vehicle t0 moving'
                                          if kind=='success' else 'failure: high-FDE rank within class among all full-horizon targets'),
                        'numeric_metrics':{k:float(row[k]) for k in ('minADE6','minFDE6','Top1ADE6','Top1FDE6','NLL')},
                        'source':'official VAL; primary overall-FDE original NLL checkpoint'}
                name=f'stage3_{cls}_{kind}_{number:03d}'
                atomic_json(folder/(name+'.json'),source)
                cases.append({'name':name,'class':cls,'kind':kind,'source_json':str((folder/(name+'.json')).relative_to(ROOT)),
                              'scene_token':row['scene_token'],'sample_token':row['sample_token'],'instance_token':row['instance_token']})
    atomic_json(ROOT/'04_evaluation/stage3_case_manifest.json',{'cases':cases,'coverage':coverage,'source_checkpoint_sha256':sha256(checkpoint),
                'selection_predeclared':'class-wise low-FDE meaningful-motion successes; high-FDE failures; different instances within each pair',
                'coordinate_adjustments':False,'smoothing':False,'primary_metrics_unchanged_by_case_selection':True})
    verify_frozen();update_manifest();print('FORMAL_VAL',measured['metrics']['full_horizon'],flush=True)
    print('VEHICLE_RETENTION=PASS; CASE_JSON_EXPORT=PASS',flush=True)


if __name__=='__main__':
    torch.set_num_threads(4);main()
