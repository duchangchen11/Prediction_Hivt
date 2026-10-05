"""Recover the exact frozen No-Type GT/mask ledger without inference or preprocessing."""
import csv
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'00_manifest'))
from stage3b_common import (SceneDataset,CLASSES,BASE_ACTORS,GT_LEDGER,FREEZE,PREREG,ROOT,
    GT_fingerprint,read_json,atomic_json,sha256)
import torch

def actor_key(row):return tuple(row[k] for k in ('scene_token','sample_token','instance_token','horizon'))

def build_ledger():
    prereg=read_json(PREREG);assert sha256(BASE_ACTORS)==prereg['NoType_actor_errors_sha256']
    with BASE_ACTORS.open() as f:base=list(csv.DictReader(f))
    by_key={actor_key(row):row for row in base};assert len(by_key)==len(base)==85027
    ds=SceneDataset('val');frozen=read_json(FREEZE);source_shards=sorted({row['file_path'] for row in ds.all_rows})
    assert len(source_shards)==150
    for name in source_shards:assert sha256(ROOT/name)==frozen['scene_shards'][name]
    fields=['scene_token','sample_token','instance_token','horizon','node_in_graph','agent_type','agent_type_id',
            'motion_state','valid_future_steps','future_mask_bits','GT_trajectory_sha256']
    seen=set();full=partial=0;temp=GT_LEDGER.with_suffix('.csv.tmp');GT_LEDGER.parent.mkdir(exist_ok=True)
    with temp.open('w',newline='') as f:
        writer=csv.DictWriter(f,fields,lineterminator='\n');writer.writeheader()
        for index in range(len(ds)):
            graph=ds[index]
            for node in torch.where(graph.target_mask)[0].tolist():
                mask=graph.future_mask[node];n=int(mask.sum());assert n>0
                horizon='full_horizon' if n==12 else 'partial_future'
                row={'scene_token':graph.scene_token,'sample_token':graph.sample_token,
                     'instance_token':graph.instance_tokens[node],'horizon':horizon,'node_in_graph':node,
                     'agent_type':CLASSES[int(graph.agent_type[node])],'motion_state':graph.t0_motion_state[node],
                     'valid_future_steps':n,**GT_fingerprint(graph.positions[node,5:],mask,graph.agent_type[node])}
                key=actor_key(row);assert key in by_key and key not in seen;original=by_key[key]
                for name in ('node_in_graph','agent_type','motion_state','valid_future_steps'):assert str(row[name])==original[name],(key,name)
                last=int(torch.where(mask)[0][-1]);displacement=float(torch.linalg.vector_norm(graph.y[node,last]))
                assert abs(displacement-float(original['GT_endpoint_displacement_m']))<1e-4
                seen.add(key);writer.writerow(row)
                if horizon=='full_horizon':full+=1
                else:partial+=1
    assert seen==by_key.keys() and full==54990 and partial==30037
    temp.replace(GT_LEDGER);ds.clear()
    atomic_json(ROOT/'04_evaluation/stage3b_frozen_GT_ledger_audit.json',{'status':'PASS','actor_windows':len(seen),
        'full_horizon_actor_windows':full,'partial_future_actor_windows':partial,'VAL_scenes':150,
        'NoType_actor_errors_sha256':sha256(BASE_ACTORS),'GT_ledger_sha256':sha256(GT_LEDGER),
        'GT_ledger_relative_path':str(GT_LEDGER.relative_to(ROOT)),
        'mask_and_GT_provenance':'Exact per-actor tensors from the SHA256-verified, unchanged Stage3A frozen scene-window shards',
        'NoType_old_CSV_has_GT_trajectory_columns':False,
        'missing_GT_fields_recovered_from_frozen_graphs_not_inferred_from_endpoint':True,
        'VAL_shards_SHA256_verified':{name:frozen['scene_shards'][name] for name in source_shards},
        'preprocessing_or_model_training':False,'test_used':False})
    print('FROZEN_NO_TYPE_GT_LEDGER=PASS',full,'full',partial,'partial',flush=True)

if __name__=='__main__':torch.set_num_threads(1);build_ledger()
