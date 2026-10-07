"""Losslessly expand normalized edge archives to the user-requested record schema."""
from pathlib import Path
import argparse,json
import numpy as np

def records(path,model):
    assert model in ('Stage3B','Stage7A')
    with np.load(path,allow_pickle=False) as z:
        for i in range(len(z['edge_actor_index'])):
            g=int(z['edge_graph_index'][i]);a=int(z['edge_actor_index'][i]);l=int(z['edge_lane_index'][i])
            yield {'scene_token':str(z['scene_token'][g]),'sample_token':str(z['sample_token'][g]),
                'instance_token':str(z['actor_instance_token'][a]),'actor_node_index':int(z['edge_actor_node_index'][i]),
                'lane_index':int(z['edge_lane_local_index'][i]),'lane_token':str(z['lane_token'][l]),
                'lane_semantic':z['lane_semantic'][l].tolist(),'distance_actor_lane':float(z['distance_actor_lane'][i]),
                'attention_per_head':z[model+'_attention_per_head'][i].tolist(),
                'mean_attention':float(z[model+'_attention_mean'][i]),'max_attention':float(z[model+'_attention_max'][i])}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('archive');p.add_argument('--model',default='Stage7A');p.add_argument('--limit',type=int,default=5)
    a=p.parse_args()
    for i,row in enumerate(records(a.archive,a.model)):
        if i>=a.limit:break
        print(json.dumps(row))
