"""Observable-only quota graph feature assembly; all current actors retained."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'00_manifest'))
from stage8a0c_common import node_features,neighbors,interaction_edges,ego_to_global
import numpy as np
import torch

@torch.no_grad()
def build_window(w,index):
    selected=torch.where(~w.history_padding[:,4])[0].numpy()
    local=w.predicted[selected].numpy().reshape(-1,12,2)
    points=ego_to_global(local,w.origin,w.yaw)
    types=np.repeat(w.actor_type[selected].numpy(),6)
    selection=index.select_batch(types,points,w.map_location)
    delta=local[:,-1]-local[:,-3];heading=np.arctan2(delta[:,1],delta[:,0])
    selection.update(index.features(points,w.map_location,w.yaw,heading,selection))
    features={k:v.reshape((len(selected),6)+v.shape[1:]) for k,v in selection.items()}
    node=node_features(w);idx,keep=neighbors(w)
    edge=interaction_edges(w,idx,keep,torch.from_numpy(selected))
    return selected,features,node,idx,keep,edge
