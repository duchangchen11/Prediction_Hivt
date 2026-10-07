"""Observe existing softmax/dropout inputs; hooks never return a replacement."""
import numpy as np

class LaneCapture:
    def __init__(self, model, representations=True):
        self.al=model.local_encoder.al_encoder
        assert not model.training and not self.al.attn_drop.training
        self.data={};self.handles=[]
        self.handles.append(self.al.register_forward_pre_hook(self.edge_hook,with_kwargs=True))
        self.handles.append(self.al.attn_drop.register_forward_pre_hook(self.alpha_hook))
        if representations:
            self.handles.append(self.al.lane_embed.register_forward_hook(self.norm_hook('geometry_norm')))
            self.handles.append(self.al.lin_k.register_forward_hook(self.norm_hook('key_norm')))
            self.handles.append(self.al.lin_v.register_forward_hook(self.value_hook))

    def edge_hook(self, module, args, kwargs):
        self.data={}
        self.data['edge_index']=kwargs['edge_index'].detach().cpu().numpy().copy()
        self.data['edge_attr']=kwargs['edge_attr'].detach().cpu().numpy().copy()
        # No return value: original arguments pass through unmodified.

    def alpha_hook(self, module, args):
        self.data['alpha']=args[0].detach().cpu().numpy().copy()

    def norm_hook(self, name):
        def observe(module,args,output):
            a=output.detach().cpu().numpy()
            self.data[name]=np.linalg.norm(a,axis=-1)
        return observe

    def value_hook(self,module,args,output):
        self.data['value']=output.detach().cpu().numpy().copy()
        self.data['value_norm']=np.linalg.norm(self.data['value'],axis=-1)

    def close(self):
        for h in self.handles:h.remove()
        self.handles=[]
