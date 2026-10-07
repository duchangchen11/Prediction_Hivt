"""Read-only, exact absolute bias statistics over all retained VAL edges/heads."""
import numpy as np

class BiasStatistics:
    def __init__(self, model):
        self.chunks={i:[] for i in range(3)}
        al=model.local_encoder.al_encoder
        self.handles=[al.register_forward_pre_hook(self.edges,with_kwargs=True),
                      al.semantic_score_mlp.register_forward_hook(self.bias)]

    def edges(self,module,args,kwargs):
        self.types=kwargs['actor_type'][kwargs['edge_index'][1]].detach().cpu().numpy()

    def bias(self,module,args,output):
        values=output.detach().cpu().numpy()
        assert np.isfinite(values).all(), 'Nonfinite semantic bias'
        for i in range(3):self.chunks[i].append(np.abs(values[self.types==i]).ravel())

    def finish(self):
        for h in self.handles:h.remove()
        arrays=[np.concatenate(self.chunks[i]) for i in range(3)]
        self.chunks.clear();rows=[]
        # In-place quantiles keep peak RAM bounded; statistics are exact, not sampled.
        overall=np.concatenate(arrays)
        for name,a in [('Overall',overall),*zip(('Vehicle','Pedestrian','Bicycle'),arrays)]:
            count=len(a);assert count and count%8==0 and np.isfinite(a).all()
            mean=float(a.mean(dtype=np.float64));maximum=float(a.max())
            q=np.quantile(a,[.5,.95],overwrite_input=True)
            rows.append({'Group':name,'Edges':count//8,'Heads':8,'mean_abs_bias':mean,
                         'median_abs_bias':float(q[0]),'p95_abs_bias':float(q[1]),'max_abs_bias':maximum,
                         'population':'all retained lane-actor VAL150 edges, including context actors'})
        return rows
