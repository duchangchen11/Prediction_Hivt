"""Capture exact frozen forwards, preserving every original message statement."""
import ast
import csv
import hashlib
import inspect
import textwrap
import types
import time
from stage4fd_common import *
import stage4a_global_interactor as formal
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import softmax

def instrumented_message():
    source=textwrap.dedent(inspect.getsource(formal.TypeConditionedGlobalInteractorLayer.message))
    tree=ast.parse(source);changed=ast.parse(source);body=[];count=0
    expected=ast.dump(ast.parse('alpha = alpha + relation_bias').body[0])
    for statement in changed.body[0].body:
        if ast.dump(statement)==expected:
            body+=ast.parse('self.stage4fd_observer.before(self, alpha, relation_bias, index, ptr, size_i)').body
            body.append(statement)
            body+=ast.parse('self.stage4fd_observer.after(self, alpha)').body;count+=1
        else:body.append(statement)
    assert count==1;changed.body[0].body=body
    restored=ast.parse(ast.unparse(changed));restored.body[0].body=[s for s in restored.body[0].body if not (isinstance(s,ast.Expr) and isinstance(s.value,ast.Call) and ast.unparse(s.value.func).startswith('self.stage4fd_observer.'))]
    assert ast.dump(restored)==ast.dump(tree)
    namespace=dict(formal.__dict__);exec(compile(ast.fix_missing_locations(changed),'<stage4fd_exact_observation>','exec'),namespace)
    return namespace['message'],{'formal_message_source':source,'instrumented_source':ast.unparse(changed),'restored_AST_equals_original':True,
        'actual_original_addition_softmax_value_aggregation_retained':True,'only_observation_statements_inserted':True}

class Observer:
    def __init__(self,name,model):
        self.name,self.model=name,model;self.layers=list(model.global_interactor.global_interactor_layers)
        self.ids={id(layer):i for i,layer in enumerate(self.layers)};self.original=[l.message for l in self.layers]
        self.pending={};self.handles=[];fn,self.ast_audit=instrumented_message()
        gi=model.global_interactor;gi.record_relation_bias=True
        if name=='F':gi.record_necessity_gate=True
        for layer in self.layers:
            layer.stage4fd_observer=self;layer.message=types.MethodType(fn,layer)
            self.handles.append(layer.attn_drop.register_forward_pre_hook(lambda m,args,layer=layer:self.attention(layer,args[0])))
        self.edge_digest=hashlib.sha256();self.edge_count=0;self.max_algebra_diff=0.;self.max_alpha_diff=0.;self.calls=[0]*3
    def prepare(self,data,full):
        self.data=data;self.full=torch.as_tensor(full,device=data.positions.device);n=data.num_nodes
        self.values=torch.zeros((n,3,8,len(METRICS)),device=data.positions.device,dtype=torch.float64)
        self.pair_sums=torch.zeros((9,3,8,4),device=data.positions.device,dtype=torch.float64)
        self.pair_counts=torch.zeros(9,device=data.positions.device,dtype=torch.long)
        self.edge=None;self.gates=None;self.degree=None
    def before(self,layer,base,bias,index,ptr,size_i):
        assert not torch.is_grad_enabled() and not self.model.training and not layer.training
        li=self.ids[id(layer)];obs=self.model.global_interactor.relation_bias_observation;edges=obs['edge_index']
        assert torch.equal(index,edges[1]);raw=obs['bias'][:,li,:]
        if self.name=='F':
            gates=self.model.global_interactor.necessity_gate_observation['gate'];expected=gates[index,None]*raw
            assert torch.equal(bias,expected)
        else:gates=torch.ones(self.data.num_nodes,device=base.device);assert torch.equal(bias,raw)
        assert base.shape==bias.shape==(len(index),8)
        if li==0:
            self.edge=edges.detach().cpu().numpy();self.gates=gates.detach().cpu().numpy()
            self.degree=torch.bincount(index,minlength=self.data.num_nodes)
            assert not (edges[0]==edges[1]).any() and not self.data.padding_mask[edges.flatten(),4].any()
            assert torch.equal(self.data.batch[edges[0]],self.data.batch[edges[1]])
            self.edge_count+=len(index)
            self.pair_counts=torch.bincount(obs['pair_ids'][self.full[index]],minlength=9)
        else:assert np.array_equal(self.edge,edges.detach().cpu().numpy())
        assert torch.equal(obs['pair_ids'],3*self.data.agent_type[edges[1]]+self.data.agent_type[edges[0]])
        self.pending[li]=(base,bias,raw,index,ptr,size_i,obs,gates)
    def after(self,layer,actual_final):
        li=self.ids[id(layer)];base,bias,*_=self.pending[li]
        diff=float((actual_final-(base+bias)).abs().max()) if base.numel() else 0.
        assert diff<1e-6;self.max_algebra_diff=max(self.max_algebra_diff,diff)
        self.pending[li]=(*self.pending[li],actual_final)
    def attention(self,layer,alpha_final):
        li=self.ids[id(layer)];base,bias,raw,index,ptr,size_i,obs,gates,final=self.pending.pop(li)
        alpha_base=softmax(base,index,ptr,size_i)
        check=softmax(final,index,ptr,size_i)
        diff=float((check-alpha_final).abs().max()) if check.numel() else 0.
        assert diff<1e-6;self.max_alpha_diff=max(self.max_alpha_diff,diff)
        assert all(torch.isfinite(t).all() for t in (base,raw,bias,final,alpha_base,alpha_final))
        n=self.data.num_nodes;deg=self.degree;safe=deg.clamp(min=1).double()[:,None]
        def reduce(t):
            result=torch.zeros((n,8),dtype=torch.float64,device=base.device);result.index_add_(0,index,t.double());return result
        b=reduce(base.abs())/safe;r=reduce(raw.abs())/safe;e=reduce(bias.abs())/safe
        sums=reduce(alpha_base);sums_final=reduce(alpha_final);valid=deg>0
        assert torch.allclose(sums[valid],torch.ones_like(sums[valid]),atol=2e-6) and torch.allclose(sums_final[valid],torch.ones_like(sums_final[valid]),atol=2e-6)
        tiny=torch.finfo(alpha_base.dtype).tiny
        hb=reduce(-alpha_base*torch.log(alpha_base.clamp_min(tiny)));hf=reduce(-alpha_final*torch.log(alpha_final.clamp_min(tiny)))
        shift=reduce((alpha_final-alpha_base).abs());assert (shift<=2.000002).all()
        winners=[];idx=index[:,None].expand(-1,8);edge_ids=torch.arange(len(index),device=base.device)[:,None].expand(-1,8)
        for alpha in (alpha_base,alpha_final):
            maxima=torch.full((n,8),-torch.inf,device=base.device);maxima.scatter_reduce_(0,idx,alpha,reduce='amax',include_self=True)
            candidates=torch.where(alpha==maxima[index],edge_ids,len(index))
            win=torch.full((n,8),len(index),dtype=torch.long,device=base.device);win.scatter_reduce_(0,idx,candidates,reduce='amin',include_self=True);winners.append(win)
        switch=(winners[0]!=winners[1]).double()
        names={'A':('A_base','A_raw','A_shift','A_switch','A_Hbase','A_Hfinal'),'F':('F_base','F_raw','F_shift','F_switch','F_Hbase','F_Hfinal')}[self.name]
        for name,value in zip(names,(b,r,shift,switch,hb,hf)):self.values[:,li,:,METRICS.index(name)]=value
        if self.name=='F':self.values[:,li,:,METRICS.index('F_effective')]=e
        full=self.full[index];pairs=obs['pair_ids'][full]
        measurements=torch.stack((base.abs(),raw.abs(),bias.abs(),gates[index,None].expand(-1,8)),dim=-1).double()
        self.pair_sums[:,li,:,:].index_add_(0,pairs,measurements[full]);self.calls[li]+=1
    def close(self):
        assert not self.pending
        for h in self.handles:h.remove()
        for l,original in zip(self.layers,self.original):l.message=original;del l.stage4fd_observer
        gi=self.model.global_interactor;gi.record_relation_bias=False;gi.relation_bias_observation=None
        if self.name=='F':gi.record_necessity_gate=False;gi.necessity_gate_observation=None

@torch.no_grad()
def smoke(networks,ds):
    torch.set_num_threads(1);data=Batch.from_data_list([ds[i] for i in range(2)]);result={}
    for name,model in networks.items():
        before=state_digest(model.state_dict());original=model(model_input(data));observer=Observer(name,model)
        try:
            observer.prepare(data,np.ones(data.num_nodes,dtype=bool));changed=model(model_input(data))
            diffs={k:float((original[k]-changed[k]).abs().max()) for k in ('raw_prediction','mode_logits','mode_prob')}
            assert all(x<1e-6 for x in diffs.values());assert before==state_digest(model.state_dict())
            result[name]={'status':'PASS','prediction_max_differences':diffs,'forward_algebra_max_diff':observer.max_algebra_diff,'actual_attention_max_diff':observer.max_alpha_diff,'AST_audit':observer.ast_audit}
        finally:observer.close()
    torch.set_num_threads(4);atomic_json(DIAG/'stage4fd_instrumentation_audit.json',{'status':'PASS','fixed_real_VAL_batch_CPU_single_thread':True,'models':result})

@torch.no_grad()
def main():
    assert not RAW.exists(),'Do not overwrite a completed/raw diagnostic run'
    if not FREEZE.exists():freeze()
    verify();protocol();networks=models();ds=SceneDataset('val');assert len(ds)==3603
    membership=read_membership();actors={actor_key(r):r for r in read_actors(ROOT/'04_evaluation/stage4f_actor_errors.csv')}
    assert actors.keys()==membership.keys();smoke(networks,ds)
    before={k:state_digest(m.state_dict()) for k,m in networks.items()}
    for m in networks.values():m.cuda()
    observers={k:Observer(k,m) for k,m in networks.items()};pair_sums={k:np.zeros((9,3,8,4)) for k in networks};pair_counts=np.zeros(9,dtype=np.int64)
    identity_fields=['id','node','scene_token','sample_token','target_instance_token','node_in_graph','type','motion_group','horizon','current_valid']
    total_id=0;windows=[];scenes=set();edge_hash=hashlib.sha256();record_count=0;counts={};start=time.monotonic();zero_degree=0
    with RAW.open('wb') as raw,IDENTITY.open('w',newline='') as identity:
        writer=csv.DictWriter(identity,identity_fields,lineterminator='\n');writer.writeheader()
        loader=DataLoader(ds,batch_size=16,shuffle=False,num_workers=0)
        for batch_id,cpu in enumerate(loader):
            original=fingerprint(cpu);records,ids,current=metadata(cpu,membership,actors,total_id);full=records['population']==1
            data=cpu.cuda();observed={}
            for name,model in networks.items():
                observer=observers[name];observer.prepare(data,full);out=model(model_input(data))
                assert torch.isfinite(out['raw_prediction']).all();del out
                observed[name]=(observer.edge.copy(),observer.degree.cpu().numpy(),observer.gates.copy(),observer.values.cpu().numpy())
                pair_sums[name]+=observer.pair_sums.cpu().numpy()
            ae,ad,ag,av=observed['A'];fe,fd,fg,fv=observed['F']
            assert np.array_equal(ae,fe) and np.array_equal(ad,fd),'Edge pairing mismatch'
            assert np.array_equal(observers['A'].pair_counts.cpu().numpy(),observers['F'].pair_counts.cpu().numpy())
            pair_counts+=observers['A'].pair_counts.cpu().numpy()
            assert original==fingerprint(cpu),'Original input changed'
            records['degree']=ad;records['gate']=fg;records['values']=av+fv
            assert np.isfinite(records['values']).all() and np.isfinite(fg[current]).all()
            # Exact ordered EdgeKeys. Each scene/sample and instance pair is unique.
            ptr=cpu.ptr.cpu().numpy();expected_edges=0
            for graph in range(cpu.num_graphs):
                ids_graph=np.flatnonzero(current[int(ptr[graph]):int(ptr[graph+1])])+int(ptr[graph]);n=len(ids_graph)
                expected_edges+=n*(n-1);scene,sample=cpu.scene_token[graph],cpu.sample_token[graph]
                windows.append((scene,sample));scenes.add(scene)
            assert len(ae[0])==expected_edges and np.unique(ae[0]*cpu.num_nodes+ae[1]).size==len(ae[0])
            keys=''.join(f'{ids[s]["scene_token"]}\t{ids[s]["sample_token"]}\t{ids[s]["target_instance_token"]}\t{ids[t]["target_instance_token"]}\n' for s,t in ae.T).encode()
            edge_hash.update(keys)
            for observer in observers.values():observer.edge_digest.update(keys)
            eligible=current;zero_degree+=int((current&(ad==0)).sum());records[eligible].tofile(raw);writer.writerows(row for row,valid in zip(ids,current) if valid)
            record_count+=int(current.sum());total_id+=cpu.num_nodes
            for population in (0,1,2):counts[population]=counts.get(population,0)+int((current&(records['population']==population)).sum())
            if (batch_id+1)%10==0 or batch_id==225:print('PAIRED_CAPTURE',batch_id+1,'/226',len(windows),'/3603',flush=True)
            del data,cpu,observed,records
    assert len(windows)==len(set(windows))==3603 and len(scenes)==150
    assert counts[1]==54990 and counts[2]==30037
    assert all(observer.calls==[226]*3 for observer in observers.values())
    assert observers['A'].edge_digest.hexdigest()==observers['F'].edge_digest.hexdigest()==edge_hash.hexdigest()
    assert observers['A'].edge_count==observers['F'].edge_count==3058206
    for name,model in networks.items():assert before[name]==state_digest(model.state_dict()) and all(p.grad is None for p in model.parameters())
    audit={'status':'PASS','VAL_scenes':150,'windows':3603,'batches':226,'actual_edges':observers['A'].edge_count,'edge_head_observations':observers['A'].edge_count*24,
        'EdgeKey_schema':['scene_token','sample_token','source_instance_token','target_instance_token'],'ordered_EdgeKey_sha256':edge_hash.hexdigest(),
        'EdgeHeadKey_layers':[1,2,3],'EdgeHeadKey_heads':list(range(1,9)),'all_ordered_edges_and_each_layer_head_paired':True,
        'window_order_sha256':hashlib.sha256(json.dumps(windows).encode()).hexdigest(),'all_current_valid_targets':record_count,'full_targets':counts[1],'partial_targets':counts[2],'context_only':counts[0],
        'zero_incoming_current_targets':zero_degree,'complete_directed_current_valid_graph_verified_every_window':True,'original_inputs_unchanged':True,
        'forward_algebra_max_diff':{k:o.max_algebra_diff for k,o in observers.items()},'actual_softmax_max_diff':{k:o.max_alpha_diff for k,o in observers.items()},
        'parameters_buffers_unchanged':True,'all_parameters_grad_None':True,'no_training_optimizer_backward_selection':True,
        'raw_target_head_schema':{'dtype':DTYPE.descr,'metrics':METRICS,'row_count':record_count,'size_bytes':RAW.stat().st_size,'sha256':sha256(RAW)},
        'raw_identity_sha256':sha256(IDENTITY),'checkpoint_sha256':{k:d for k,(p,d) in CHECKPOINTS.items()},'elapsed_seconds':time.monotonic()-start}
    atomic_json(DIAG/'stage4fd_pair_aggregates.json',{'status':'PASS','counts':pair_counts.tolist(),'A_sums':pair_sums['A'].tolist(),'F_sums':pair_sums['F'].tolist(),'measurement_order':['abs_base','abs_raw','abs_effective','gate'],'population':'full-horizon target incoming edges'})
    for observer in observers.values():observer.close()
    ds.clear();verify();atomic_json(DIAG/'stage4fd_capture_audit.json',audit);print('STAGE4FD_CAPTURE_PASS',audit['actual_edges'],flush=True)

if __name__=='__main__':torch.set_num_threads(4);main()
