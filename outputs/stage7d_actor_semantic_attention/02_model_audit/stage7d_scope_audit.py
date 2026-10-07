"""AST-equivalence of original equations after removing only authorized score inputs."""
from pathlib import Path
import ast,copy,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'00_manifest'))
from stage7d_common import atomic_json,sha256,PROJECT

class StripAuthorized(ast.NodeTransformer):
    def visit_Name(self,n):
        n.id={'SemanticALEncoder':'ALEncoder','SemanticLocalEncoder':'LocalEncoder'}.get(n.id,n.id)
        return n
    def visit_Call(self,n):
        self.generic_visit(n)
        if isinstance(n.func,ast.Attribute):
            if n.func.attr in ('al_encoder','propagate'):
                n.keywords=[k for k in n.keywords if k.arg not in ('lane_semantic','actor_type')]
            if n.func.attr=='_mha_block' and len(n.args)==11:n.args=n.args[:-2]
        return n
    def visit_FunctionDef(self,n):
        drop={'lane_semantic','actor_type','lane_semantic_j','actor_type_i'}
        args=n.args.args;default_start=len(args)-len(n.args.defaults)
        defaults=[(arg,default) for arg,default in zip(args[default_start:],n.args.defaults) if arg.arg not in drop]
        n.args.args=[a for a in args if a.arg not in drop];n.args.defaults=[d for a,d in defaults]
        body=[]
        for stmt in n.body:
            if isinstance(stmt,ast.Assign):
                target=stmt.targets[0]
                if isinstance(target,ast.Name) and target.id in ('relative','z'):continue
                if isinstance(target,ast.Attribute) and target.attr=='semantic_score_mlp':continue
                if isinstance(target,ast.Name) and target.id=='alpha' and isinstance(stmt.value,ast.BinOp) and isinstance(stmt.value.op,ast.Add):continue
            if isinstance(stmt,ast.Expr) and isinstance(stmt.value,ast.Call) and isinstance(stmt.value.func,ast.Attribute) and stmt.value.func.attr=='zeros_':continue
            body.append(stmt)
        n.body=body
        return self.generic_visit(n)

def methods(source,name):
    cls=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name==name)
    return {n.name:n for n in cls.body if isinstance(n,ast.FunctionDef)}

def main():
    old=(PROJECT/'models/hivt_runtime/local_encoder.py').read_text()
    own=(ROOT/'00_manifest/stage7d_local_encoder.py').read_text();checked=[]
    for base,new in [('LocalEncoder','SemanticLocalEncoder'),('ALEncoder','SemanticALEncoder')]:
        b=methods(old,base);d=methods(own,new);assert set(b)==set(d)
        for name in b:
            node=StripAuthorized().visit(copy.deepcopy(d[name]))
            assert ast.dump(node,include_attributes=False)==ast.dump(b[name],include_attributes=False),(base,name)
            checked.append(base+'.'+name)
    a=(ROOT.parent/'stage7a_semantic_map_hivt/00_manifest/stage7a_dataset.py').read_text()
    d=(ROOT/'00_manifest/stage7d_dataset.py').read_text()
    fn=lambda s:next(n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name=='semantic_vector')
    assert ast.dump(fn(a),include_attributes=False)==ast.dump(fn(d),include_attributes=False)
    atomic_json(ROOT/'02_model_audit/stage7d_scope_audit.json',{'status':'PASS','methods_equivalent_after_removing_only_authorized_score_change':checked,
        'removed_changes':'semantic inputs and one score addition; 14→32→8 zero-last MLP constructor',
        'original_KV_embedding_gate_FFN_equations_identical':True,'semantic_vector_AST_identical_to_Stage7A':True,
        'runtime_source_sha256':sha256(PROJECT/'models/hivt_runtime/local_encoder.py'),'Stage7E_executed':False,'R2_executed':False})
    print('ARCHITECTURE_SCOPE_PASS',checked)

if __name__=='__main__':main()
