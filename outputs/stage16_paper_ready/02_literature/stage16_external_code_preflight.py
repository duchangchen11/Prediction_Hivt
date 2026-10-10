"""Shape/gradient checks of actual published scoring code, without optimizer steps."""
from pathlib import Path
import ast,sys,types,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'00_manifest'))
from stage16_common import *
def selected(path,names,scope):
 tree=ast.parse(path.read_text());nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
 exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(path),'exec'),scope)
def main():
 runtime();torch.manual_seed(2022);folder=ROOT/'02_literature/cache';scope={'torch':torch,'nn':torch.nn,'F':torch.nn.functional}
 base=folder/'TNT-Trajectory-Prediction/core/model/layers'
 selected(base/'basic_module.py',{'MLP'},scope);selected(base/'scoring_and_selection.py',{'distance_metric','TrajScoreSelection'},scope)
 tnt=scope['TrajScoreSelection'](64,horizon=12)
 # Context zeros check only tensor contract, not a replacement for VectorNet context.
 x=torch.zeros(4,1,64);tr=torch.randn(4,6,24);gt=torch.randn(4,24);p=tnt(x,tr);loss=tnt.loss(x,tr,gt);loss.backward()
 assert p.shape==(4,6) and torch.isfinite(loss) and all(v.grad is not None and torch.isfinite(v.grad).all() for v in tnt.parameters())
 rrpath=folder/'R-RNet/rrnet/modeling/rrnet.py';tree=ast.parse(rrpath.read_text());cl=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='RRNet')
 init=next(n for n in cl.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
 assignments=[n for n in init.body if isinstance(n,ast.Assign) and any(isinstance(a,ast.Attribute) and a.attr in ('confidence_encoder','confidence_decoder') for a in n.targets)]
 rrinit=ast.parse('def __init__(self):\n super().__init__()\n self.cfg=types.SimpleNamespace(DEC_OUTPUT_DIM=2)').body[0];rrinit.body+=assignments
 pred=next(n for n in cl.body if isinstance(n,ast.FunctionDef) and n.name=='predict_prob')
 rrclass=ast.ClassDef(name='ScorerContract',bases=[ast.parse('torch.nn.Module',mode='eval').body],keywords=[],body=[rrinit,pred],decorator_list=[])
 rscope={'torch':torch,'nn':torch.nn,'F':torch.nn.functional,'types':types}
 exec(compile(ast.fix_missing_locations(ast.Module(body=[rrclass],type_ignores=[])),str(rrpath),'exec'),rscope)
 selected(folder/'R-RNet/rrnet/layers/loss.py',{'prob_loss','custom_function'},rscope)
 rr=rscope['ScorerContract']();history=torch.randn(4,6,5,2);goal=torch.randn(4,6,1,2);raw,prob=rr.predict_prob(history,goal);rl=rscope['prob_loss'](raw,goal[:,:,0],torch.randn(4,2));rl.backward()
 assert prob.shape==(4,6,1) and torch.isfinite(rl) and all(v.grad is not None and torch.isfinite(v.grad).all() for v in rr.parameters())
 for q in (p,prob.squeeze(-1)):assert torch.allclose(q.sum(-1),torch.ones(4),atol=1e-6)
 atomic_json(ROOT/'02_literature/stage16_code_feasibility.json',{'Status':'PASS_SCORER_CONTRACT_ONLY','TNT':{'Parameters':sum(v.numel() for v in tnt.parameters()),'Input':'context[4,1,64], candidates[4,6,24]','Output':list(p.shape),'GTInput':False,'GTOnlyLoss':True},'RRNet':{'Parameters':sum(v.numel() for v in rr.parameters()),'Input':'history[4,6,5,2], goals[4,6,1,2]','Output':list(prob.shape),'GTInput':False,'GTOnlyLoss':True},'OptimizerSteps':0,'Data':'synthetic shape probes, no evaluation metrics','SyntheticContextNotScientificBaseline':True,'FullPaperReproduction':False,'TNTFiles':read_json(ROOT/'02_literature/TNT-Trajectory-Prediction_code_audit.json'),'RRNetFiles':read_json(ROOT/'02_literature/R-RNet_code_audit.json')})
 print('PASS both actual scorer tensor/gradient contracts; zero optimizer steps')
if __name__=='__main__':main()
