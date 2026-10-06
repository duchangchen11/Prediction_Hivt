"""Descriptive first-layer norms and one fixed permutation per feature on HeadDev."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'00_manifest'),str(ROOT/'04_evaluation')]
from stage6a_common import *
from stage6a_head import ranking_loss
from stage6a_features import FEATURE_NAMES
from stage6a_evaluate import load_head
import numpy as np

@torch.no_grad()
def assess(model,data,permutation=None,column=None):
    n=len(data['base_logits']);fde=loss=0.;chosen=[]
    for start in range(0,n,4096):
        end=min(n,start+4096);x=data['features_R2'][start:end].clone()
        if column is not None:x[:,:,column]=data['features_R2'][permutation[start:end],:,column]
        out=model(x,data['base_logits'][start:end]);top=out['mode_prob'].argmax(-1)
        idx=torch.arange(len(top),device='cuda');errors=data['FDE_by_mode'][start:end]
        fde+=float(errors[idx,top].double().sum());loss+=float(ranking_loss(out['mode_logits'],errors))*len(top)
        chosen.append(top.cpu())
    return {'Top1FDE':fde/n,'rank_loss':loss/n,'top1_mode':torch.cat(chosen)}

def main():
    verify_frozen();model,_=load_head('R2');cpu=load_features('headdev');data={k:v.cuda() for k,v in cpu.items() if torch.is_tensor(v)}
    baseline=assess(model,data);weights=model.net[0].weight.detach().double().norm(dim=0).cpu().tolist()
    generator=torch.Generator().manual_seed(2022);permutation=torch.randperm(len(data['base_logits']),generator=generator).cuda()
    rows=[]
    for column in range(19):
        row={'Feature':FEATURE_NAMES[column],'Column':column,'First_layer_weight_L2_norm':weights[column],
             'Interpretation':'descriptive; correlated features; no feature selection or retraining'}
        if column>=12:
            measured=assess(model,data,permutation,column)
            row.update(HeadDev_permutation_Top1FDE_delta=measured['Top1FDE']-baseline['Top1FDE'],
                HeadDev_permutation_rank_loss_delta=measured['rank_loss']-baseline['rank_loss'],
                HeadDev_top1_changed_rate=float((measured['top1_mode']!=baseline['top1_mode']).float().mean()))
        else:row.update(HeadDev_permutation_Top1FDE_delta=None,HeadDev_permutation_rank_loss_delta=None,HeadDev_top1_changed_rate=None)
        rows.append(row)
    write_csv(ROOT/'06_tables/stage6a_feature_importance.csv',rows)
    atomic_json(ROOT/'04_evaluation/stage6a_headdev_importance_audit.json',{'status':'PASS','seed':2022,
        'head':'R2 fixed best','dataset':'HeadDev70 TRAIN scenes only','actors':len(data['base_logits']),
        'permutations_per_interaction_feature':1,'shuffle_unit':'actor; same actor permutation across all six mode values; same permutation for each feature',
        'baseline_Top1FDE':baseline['Top1FDE'],'baseline_rank_loss':baseline['rank_loss'],
        'VAL_used':False,'features_deleted':False,'retraining':False,'rows':rows})
    verify_frozen();print('STAGE6A_HEADDEV_IMPORTANCE_PASS',flush=True)

if __name__=='__main__':torch.set_num_threads(4);main()
