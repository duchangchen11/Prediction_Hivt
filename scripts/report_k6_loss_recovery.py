"""Stage 2B report with explicit gates, skipped phases and subgroup results."""
import json

from preprocessing.common import PROJECT_ROOT, write_json
from scripts.k6_loss_recovery import ROOT, GROUPS, verify_previous


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def number(value):
    return "NOT_RUN" if value is None else f"{value:.6f}"


def main():
    a=read(ROOT/"A_fixed_scale/metrics.json")
    warm=read(ROOT/"B_warmup/metrics.json");nll=read(ROOT/"B_original_nll/metrics.json")
    mode=read(PROJECT_ROOT/"outputs/reports/k6_mode_audit.json")
    bounded={str(int(b)):read(ROOT/f"C_bmax{int(b)}/metrics.json") for b in (2.,4.,8.)}
    selected=read(ROOT/"selected_protocol.json")
    tiny=read(ROOT/"full_tiny_result.json");mini=read(ROOT/"mini_result.json")
    comparison=read(ROOT/"tiny_comparison.json")
    previous=read(ROOT/"frozen_previous_manifest.json")
    decision={"Stage2_baseline":"PASS" if selected and tiny and tiny["status"]=="PASS" and mini and mini["status"]=="COMPLETE" else "FAIL",
              "Allow_Stage3_discussion":"YES" if selected and tiny and tiny["status"]=="PASS" and mini and mini["status"]=="COMPLETE" else "NO",
              "Stage3_executed":False,
              "hard_stop":"K6_FIXED_SCALE FAIL" if a and a["status"]=="FAIL" else "Full Tiny FAIL" if tiny and tiny["status"]=="FAIL" else None}
    result={"A":a["status"] if a else "NOT_RUN","B_warmup":warm["status"] if warm else "NOT_RUN",
            "B_original_NLL":nll["status"] if nll else "NOT_RUN","C":{k:v["status"] if v else "NOT_RUN" for k,v in bounded.items()},
            "selected_protocol":selected,"Full_Tiny":tiny["status"] if tiny else "NOT_RUN","Mini":mini["status"] if mini else "NOT_RUN",
            "decision":decision,"original_commit":previous["commit"],"previous_artifacts_unchanged":True}
    write_json(PROJECT_ROOT/"outputs/reports/k6_loss_recovery_summary.json",result)
    lines=["# Stage 2B: K=6 Loss Recovery & Baseline Finalization", "",
           "工作目录 /home/lrj/Prediction_Hivt；分支 stage2/vehicle-hivt-baseline。复用 ped_intent，没有安装/升级依赖。",
           f"冻结前提交 `{previous['commit']}`；原 tiny、moving audit 和 moving_* 报告共 {len(previous['files'])} 个原文件 SHA256 保持相同。新结果仅写入 k6_loss_recovery/ 与 k6_* 报告。",
           "K=6、Th=5、Tf=12、约2s/6s、HiVT-64、原 encoder/global interactor/MLP decoder、dropout=.1、weight_decay=1e-4、地图半径50m/间距2m 均保持不变。不删除 stopped/parked，不重采样，不加入创新模块。", "",
           "## 【A K6 Fixed Scale】", ""]
    if a:
        f=a["final"]
        lines += [f"epochs={a['epochs']}；ADE={f['ADE']:.6f}m；FDE={f['FDE']:.6f}m；MR={f['MR']:.6f}；**{a['status']}**。",
                  "同一 scene-0655 moving actor，GT endpoint 49.5348m；51 vehicle nodes + 666 lane segments 全部保留，只有该车参与监督。",
                  "regression=mean(|y−μ|)，保留原 best-sum-L2 mode selection 和 detached-soft-target mode classification。b=1，regression 不含 log(2) 常数；NLL 另记录。固定 lr=.001，seed2022，上限1000，每10 epoch评估。严格 gate ADE<.5/FDE<1，未放宽。",
                  f"初始 ADE/FDE={a['initial']['ADE']:.6f}/{a['initial']['FDE']:.6f}m；初始模型 SHA256={a['initial_state_sha256']}。",
                  f"最后 train backward head gradients：loc={a['gradient_last']['loc_grad_norm']:.6g}，scale={a['gradient_last']['scale_grad_norm']:.6g}，pi={a['gradient_last']['pi_grad_norm']:.6g}。",
                  "![A predictions](../stage2/k6_loss_recovery/A_fixed_scale/predictions.png)", ""]
    else:lines += ["NOT_RUN", ""]
    lines += ["## 【Mode Audit】", ""]
    if mode:
        lines += ["| Mode | Endpoint local xy (m) | Endpoint ego xy (m) | Displacement (m) | Probability | sum L2 (m) | FDE (m) |", "|---|---|---|---:|---:|---:|---:|"]
        for m in mode["modes"]:
            fmt=lambda xy:"/".join(f"{v:.6f}" for v in xy)
            lines.append(f"| {m['mode']} | {fmt(m['endpoint_local_xy_m'])} | {fmt(m['endpoint_ego_xy_m'])} | {m['endpoint_displacement_m']:.6f} | {m['probability']:.6f} | {m['regression_sum_L2_m']:.6f} | {m['final_FDE_m']:.6f} |")
        lines += ["",f"best training mode={mode['best_training_mode']}；best FDE mode={mode['best_FDE_mode']}；一致={mode['best_modes_agree']}。",
                  f"MODE_COLLAPSE={mode['MODE_COLLAPSE']}；最大 mode pairwise trajectory distance={mode['max_pairwise_trajectory_distance_m']:.6g}m；预先定义 collapse tolerance=.001m。单目标 collapse 可以接受，没有因此修改模型。",
                  "全部6个 mode 的完整12步 local/ego trajectory 与 probability 保存于 k6_mode_audit.json。", ""]
    else:lines += ["NOT_RUN：A 尚未 PASS 或失败。", ""]
    lines += ["## 【B Warm-up】", ""]
    if warm:
        f=warm["final"]
        lines += [f"warm-up epochs={warm['epochs']}；ADE={f['ADE']:.6f}m；FDE={f['FDE']:.6f}m；{warm['status']}。",
                  "B 从 fresh seed2022 初始化，未使用 A final checkpoint。fixed-scale + classification、LR=.001，gate ADE<1/FDE<2，上限700。",
                  f"raw pre-ELU scale head mean/min/max={f['raw_scale_head_mean']:.6f}/{f['raw_scale_head_min']:.6f}/{f['raw_scale_head_max']:.6f}；processed raw scale mean/max={f['raw_processed_scale_mean']:.6f}/{f['raw_processed_scale_max']:.6f}。该分支此时不参与 regression。",
                  "warmup_checkpoint.pt 保存 weights、AdamW state 及 Torch CPU/CUDA RNG。", ""]
    else:lines += ["NOT_RUN", ""]
    lines += ["## 【Original NLL restored】", ""]
    if nll:
        f=nll["final"]
        lines += [f"epochs={nll['epochs']}；ADE={f['ADE']:.6f}m；FDE={f['FDE']:.6f}m；MR={f['MR']:.6f}；NLL={f['NLL']:.6f}；scale mean/max={f['scale_mean']:.6f}/{f['scale_max']:.6f}；**{nll['status']}**。",
                  "从 B warmup_checkpoint 继续，保留 AdamW state，LR降至1e-4，不用 scheduler，运行完整300 epochs。恢复原 free-scale LaplaceNLL 与原 mode classification。",
                  "最终 gate 预注册为 ADE<1/FDE<2，且相对 warm-up 的 ADE 增量≤.5m、FDE 增量≤1m；finite loss/gradients 必须通过。该增量定义在任何 B 结果前写入配置，用于量化‘不能明显退化’。", ""]
    else:lines += ["NOT_RUN", ""]
    lines += ["## 【C bounded scale】", ""]
    if any(bounded.values()):
        lines += ["仅 A PASS、B FAIL 后运行；fresh seed2022；除 b_max=2/4/8 外完全相同配置，LR=.001，每个最多700。b=clamp(raw b,min=.001,max=b_max)。采用与 A 相同的单移动 gate ADE<.5/FDE<1。", "",
                  "| b_max | Epochs | ADE | FDE | MR | NLL | scale mean/max | Status |", "|---|---:|---:|---:|---:|---:|---|---|"]
        for b,r in bounded.items():
            if r:
                f=r["final"];lines.append(f"| {b} | {r['epochs']} | {f['ADE']:.6f} | {f['FDE']:.6f} | {f['MR']:.6f} | {f['NLL']:.6f} | {f['scale_mean']:.6f}/{f['scale_max']:.6f} | {r['status']} |")
    else:lines += ["NOT_RUN：只有 A PASS 且 B FAIL 才允许执行。"]
    lines += ["", "## 【Selected Protocol】", ""]
    if selected:lines += [f"protocol={selected['protocol']}；name={selected['model_name']}。",selected["reason"],"architecture unchanged；保留多模态与 mode probability。", ""]
    else:lines += ["NOT_SELECTED", ""]
    lines += ["## 【Full Tiny】", ""]
    if tiny:
        lines += [f"status={tiny['status']}；原16 tiny windows、原 scene/sample tokens 和原所有 vehicle target masks，未更改数据。完整未来作为主要指标，partial 单独报告。", "",
                  "| Group | Count | ADE | FDE | MR |", "|---|---:|---:|---:|---:|"]
        for group,v in tiny["final"]["metrics"]["full_horizon"].items():
            lines.append(f"| {group} | {v['count']} | {number(v['minADE'])} | {number(v['minFDE'])} | {number(v['MR'])} |")
        lines += ["",f"预先固定的 gate：{tiny['gate']}。最终 epoch 验收，不以最佳中间 epoch 替代。",
                  "Attributes 使用真实 t0 annotation，不能用未来位移重贴 moving/stopped/parked 标签。Stopped 之后起步的车辆保留在 stopped 组。",
                  "minADE 使用最低FDE的mode（上游HiVT评估约定），independent_minADE另存JSON；MR=末端误差>2m。Actor-window等权平均，重叠窗口不等于独立车辆。", "",
                  "Partial future 单独统计，不混入约6s主指标：", "",
                  "| Group | Count | ADE | FDE | MR |", "|---|---:|---:|---:|---:|"]
        for group,v in tiny["final"]["metrics"]["partial_future"].items():
            lines.append(f"| {group} | {v['count']} | {number(v['minADE'])} | {number(v['minFDE'])} | {number(v['MR'])} |")
        lines.append("")
    else:lines += ["NOT_RUN：只有最终 K=6 protocol 确定后允许重跑。", ""]
    lines += ["## 【CV comparison】", ""]
    if comparison:
        lines += ["所有方法使用相同原16 tiny windows、相同 eligible masks、相同 t0 attributes；CV 使用最后两个有效 past observations 和真实 timestamps。", "",
                  "| Method | Overall ADE/FDE | Moving ADE/FDE | Stopped ADE/FDE | Parked ADE/FDE |", "|---|---|---|---|---|"]
        for name,values in comparison.items():
            row=[name]
            for group in GROUPS[:4]:
                v=values["metrics"]["full_horizon"][group];row.append(number(v["minADE"])+"/"+number(v["minFDE"]))
            lines.append("| "+" | ".join(row)+" |")
    else:lines += ["NOT_RUN", ""]
    lines += ["", "## 【Mini】", ""]
    if mini:
        lines += [f"status={mini['status']}；既有 mini train scenes / val scenes。Test 不用于调参或 checkpoint selection。",
                  f"Fresh seed2022；warm-up {mini['warmup_epochs']} epochs + 原 NLL {mini['NLL_epochs']} epochs。固定LR .001/.0001，无scheduler。最终NLL阶段按val full-horizon overall minFDE选择 epoch {mini['selected_epoch']}，不选择 warm-up 或 test checkpoint。",
                  "沿用 Stage 1 mini 自定义6/2/2 scene划分，不是官方nuScenes leaderboard；完成工程 baseline 不等于验证模型泛化优于CV。", "",
                  "| Method | Group | Count | ADE | FDE | MR |", "|---|---|---:|---:|---:|---:|"]
        for method in ("CV","HiVT"):
            for group,v in mini[method]["metrics"]["full_horizon"].items():
                lines.append(f"| {method} | {group} | {v['count']} | {number(v['minADE'])} | {number(v['minFDE'])} | {number(v['MR'])} |")
        lines += ["", "Partial future：", "", "| Method | Group | Count | ADE | FDE | MR |", "|---|---|---:|---:|---:|---:|"]
        for method in ("CV","HiVT"):
            for group,v in mini[method]["metrics"]["partial_future"].items():
                lines.append(f"| {method} | {group} | {v['count']} | {number(v['minADE'])} | {number(v['minFDE'])} | {number(v['MR'])} |")
        lines += ["",f"可视化覆盖：{mini['visualization_coverage']}；完整={mini['visualizations_complete']}。GT≥5m的真实 moving 目标，成功FDE≤2m、失败>2m；优先不同instance。图中lane/history/GT/6 modes/best-FDE/probabilities俱全。",
                  "[可视化清单与每个case的6条完整轨迹](../stage2/k6_loss_recovery/mini_visualization_audit.json)", ""]
    else:lines += ["NOT_RUN：Full Tiny 未 PASS 时禁止启动。"]
    lines += ["", "## 【Decision】", "", f"Stage2 baseline={decision['Stage2_baseline']}；Allow Stage3 discussion={decision['Allow_Stage3_discussion']}。本轮没有执行 Stage 3，也不 merge main。",
              f"Stop reason={decision['hard_stop']}。", "", "命令记录见 docs/stage2b_execution_commands.md；各实验 config/metrics/curve/gradient logs/figures 全部保留。Weights、optimizer states 和完整冻结副本仅保存在本机。"]
    (PROJECT_ROOT/"outputs/reports/k6_loss_recovery_report.md").write_text("\n".join(lines)+"\n")
    verify_previous();print(json.dumps(result,indent=2,ensure_ascii=False),flush=True)


if __name__=="__main__":main()
