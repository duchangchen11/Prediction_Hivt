"""Read-only evidence QA and six Stage15A reports; no torch/forecast/optimizer."""
import ast
import csv
import hashlib
import json
import shutil
import statistics
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
BASE = "7b14f630fa42a4320214e8c6c63ab8794b48d43b"
BRANCH = "stage15a/isolated-predictor-preflight"
S14 = PROJECT / "outputs/stage14b_paper_validation"


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1048576), b""):
            h.update(b)
    return h.hexdigest()


def write(p, x):
    Path(p).write_text(json.dumps(x, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def table(rows):
    assert rows
    columns = list(rows[0])
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    lines += ["| " + " | ".join(str(row[c]) for c in columns) + " |" for row in rows]
    return "\n".join(lines)


def reference(relative, text):
    p = ROOT / relative
    lineno = next(i for i, line in enumerate(p.read_text().splitlines(), 1) if text in line)
    url = f"https://github.com/duchangchen11/Prediction_Hivt/blob/{BRANCH}/outputs/stage15a_isolated_predictor/{relative}#L{lineno}"
    return f"[{p.name}:{lineno}]({url})"


def main():
    protocol = read(ROOT / "00_manifest/stage15a_protocol.json")
    assert protocol["BaseCommit"] == BASE and not protocol["FullTrainingAuthorized"] and not protocol["OuterInferenceAuthorized"]
    frozen = read(ROOT / "00_manifest/stage15a_frozen_history.json")
    for key in ("historical_files", "preserved_untracked", "checkpoints"):
        for p, h in frozen[key].items():
            assert sha(PROJECT / p) == h, (key, p)
    for p, h in protocol["source_sha256"].items():
        assert sha(PROJECT / p) == h
    first = read(ROOT / "00_manifest/stage15a_code_registration.json")
    correction = read(ROOT / "00_manifest/stage15a_audit_correction_registration.json")
    for p, h in first["SourceSHA256"].items():
        if p == correction["ChangedSource"]:
            assert h == correction["OldSHA256"] == sha(ROOT / correction["ArchivedOriginal"])
            assert sha(ROOT / p) == correction["CurrentSHA256"]
        else:
            assert sha(ROOT / p) == h
    assert correction["TrainingSourceChanged"] is False and correction["DataOrProtocolChanged"] is False
    trained_reg = read(ROOT / "00_manifest/stage15a_trained_interface_registration.json")
    assert trained_reg["SourceSHA256"] == sha(ROOT / "03_checks/stage15a_trained_interface_check.py")
    checks, ranks, interfaces, isolation, resources = [], [], [], [], []
    cp = {}
    for fold in (1, 2, 3):
        d = read(ROOT / f"03_checks/stage15a_fold{fold}_checks.json")
        r = read(ROOT / f"05_candidate_interface/stage15a_fold{fold}_ranking_audit.json")
        ti = read(ROOT / f"03_checks/stage15a_fold{fold}_trained_interface.json")
        iso = read(ROOT / f"01_data_isolation/stage15a_fold{fold}_runtime_isolation.json")
        res = read(ROOT / f"06_resources/stage15a_fold{fold}_resources.json")
        assert all(v["Status"] == "PASS" for v in (d, r, ti, iso))
        assert d["Seed"] == protocol["folds"][fold - 1]["seed"] and d["Initialization"]["Parameters"] == 650403
        assert d["Initialization"]["LoadsAtInitialization"] == []
        assert max(d["ModelInputAndLoss"]["GTPoisonMaxDiff"].values()) == 0
        assert d["Recovery"]["ModelMaxDiff"] == d["Recovery"]["OptimizerMomentsMaxDiff"] == 0
        assert r["GTPoisonFeatureMaxDiff"] == ti["GTFeatureMaxDiff"] == 0
        assert ti["PredictorStateSHA256"] != d["Initialization"]["StateSHA256"]
        assert not iso["OuterPerformanceComputed"] and not iso["OfficialVALLoaded"] and not iso["HeadDevLoaded"]
        assert all(x["role"] == "InnerTrain" for x in iso["BatchRecords"] if x["purpose"] == "optimization")
        parts = protocol["folds"][fold - 1]["parts"]
        for record in iso["BatchRecords"] + ti["BatchRecords"]:
            assert set(record["scene_tokens"]) <= set(parts[record["role"]])
        assert not r["HistoricalHeadWeightsLoaded"] and not ti["HistoricalHeadWeightsLoaded"]
        assert r["HeadOptimizerUpdates"] == ti["HeadOptimizerUpdates"] == 0
        assert ti["OuterInferenceRuns"] == ti["PredictorOptimizerUpdates"] == 0
        for runtime in (iso, ti):
            for p, h in runtime["LoadedShardSHA256"].items():
                assert sha(PROJECT / p) == h
            loads = runtime.get("Loaded", runtime.get("LoadedPayloads", []))
            assert all(x["scope"] in ("graph", "preflight_checkpoint") for x in loads)
        training = read(ROOT / f"04_checkpoints/fold{fold}/stage15a_training_probe.json")
        assert training["SourceConfig"]["protocol_sha256"] == sha(ROOT / "00_manifest/stage15a_protocol.json")
        for p, h in training["SourceConfig"]["fitting_source_sha256"].items():
            assert sha(ROOT / p) == h
        for name, value in training["Checkpoints"].items():
            path = ROOT / f"04_checkpoints/fold{fold}" / name
            assert sha(path) == value["sha256"]
        path = ROOT / f"04_checkpoints/fold{fold}/stage15a_preflight_nll_best.pt"
        assert sha(path) == ti["NLLCheckpointSHA256"]
        cache = ROOT / f"05_candidate_interface/cache/fold{fold}/stage15a_trained_candidates.pt"
        assert sha(cache) == ti["CandidateCacheSHA256"]
        assert sha(ROOT / f"05_candidate_interface/stage15a_fold{fold}_trained_normalization.json") == ti["NormalizationSHA256"]
        for path in (ROOT / f"04_checkpoints/fold{fold}").glob("*.pt"):
            cp[str(path.relative_to(ROOT))] = {"SHA256": sha(path), "Bytes": path.stat().st_size,
                "Scope": "PREFLIGHT_ONLY_NOT_ELIGIBLE_FOR_FORMAL_TRAINING"}
        checks.append(d); ranks.append(r); interfaces.append(ti); isolation.append(iso); resources.append(res)
    assert len({d["Initialization"]["StateSHA256"] for d in checks}) == 3
    assert sum(x["ActualOptimizerUpdates"] for x in resources) == 30 <= protocol["preflight"]["maximum_optimizer_updates_all_checks"]
    assert read(ROOT / "07_logs/stage15a_fold2_timestamp_failure.json")["ActualOptimizerUpdates"] == 0
    assert read(ROOT / "02_training/stage15a_cli_smoke.json")["Status"] == "PASS"
    assert len(cp) == 12
    write(ROOT / "04_checkpoints/stage15a_preflight_checkpoint_manifest.json", {"Status": "PASS", "Checkpoints": cp,
        "FormalEligible": False, "FormalHiVTRuns": 0, "FutureStage15BMustStartFresh": True})
    fwd = [x["wall_seconds"] for r in resources for x in r["forward"]["Rows"]]
    training_times = [x["wall_seconds"] for r in resources for x in r["training_rows"]]
    bench = resources[0]["StepBenchmark"]
    means = [statistics.mean(x["wall_seconds"] for x in r["training_rows"]) for r in resources]
    stress = resources[1]["StressBatch"]
    assert stress["Status"] == "PASS" and stress["Windows"] == 16 and stress["Actors"] == 1633
    total = resources[1]["DeviceTotalBytes"]
    peak = max([x["peak_allocated_bytes"] for r in resources for x in r["training_rows"]] + [stress["peak_allocated_bytes"]])
    reserved = max([x["peak_reserved_bytes"] for r in resources for x in r["training_rows"]] + [stress["peak_reserved_bytes"]])
    assert peak < total and reserved < total
    cost = read(S14 / "00_protocol/stage14b_scene_design_cost_audit.json")
    hist_cost = read(S14 / "00_protocol/stage14b_updated_resource_estimate.json")["Schemes"][0]
    meta = read(PROJECT / "outputs/stage6a_future_interaction_reliability/01_cache/stage6a_cache_manifest.json")
    context700_bytes = sum(x["bytes"] for x in meta["batches"] if x["split"] == "train")
    train_index_windows = sum(int(x["windows"]) for x in meta["batches"] if x["split"] == "train")
    head630_windows = sum(v["windows"] for v in protocol["folds"][0]["metadata"].values() if v is not protocol["folds"][0]["metadata"]["QuarantinedHeadDev"])
    context_proxy = context700_bytes * (3 * head630_windows / train_index_windows)
    rows = cost["EstimatedMergedThreefoldTargetCacheRows"]
    assert rows == 780453
    target_cache = cost["EstimatedMergedAorBTargetCacheBytesAtUnchangedDtypeShapes"]
    r2_extra = rows * (6 * 19 * 4 + 3)
    identity_budget = rows * 256
    checkpoint_budget = .5 * 1024 ** 3
    scratch_budget = 2 * 1024 ** 3
    reserve_budget = 8 * 1024 ** 3
    planned_storage = target_cache + context_proxy + r2_extra + identity_budget + checkpoint_budget + scratch_budget
    free = shutil.disk_usage(ROOT).free
    storage_pass = free >= planned_storage + reserve_budget
    # Measured step extrapolation is shown beside, not substituted for, the
    # historical wall-time anchor. Neither includes a measured full cache build.
    low11500 = 3 * 11500 * min(means) / 3600
    high21000 = 3 * 21000 * max(means) / 3600
    dev_batches = sum(math_ceil(f["metadata"]["InnerDev"]["supervised_windows"], 16) for f in protocol["folds"])
    dev_forward_max_hours = 42 * dev_batches * statistics.mean(fwd) / 3600
    nominal_mib = int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"], text=True).strip().splitlines()[0])
    new = {"Status": "PASS" if storage_pass else "STORAGE_BLOCKED", "Device": resources[0]["Device"],
        "DeviceGiB": total / 1024 ** 3, "NominalDeviceGiB": nominal_mib / 1024,
        "MemoryScope": "torch device properties give CUDA-visible total; nvidia-smi gives nominal framebuffer total",
        "PeakAllocatedGiB": peak / 1024 ** 3, "PeakReservedGiB": reserved / 1024 ** 3,
        "StressWindows": 16, "StressActors": 1633, "ForwardMeanSecondsPer16WindowBatch": statistics.mean(fwd),
        "ForwardMedianSeconds": statistics.median(fwd), "TrainingStepMeanSeconds": statistics.mean(training_times),
        "TrainingStepMedianSeconds": statistics.median(training_times), "ColdAndWarmStepRangeSeconds": [min(training_times), max(training_times)],
        "TrainingStepBenchmarkMeanSeconds": statistics.mean(x["seconds"] for x in bench),
        "CheckpointMinBytes": min(x["Bytes"] for x in cp.values()), "CheckpointMaxBytes": max(x["Bytes"] for x in cp.values()),
        "PreflightCheckpointTotalBytes": sum(x["Bytes"] for x in cp.values()),
        "PreflightCandidateCacheTotalBytes": sum(p.stat().st_size for p in (ROOT / "05_candidate_interface/cache").rglob("*.pt")),
        "DiskFreeGiB": free / 1024 ** 3, "Formal3FoldTargetCacheGiB": target_cache / 1024 ** 3,
        "Formal3FoldContextCacheProxyGiB": context_proxy / 1024 ** 3, "FormalR2ExtraGiB": r2_extra / 1024 ** 3,
        "FormalIdentityBudgetGiB": identity_budget / 1024 ** 3, "FormalCheckpointBudgetGiB": .5, "FormalScratchBudgetGiB": 2,
        "PlannedNewStorageGiB": planned_storage / 1024 ** 3, "RequestedOperationalReserveGiB": 8,
        "StoragePassAtAuditTime": storage_pass, "SharedCandidatesAndBoundedStreamingRequired": True,
        "PredictorMeasuredStepExtrapolationHours": {"Historical11500LengthMinFoldMean": low11500, "Maximum21000LengthMaxFoldMean": high21000},
        "DevForwardKernelWallProxyAtMaxBudgetHours": dev_forward_max_hours,
        "HistoricalSchemeACompleteKnownComponentAnchorHours": [hist_cost["PlanningObservedScheduleAnchorHours"], hist_cost["PlanningFullPredictorBudgetAnchorHours"]],
        "MeasuredHistorical18HeadHours": hist_cost["Historical15HeadHours"] + hist_cost["MeasuredMatchedThreeHeadHours"],
        "EstimatedHiVTRuns": 3, "FutureRankingRuns": 18, "FormalHiVTRunsExecuted": 0,
        "ActualDiagnosticOptimizerUpdates": 30, "HeadOptimizerUpdates": 0,
        "Limits": "few fixed train/dev samples; elapsed step includes loading/hash/collation/H2D/backward/AdamW; forward excludes transfer/I/O; dense old-context sizes are metadata proxies, not new cache measurements; no per-method duplication and no permanent double dense staging"}
    write(ROOT / "06_resources/stage15a_resource_summary.json", new)
    lines = []
    for n, d, rr in zip((1, 2, 3), checks, resources):
        lines.append({"Fold": n, "Seed": d["Seed"], "TrainScenes": 378, "DevScenes": 42, "OuterExcluded": 210, "HeadDevExcluded": 70,
            "TrainWindows": protocol["folds"][n - 1]["metadata"]["InnerTrain"]["supervised_windows"],
            "DevWindows": protocol["folds"][n - 1]["metadata"]["InnerDev"]["supervised_windows"],
            "OptimizerChecks": rr["ActualOptimizerUpdates"], "ForwardMean_s": f"{statistics.mean(x['wall_seconds'] for x in rr['forward']['Rows']):.6f}",
            "StepMean_s": f"{statistics.mean(x['wall_seconds'] for x in rr['training_rows']):.6f}",
            "RecoveryMaxDiff": d["Recovery"]["ModelMaxDiff"]})
    with (ROOT / "06_resources/stage15a_resource_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(lines[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(lines)
    data_ref = reference("00_manifest/stage15a_common.py", "class FoldContext")
    batch_ref = reference("00_manifest/stage15a_common.py", "def check_batch")
    init_ref = reference("00_manifest/stage15a_common.py", "def model_new")
    input_ref = reference("00_manifest/stage15a_common.py", "def model_input")
    checkpoint_ref = reference("02_training/stage15a_train.py", "def checkpoint_save")
    restore_ref = reference("02_training/stage15a_train.py", "def restore")
    transition_ref = reference("02_training/stage15a_train.py", "def transition_from_own_warmup")
    loop_ref = reference("02_training/stage15a_train.py", "def train_registered_phase")
    entry_ref = reference("02_training/stage15a_train.py", "def parse_args")
    rank_ref = reference("05_candidate_interface/stage15a_ranking.py", "def pack_window")
    norm_ref = reference("05_candidate_interface/stage15a_ranking.py", "def fit_normalization")
    ref = "[冻结协议](00_manifest/stage15a_protocol.json)、[初始源码登记](00_manifest/stage15a_code_registration.json)、[CLI检查](02_training/stage15a_cli_smoke.json)"
    common = "本阶段仅工程 preflight；未启动完整三折预测器训练，未运行 OuterTest forward、性能评价或模型选择。30次更新均为带 `PREFLIGHT_ONLY_NOT_ELIGIBLE_FOR_FORMAL_TRAINING` 标记的检查，不可作为正式模型或下一阶段初始化。"
    reports = {}
    reports["stage15a_data_isolation_audit.md"] = f"""# Stage15A 场景隔离审计

SceneIsolation = PASS。方案A严格复用 Stage14B 的 scene token 与原顺序，每折378训练/42开发/210外层排除/70隔离，700场景全部对账；不采用448备选。{common}

{table(lines)}

显式列表见 [split integrity](01_data_isolation/stage15a_split_integrity.json) 和 `01_data_isolation/stage15a_foldN_{{InnerTrain,InnerDev,OuterTest,QuarantinedHeadDev}}.json`。列表 SHA、fold、seed、角色和完整顺序由 {data_ref} 校验。只读取原 `stage3_train_index.csv` 的元数据，不调用硬编码 train/val Dataset；训练与开发分别引用原378/42的shard，不复制原数据。

{batch_ref} 检查每个实际优化batch仅含InnerTrain；开发forward只允许InnerDev。Outer/HeadDev shard打开、Outer Dataset创建、Dev优化、VAL index读取、旧TRAIN700权重、错seed、交换列表、历史输出路径均有明确拒绝探针。失败探针在读入payload前被拒绝；runtime JSON 的 ReadPaths 包含打开尝试，必须结合 BlockedNegativeProbes 和 LoadedPayloads理解，不能当作成功读取。

运行来源与batch证据见 [Fold1](01_data_isolation/stage15a_fold1_runtime_isolation.json)、[Fold2](01_data_isolation/stage15a_fold2_runtime_isolation.json)、[Fold3](01_data_isolation/stage15a_fold3_runtime_isolation.json)。新NLL候选来源另见 `03_checks/stage15a_foldN_trained_interface.json`。所有加载的shard SHA与历史manifest一致。

原始scene/sample/instance、annotation链、17帧时间戳和ego/global坐标对账了每fold4个训练/开发窗口。SQL只查询明确选中的train/dev键；原数据库只读。缓存相对时间与源 timestamp 精确一致，坐标误差均<1e-4m。模型仍使用5历史/12未来关键帧和名义2s/6s定义，不插值或重采样。

Fold2初次审计在优化前被额外0.15s名义时长检查挡住；真实future末帧可为6.248027s且缓存与源完全一致。该门槛未在Stage14B协议登记，已移除并改为精确源时间戳匹配与真实jitter记录。保留 [失败证据](07_logs/stage15a_fold2_timestamp_failure.json)、[原审计源码](07_logs/stage15a_checks_initial_timestamp_audit.py)、[审计修正登记](00_manifest/stage15a_audit_correction_registration.json)。训练源码、超参数、原数据及Stage14B协议未修改，失败尝试优化更新0。

场景训练隔离的工程条件已通过。历史TRAIN/VAL用于方法开发这一事实保持，现有场景仍不能称为全新独立确认集。
"""
    reports["stage15a_training_interface_audit.md"] = f"""# Stage15A 训练入口审计

TrainingInterface = PASS。新增入口显式接收 `--fold --seed --training-scenes --development-scenes --output-dir`；三折seed固定2022/2122/2222。默认describe只核对配置与元数据，preflight最多运行已登记的小样本步骤，formal明确拒绝。CLI三折describe均exit0，formal拒绝均exit1且没有启动优化。代码：{entry_ref}。

冻结配置：batch16、FP32、AdamW，原 named-module decay/no_decay 分区分别wd0.0001/0；fixed-scale warm-up5000步/lr0.001，原NLL最多16000步/lr0.0001，总上限21000，开发每500步、NLL连续5次无改善停止。选择只用42开发scene的full-horizon Overall minFDE6，严格改善；最终只用NLL best。完整循环已实现并由Stage15A关闭的授权gate保护：{loop_ref}。本阶段没有改变500步验证间隔，也没有用早期小样本指标代替正式选择。

实测每fold4次warm+2次NLL primary检查，6次replay对照、5次训练计时和1次压力更新，合计30。开发4窗的probe在正式500步日程之外，checkpoint标记仅供检查；其 best_step=5002是NLL协议索引，不能解释为已执行5002更新。

{checkpoint_ref} 保存模型、AdamW、phase/严格best/坏验证次数、scene sampler epoch+cursor以及CPU/CUDA/Python/NumPy RNG；{restore_ref} 严格绑定fold、seed、列表SHA、输出目录和fitting源码SHA，禁止历史模型或其他fold替代。三折恢复后下一步模型、AdamW moments、loss/梯度、batch身份、游标和全部RNG逐位一致，maxdiff0。NLL checkpoint也完成严格load与候选forward核验。

{transition_ref} 从本实验自己的warm best恢复模型/优化器/RNG，仅将LR0.001改为0.0001，AdamW moments与其他group选项保持；NLL sampler游标重置，epoch偏移100000。原SceneSampler算法逐scene/窗shuffle复用；新按步重建loader只模拟原每新epoch一次CPU base-seed抽样，防止恢复时额外消耗RNG。确定性算法与 `CUBLAS_WORKSPACE_CONFIG=:4096:8`用于可复现检查，batch、损失、模型和预算保持原定义。

边界模拟验证严格tie不改善、NLL patience5、warm固定5000及21000总上限；这些模拟不训练。有限梯度、type embedding、所有expert/router参数在4步内都收到非零梯度。任何隔离、非有限或恢复失败即退出；没有历史模型填补、自动换seed/batch或额外训练预算。

{ref}。正式启动仍需大脑AI审查与单独授权；下一阶段必须登记自己的formal输出/源码并从随机初始化开始，不能打开本阶段gate后复用这些tiny权重。
"""
    reports["stage15a_model_integrity.md"] = f"""# Stage15A 模型与GT输入完整性

PredictorInitialization = PASS；GTLeakageAudit = PASS；CheckpointRecovery = PASS。实际模型仍为 Stage5A `HiVTMotionAwareDecoder`：原HiVT-64 local/global backbone、3×64类型embedding、两个64→16→64 motion residual experts、6→16→2 router、原K6 decoder与原pi评分。总参数和requires_grad参数均650403，无新增预测/语义/规划模块。{init_ref}复用原 fresh No-Type→Type→Motion构造及neutral初始化，fold1 seed2022与字面历史Stage5A fresh step0状态完全一致；三fold各自重复初始化一致且三seed状态不同。初始化没有任何checkpoint加载，shared state仅来自本次随机构造的内存对象。

{table([{'Fold': d['Fold'], 'Seed': d['Seed'], 'Parameters': d['Initialization']['Parameters'], 'FreshStateSHA256': d['Initialization']['StateSHA256']} for d in checks])}

{input_ref}实际构造只含历史观测、几何lane/actor边及类型的Data，positions/padding截为5历史帧，物理移除y、future_mask、target_mask、full_horizon_mask、ego_future及身份/未来标签。监督只在forward完成后进入原recovery_loss。完整原容器forward仅用于参考数值对照，实际入口始终调用allowlist。

普通参考与新allowlist的raw_prediction/mode_logits/mode_prob均maxdiff0；GT trajectory、y、未来padding/mask、target mask、ego future、未来time置NaN或翻转后实际forward仍maxdiff0。独立手工fixed-scale与原NLL loss重算均maxdiff0。Loss和原训练公式未改变，scale仍由原decoder预测。

raw为[K6,N,T12,4]，对应logits/prob为[N,6]；候选转换为[N,6,12,2]的t0-ego米坐标：local row vector@rotation^T+current_position。逐模式配对、prob归一化、显式坐标换算和原数据身份保持。恢复后原NLL checkpoint产生K6且与其state SHA绑定；没有用GT选择模型的推理输入或邻居。示例证据：[Fold1检查](03_checks/stage15a_fold1_checks.json)、[Fold2检查](03_checks/stage15a_fold2_checks.json)、[Fold3检查](03_checks/stage15a_fold3_checks.json)。

原mask监督与empty-window过滤完全保留；它们决定监督资格，不删除完整window中没有未来标签的当前参与者。历史5/12关键帧真实采样间隔存在jitter，审计按原timestamps精确对账而不修改数据。名义6s自定义协议不能据此直接等同官方固定时间点协议。

历史3206个受版本管理文件、5个Stage2C未提交文件、38个历史checkpoint逐项SHA保持。新12个checkpoint均仅供preflight，[清单](04_checkpoints/stage15a_preflight_checkpoint_manifest.json)记录SHA与约8.2MB大小，模型权重不上传Git。
"""
    reports["stage15a_ranking_compatibility.md"] = f"""# Stage15A 排序接口兼容性

RankingCompatibility = PASS，涵盖随机初始化及少量NLL更新后候选，使用新随机排序头，无历史排序头权重、无排序头optimizer更新。{rank_ref}先对全当前window按原GT-free Stage8特征生成节点/边，再按监督资格选择训练target；未按未来GT筛除邻居。

| 方法 | 参数量 | 输入/固定角色 |
| --- | --- | --- |
| R2 | 673 | [N,6,19]历史/候选/概率加权预测邻居特征 |
| NG-A / NG-C | 7425 | shared目标15维特征，仅目标自身路径 |
| G-A / G-C | 24066 | 节点[N,9,6,15]；边[N,6,8,6,17]；8邻居mask |
| Matched-NG-C | 24001 | 目标自身64→128→64 adapter，原capacity control结构 |

K6、t0≤50m/最多8邻居、独立6×6未来候选关系保持；原logits为独立[N,6] residual基准，节点column3原logit与同mode匹配。G1 map输入为None，实际结构没有语义/map message，不能加入新语义模块。R2/图特征函数与原Stage11B graph_normalize/A/C loss AST直接复用，避免导入历史stage写hook、旧缓存和全700预测器。

{norm_ref}只接受该fold InnerTrain，人口std(ddof0)+1e-6，图节点continuous3–14与真实边0–10按原valid mask；one-hot/flags不变，padding归零，原logits residual不归一化。R2缺失邻居distance归一化后设1、冲突设0的约定保留。Dev拟合norm明确拒绝。本阶段统计仅来自少量train窗并标formal_eligible=false，未来必须在新预测器冻结候选上重算整个378场景norm，不能使用这些检查统计或旧norm。

六头zero-init产生与新原logits完全一致的输出；非零随机评分fixture的共同mode置换前后maxdiff<1e-6，证明候选、原分数与边target/neighbor模式配对正确。fixture仅用于检查，没有改变历史模型。训练target上的原A/C或R2 loss有限且头反向有梯度，预测器没有梯度与状态变化；GT/未来mask扰动后全部graph/R2输入maxdiff0。

Bicycle未来统一路由到同fold重新训练R2；post-update接口检查实际Bicycle与固定路由fixture均通过。详见 [Fold1 trained接口](03_checks/stage15a_fold1_trained_interface.json)、[Fold2](03_checks/stage15a_fold2_trained_interface.json)、[Fold3](03_checks/stage15a_fold3_trained_interface.json)。候选payload无GT，标签另存sidecar；绑定scene/sample/instance、fold/seed/role、predictor SHA、历史mask、frame原点/方向、timestamp元数据。只保存少量train/dev候选，未生成Outer候选。

正式方案A每fold必须新训R2、NG-A、NG-C、G-A、G-C、Matched-NG-C，共18头；共同新预测器候选与mode顺序完全相同。不得直接复用旧头作为公平端到端模型，minFDE6仍是冻结共同候选的完整性指标。
"""
    reports["stage15a_resource_report.md"] = f"""# Stage15A GPU与存储资源

GPUResource = PASS；StorageResource = {'PASS_WITH_SHARED_STREAMING' if storage_pass else 'BLOCKED'}。设备{new['Device']}，标称{new['NominalDeviceGiB']:.2f}GiB、当前CUDA可见{new['DeviceGiB']:.3f}GiB，原torch/CUDA环境不升级。{common}

{table(lines)}

15次预先转移的eval forward（3fold×5）均值{new['ForwardMeanSecondsPer16WindowBatch']:.6f}s/16window batch，排除shard I/O、H2D和排序特征构造。18次primary完整train step均值{new['TrainingStepMeanSeconds']:.6f}s，中位{new['TrainingStepMedianSeconds']:.6f}s，范围{min(training_times):.6f}–{max(training_times):.6f}s；包含shard校验/读取、collation、H2D、forward、backward、AdamW及同步。额外5次step benchmark均值{new['TrainingStepBenchmarkMeanSeconds']:.6f}s，计入30次实际更新。

元数据选出的Fold2 InnerTrain最大16个actor-count窗口共1633演员，完整训练step成功，峰值allocated{new['PeakAllocatedGiB']:.3f}GiB、reserved{new['PeakReservedGiB']:.3f}GiB。该批次选择只依赖t0 actor_count，不读取GT error或Outer；它提供batch16的显存证据，不代表全程穷尽所有地图/边密度组合。

每个模型+AdamW+RNG检查checkpoint {new['CheckpointMinBytes']/1e6:.3f}–{new['CheckpointMaxBytes']/1e6:.3f}MB；12个共{new['PreflightCheckpointTotalBytes']/1e6:.3f}MB。实际小样本候选/label缓存总{new['PreflightCandidateCacheTotalBytes']/1e6:.3f}MB，均留本地不上传。详见 [原始计时与资源JSON](06_resources/stage15a_resource_summary.json)、[各fold汇总CSV](06_resources/stage15a_resource_summary.csv)。磁盘本次audit剩余{new['DiskFreeGiB']:.3f}GiB。

正式方案A为3次新HiVT＋18头；从本次不同fold少量step均值延伸，3×11500步的较短历史schedule代理约{low11500:.2f}小时，3×21000上限的较慢fold代理约{high21000:.2f}小时（预测器train step部分）。这些不是收敛承诺或时间置信区间。上限126次完整Dev42评价的forward-only代理{dev_forward_max_hours:.3f}小时，还要加开发shard I/O/保存。历史18头实测代理{new['MeasuredHistorical18HeadHours']:.3f}小时。

保留Stage14B完整已知组件墙钟预算{hist_cost['PlanningObservedScheduleAnchorHours']:.2f}–{hist_cost['PlanningFullPredictorBudgetAnchorHours']:.2f} GPU小时作为排程参考；未来新候选分布/early stop、缓存构造/I/O和诊断未完整计时，不能把少量forward推为总FPS或保证耗时。建议单RTX3080顺序运行，安排约1–2天设备可用窗口，仍只执行登记21000更新上限；不为排程增加任何模型训练预算。

三折合并630场景/780453 target occurrence dense目标缓存代理{target_cache/1024**3:.3f}GiB；加新R2约{r2_extra/1024**3:.3f}GiB、全当前上下文旧格式体积代理{context_proxy/1024**3:.3f}GiB、身份记录{identity_budget/1024**3:.3f}GiB、CP预算0.5GiB、bounded scratch2GiB，预计新增峰值{planned_storage/1024**3:.3f}GiB，另留8GiB运行余量。当前余量满足该设计。

存储结论以共同候选被六种方法复用、streaming写一次合并缓存、避免永久保留完整chunk和merged双副本为条件。原scene shard只引用；不删除历史。若未来按旧chunk+merged双套直接保留，将增加dense缓存近一倍，当前余量不能承诺。正式缓存实现和执行前磁盘空余需再次核对，本阶段没有构造大缓存。
"""
    ready = "ENGINEERING_PASS_PENDING_REVIEW_AND_AUTHORIZATION" if storage_pass else "NO_STORAGE_BLOCKED"
    reports["stage15a_final_report.md"] = f"""# Stage15A 最终报告

Stage15A工程检查完成。基础commit `{BASE}`，分支 `{BRANCH}`，新增内容全部位于本目录。{common}

| 项目 | 结论 |
| --- | --- |
| SceneIsolation | PASS：逐fold378/42/210/70，原scene token/顺序精确一致 |
| TrainingInterface | PASS：显式fold/seed/list/output；完整预算循环已实现，formal gate关闭 |
| PredictorInitialization | PASS：650403参数fresh Stage5A；fold1原step0完全一致，3seed不同 |
| GTLeakageAudit | PASS：未来GT物理剔除；扰动raw/logits/prob与graph/R2 feature maxdiff0 |
| RankingCompatibility | PASS：六种新随机头，初始与更新后NLL候选K6/15维节点/17维边/19维R2 |
| CheckpointRecovery | PASS：三fold下一步模型、AdamW、batch/cursor及四类RNG逐位一致；NLL strict-load通过 |
| GPUResource | PASS：最大16窗1633actor训练峰值allocated{new['PeakAllocatedGiB']:.3f}GiB，CUDA可见{new['DeviceGiB']:.3f}GiB |
| StorageResource | {'PASS_WITH_SHARED_STREAMING' if storage_pass else 'BLOCKED'}：free{new['DiskFreeGiB']:.2f}GiB，新增峰值设计{planned_storage/1024**3:.2f}GiB＋8GiB余量 |
| Stage15BReady | {ready} |

资源排程继续参考约8.41–13.70已知组件GPU小时；未来正式方案A需3次新预测器、18次新头，缓存构造与I/O另计。本阶段没有执行正式预测器训练或排序头拟合，不可将30次检查更新描述成已完成3个正式预测器。

保留一次Fold2优化前时间审计失败：本地额外0.15s名义时长门槛误拒原始timestamp jitter；已改为逐点源timestamp精确一致性，无数据/训练/超参数改变。修正登记与原源码/失败记录可复核。5/12帧的名义2/6s并非每条样本严格均匀时间；官方协议仍需单独token/坐标/指标/采样adapter与协议重置，不能提前宣称official兼容或全新独立确认。

全部历史文件/38 checkpoint/5未提交Stage2C文件SHA保持。新小样本CP、候选和标签缓存只保留本地；可版本管理源码、配置、split列表、日志证据及本六份报告提交推送，不merge main。

六份交付：[数据隔离](stage15a_data_isolation_audit.md)、[训练入口](stage15a_training_interface_audit.md)、[模型完整性](stage15a_model_integrity.md)、[排序兼容](stage15a_ranking_compatibility.md)、[资源](stage15a_resource_report.md)、本最终报告。执行/重放说明见 [README](README.md)；机器审核见 [final audit](00_manifest/stage15a_final_audit.json)。

Stage15BReady只表示当前工程检查通过。下一阶段仍等待大脑AI审查、单独正式训练授权及新阶段输出/源码登记；正式模型必须从fresh初始化开始，不能使用本阶段tiny CP/norm或旧TRAIN700/排序头。完成后STOP，未启动完整三折训练。
"""
    for name, value in reports.items():
        (ROOT / name).write_text(value)
    readme = """# Stage15A isolated predictor preflight

六份审计报告位于本目录根部。工程检查仅使用InnerTrain/InnerDev；本阶段formal入口关闭，所有checkpoint/norm/candidate标记preflight，不能复用为正式训练产物。

- [数据隔离](stage15a_data_isolation_audit.md)
- [训练入口](stage15a_training_interface_audit.md)
- [模型完整性](stage15a_model_integrity.md)
- [排序兼容](stage15a_ranking_compatibility.md)
- [资源报告](stage15a_resource_report.md)
- [最终报告](stage15a_final_report.md)

使用既有 `/home/lrj/anaconda3/envs/ped_intent/bin/python`，环境 `PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8`。不要创建新环境、升级依赖或重新预处理数据。

可只读复核并再生成报告：`python3 outputs/stage15a_isolated_predictor/08_reports/stage15a_finalize.py`。`stage15a_prepare.py`首次登记只运行一次，保留登记与审计修正记录。数据引用原Stage3 shard，SQLite仅查询少量明确train/dev键，原数据不复制。

入口示例（fold1）：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /home/lrj/anaconda3/envs/ped_intent/bin/python \
  outputs/stage15a_isolated_predictor/02_training/stage15a_train.py \
  --fold 1 --seed 2022 \
  --training-scenes outputs/stage15a_isolated_predictor/01_data_isolation/stage15a_fold1_InnerTrain.json \
  --development-scenes outputs/stage15a_isolated_predictor/01_data_isolation/stage15a_fold1_InnerDev.json \
  --output-dir outputs/stage15a_isolated_predictor/02_training/new_description/fold1 \
  --mode describe
```

全部small检查执行`03_checks/stage15a_checks.py`并将mode改为preflight；fold2/3的seed为2122/2222。为复现使用新的输出目录，不能覆盖已有04_checkpoints/foldN；候选缓存/最终汇总也应在独立副本中重放以保留当前冻结证据。正式fold训练仍被gate拒绝，不能因有训练代码就擅自启动。

诊断来源与命令见`07_logs/stage15a_corrected_process_receipts.json`、`stage15a_trained_interface_receipts.json`及`02_training/stage15a_cli_smoke.json`。Fold1首次执行通过，在shell工具运行，无subprocess计时receipt；其训练/forward/恢复时序与来源完整保存在fold1 JSON。Fold2初次timestamp审计失败及修正登记均保留。large checkpoint/array/log/cache不上传，small机器JSON和源码可复核。
"""
    (ROOT / "README.md").write_text(readme)
    audit = {"Status": "PASS", "SceneIsolation": "PASS", "TrainingInterface": "PASS", "PredictorInitialization": "PASS",
        "GTLeakageAudit": "PASS", "RankingCompatibility": "PASS", "CheckpointRecovery": "PASS", "GPUResource": "PASS",
        "StorageResource": "PASS_WITH_SHARED_STREAMING" if storage_pass else "BLOCKED", "Stage15BReady": ready,
        "BaseCommit": BASE, "Branch": BRANCH, "ProtocolSHA256": sha(ROOT / "00_manifest/stage15a_protocol.json"),
        "HistoricalTrackedFilesVerified": len(frozen["historical_files"]), "HistoricalCheckpointsVerified": len(frozen["checkpoints"]),
        "PreservedUntrackedVerified": len(frozen["preserved_untracked"]), "DiagnosticOptimizerUpdates": 30,
        "FormalPredictorRuns": 0, "RankingOptimizerUpdates": 0, "OuterInferenceRuns": 0, "OuterPerformanceComputed": False,
        "FailedInitialAudit": {"fold": 2, "optimizer_updates": 0, "audit_only_correction": True},
        "FittingSourcesUnchanged": True, "AuditCorrectionRegistrationSHA256": sha(ROOT / "00_manifest/stage15a_audit_correction_registration.json"),
        "PreflightCheckpointCount": 12, "PreflightCheckpointSHA256": {p: x["SHA256"] for p, x in cp.items()},
        "RequiredReportSHA256": {p: sha(ROOT / p) for p in reports}, "PristineIndependentConfirmation": False,
        "FullTrainingAuthorized": False, "STOP": True, "Stage15BStarted": False}
    write(ROOT / "00_manifest/stage15a_final_audit.json", audit)
    print("STAGE15A_FINALIZE_PASS", ready, "updates30", "formal0", "outer0")


def math_ceil(n, d):
    return (n + d - 1) // d


if __name__ == "__main__":
    main()
