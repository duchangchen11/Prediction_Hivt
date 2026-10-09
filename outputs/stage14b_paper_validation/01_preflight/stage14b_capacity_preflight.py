"""Real InnerTrain feature provenance/poison checks and public gradient guard."""
from pathlib import Path
import copy
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "00_protocol"))
from stage14b_common import *

sys.path.insert(0, str(S8 / "00_manifest"))
from stage8a_graph import observable_window, node_features, neighbors, interaction_edges


def gradients(model, predictor):
    """Original finite-gradient guard, plus exact active-module membership."""
    assert sum(parameter.numel() for parameter in model.parameters()) == 24001
    allowed = ("node_encoder.", "norm.", "head.", "own_adapter.")
    assert all(name.startswith(allowed) for name, _ in model.named_parameters())
    values = [parameter.grad for parameter in model.parameters()]
    assert values and all(value is not None and torch.isfinite(value).all() for value in values)
    norm = float(torch.stack([value.square().sum() for value in values]).sum().sqrt())
    assert norm > 0
    assert not predictor.training and not any(parameter.requires_grad or parameter.grad is not None for parameter in predictor.parameters())
    return norm


def raw_graph(window, targets):
    obs = observable_window(window, "", np.zeros(2), 0.)
    nodes = node_features(obs)
    idx, keep = neighbors(obs)
    edge = interaction_edges(obs, idx, keep, targets)
    return torch.cat((nodes[targets, None], nodes[idx[targets]]), 1), edge, keep[targets], window["mode_logits"][targets]


@torch.no_grad()
def main():
    seed()
    assert REG.exists(), "registration must be frozen before real preflight"
    verify(history=True)
    structure = read_json(ROOT / "01_preflight/stage14b_model_integrity.json")
    assert structure["Status"] == "PASS" and structure["OwnAdapterParameters"] == 16576
    historical = read_json(S14A / "01_preflight/stage14a_input_loss_integrity.json")
    records = pd.read_csv(S14A / "01_preflight/stage14a_gt_poison_windows.csv")
    assert historical["Status"] == "PASS" and historical["GTpoisonWindows"] == len(records) == 10
    f = frame()
    allowed = set(split(1)["InnerTrain"])
    lookup = {(row.scene_token, row.sample_token, row.instance_token): index for index, row in enumerate(f.itertuples())}
    store = Store(1)
    candidate_path = S11A / "01_identity_audit/cache/candidates.npy"
    candidates = np.load(candidate_path, mmap_mode="r")
    candidate_digest = sha256(candidate_path)
    model = fresh(1, "cpu").eval()
    generator = torch.Generator().manual_seed(2022 + 1401)
    model.head[-1].weight.copy_(.02 * torch.randn(model.head[-1].weight.shape, generator=generator))
    model.head[-1].bias.copy_(.005 * torch.randn(model.head[-1].bias.shape, generator=generator))
    checked = []
    for record in read_json(S6 / "01_cache/stage6a_cache_manifest.json")["batches"]:
        if record["split"] != "train": continue
        if "scene_tokens" in record and not allowed.intersection(record["scene_tokens"]): continue
        assert sha256(S6 / record["path"]) == record["sha256"]
        block = torch.load(S6 / record["path"], map_location="cpu", weights_only=False)
        for window in block["windows"]:
            if window["scene_token"] not in allowed: continue
            selected = [(node, lookup[(window["scene_token"], window["sample_token"], instance)])
                        for node, instance in enumerate(window["instance_tokens"])
                        if (window["scene_token"], window["sample_token"], instance) in lookup]
            if not selected: continue
            targets = torch.tensor([node for node, _ in selected])
            ids = np.array([index for _, index in selected])
            assert np.isin(ids, indices(1, "InnerTrain")).all()
            graph = raw_graph(window, targets)
            poisoned = copy.deepcopy(window)
            poisoned["GT"].fill_(float("nan"))
            poisoned["future_mask"] = ~poisoned["future_mask"]
            poisoned["target_mask"] = ~poisoned["target_mask"]
            other = raw_graph(poisoned, targets)
            assert all(torch.equal(a, b) for a, b in zip(graph, other))
            source = store.source[ids]
            assert np.array_equal(candidates[source], window["ego_prediction"][targets].numpy())
            for index in range(3):
                assert np.array_equal(store.args[index][source], graph[index].numpy())
            args, fde, ade = store.batch(ids, "cpu", part="InnerTrain")
            output = model(*args)
            node, edge, keep = graph_normalize(*other[:3], store.norm["graph"])
            again = model(node, edge, keep, None, None, None, other[3])
            assert all(torch.equal(output[key], again[key]) for key in output)
            altered_labels = fde.flip(-1)
            loss_change = float((objective(output["mode_logits"], fde, "C") -
                                 objective(output["mode_logits"], altered_labels, "C")).abs().max())
            assert loss_change > 0
            checked.append(dict(SceneToken=window["scene_token"], SampleToken=window["sample_token"], Actors=len(ids),
                                GraphPoisonMaxDiff=0., MatchedNoGraphPoisonMaxDiff=0., CandidateCoordinateMaxDiff=0.,
                                SeparateLabelPoisonLossChange=loss_change,
                                RawPredictionCacheSHA256=record["sha256"]))
            if len(checked) == 10: break
        if len(checked) == 10: break
    assert len(checked) == 10
    assert [(row["SceneToken"], row["SampleToken"]) for row in checked] == list(zip(records.SceneToken, records.SampleToken))
    initial = []
    for fold in (1, 2, 3):
        model = fresh(fold, "cpu")
        original = SparseGraphReranker("G1", 2022 + 100 * (fold - 1))
        assert all(torch.equal(value, original.state_dict()[name]) for name, value in model.state_dict().items()
                   if not name.startswith("own_adapter."))
        assert all(parameter.requires_grad for parameter in model.parameters())
        initial.append(dict(Fold=fold, Seed=2022 + 100 * (fold - 1), StateSHA256=state_sha(model),
                            SharedG1Initialization="BITWISE_EQUAL", Params=24001, ActiveAdapterParams=16576))
    with torch.enable_grad():
        logits = torch.tensor([[.1, .2, .3, .4, .5, .6], [1., 2., 3., 4., 5., 6.]], requires_grad=True)
        errors = torch.tensor([[1., 1., 2., 3., 4., 5.], [2., 3., 4., 5., 6., 7.]], requires_grad=True)
        costs = errors.detach() - errors.detach().min(-1, keepdim=True).values
        expected = (logits.softmax(-1) * (costs / costs.mean(-1).clamp(min=1.)[:, None])).sum(-1)
        loss = objective(logits, errors, "C")
        assert torch.equal(loss, expected)
        assert torch.autograd.grad(loss.sum(), errors, allow_unused=True, retain_graph=True)[0] is None
    assert sha256(candidate_path) == candidate_digest
    dump("01_preflight/stage14b_initialization.csv", initial)
    dump("01_preflight/stage14b_gt_poison_windows.csv", checked)
    atomic_json(ROOT / "01_preflight/stage14b_capacity_preflight.json", dict(
        Status="PASS", Structure="PASS", TotalParameters=24001, TrainableParameters=24001,
        G1Parameters=24066, AdapterParameters=16576, AddedParametersAllActive=True,
        CapacityDifferenceFraction=65 / 24066, ArchitectureSearchPerformed=False,
        SourceHistoricalInputAuditSHA256=sha256(S14A / "01_preflight/stage14a_input_loss_integrity.json"),
        HistoricalRawGTPoisonRecordsReused=10, CurrentRawGTPoisonWindows=10,
        PoisonScope="Fold1 InnerTrain only; raw GT NaN and masks inverted; nonzero audit-only final score head",
        OriginalCachedInputArrays="BITWISE_EQUAL", CurrentGTOrCandidateDrivenInputChanges=False,
        MatchedNoGraphRawGTPoisonMaxDiff=0., ForwardLabelsNeverPassed=True,
        SeparateLabelPoisonChangesLoss=True, LabelsDetached=True,
        LossDefinition="exact Stage11B normalized C AST, same max(1m,mean(regret)) floor; closed form byte equal",
        OwnNodesOnly=True, NeighborInputAblatedOrCorruptedForModel=False,
        CandidateIdentity="PASS", CandidateCoordinateMaxDiff=0., CandidateSHA256=candidate_digest,
        SharedG1InitialState="BITWISE_EQUAL", CurrentOuterTestOrHeadDevOrVALUsed=False,
        FormalTrainingPermitted=False, NextGate="original128target300update normalizedC tiny PASS required",
        SharedScalarBiasNote=structure["SharedScoreBiasNote"],
        StructuralModelAuditSHA256=sha256(ROOT / "01_preflight/stage14b_model_integrity.json")))
    verify(history=True)
    print("STAGE14B_CAPACITY_REAL_INPUT_PREFLIGHT_PASS", flush=True)


if __name__ == "__main__": main()
