"""Only the original128target/300update normalized-C gate; no formal runs."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "00_protocol"))
from stage14b_common import *
from stage14b_capacity_preflight import gradients


def main():
    seed()
    assert REG.exists()
    verify()
    capacity = read_json(ROOT / "01_preflight/stage14b_capacity_preflight.json")
    assert capacity["Status"] == "PASS" and capacity["AdapterParameters"] == 16576
    dest = ROOT / "01_preflight/stage14b_tiny_audit.json"
    assert not dest.exists(), "preserve completed tiny attempt; no tuning/retry after scientific failure"
    historical = read_json(S11B / "01_preflight/stage11b_tiny_targets.json")
    ids = np.array(historical["local_indices"], np.int64)
    assert np.array_equal(ids, np.random.default_rng(2022).choice(indices(1, "InnerTrain"), 128, replace=False))
    assert historical["fold"] == 1 and historical["partition"] == "InnerTrain"
    atomic_json(ROOT / "01_preflight/stage14b_tiny_targets.json", historical)
    store = Store(1)
    args, fde, ade = store.batch(ids, part="InnerTrain")
    assert not fde.requires_grad and not ade.requires_grad
    assert not any(value.requires_grad for value in args if value is not None)
    candidate_path = S11A / "01_identity_audit/cache/candidates.npy"
    candidates_before = sha256(candidate_path)
    predictor = frozen_predictor("cpu")
    predictor_before = state_sha(predictor)
    seed(2022)
    model = fresh(1)
    initial_state = state_sha(model)
    expected = pd.read_csv(ROOT / "01_preflight/stage14b_initialization.csv")
    assert initial_state == expected.loc[expected.Fold == 1].iloc[0].StateSHA256
    model.eval()
    with torch.no_grad():
        initial_output = model(*args)
        assert torch.equal(initial_output["mode_logits"], args[6])
        initial = float(objective(initial_output["mode_logits"], fde, "C").mean())
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    assert {id(value) for group in optimizer.param_groups for value in group["params"]} == {id(value) for value in model.parameters()}
    assert all(id(value) not in {id(parameter) for parameter in predictor.parameters()}
               for group in optimizer.param_groups for value in group["params"])
    curve = []
    gradient_norms = {name: 0. for name, _ in model.named_parameters()}
    started = time.monotonic()
    model.train()
    for update in range(1, 301):
        optimizer.zero_grad(set_to_none=True)
        output = model(*args)
        loss = objective(output["mode_logits"], fde, "C").mean()
        assert torch.isfinite(loss)
        loss.backward()
        norm = gradients(model, predictor)
        for name, parameter in model.named_parameters():
            gradient_norms[name] = max(gradient_norms[name], float(parameter.grad.norm()))
        optimizer.step()
        assert torch.isfinite(output["mode_prob"]).all()
        assert torch.allclose(output["mode_prob"].sum(-1), torch.ones(128, device="cuda"), atol=1e-6, rtol=0.)
        if update % 10 == 0:
            curve.append(dict(Model="Matched-NG-C", Update=update, Loss=float(loss.detach()), GradientNorm=norm,
                              AdapterGradientNorm=float(torch.stack([parameter.grad.square().sum() for parameter in model.own_adapter.parameters()]).sum().sqrt())))
    model.eval()
    with torch.no_grad():
        final = float(objective(model(*args)["mode_logits"], fde, "C").mean())
    reduction = 1 - final / initial
    passed = reduction >= .8
    adapter_norms = {name: value for name, value in gradient_norms.items() if name.startswith("own_adapter.")}
    assert len(adapter_norms) == 4 and all(value > 1e-9 for value in adapter_norms.values())
    assert state_sha(predictor) == predictor_before and sha256(candidate_path) == candidates_before
    with torch.no_grad():
        output = model(*args)
        poisoned_labels = fde.flip(-1)
        repeated = model(*args)
        assert all(torch.equal(output[key], repeated[key]) for key in output)
        poisoned_loss_change = float((objective(output["mode_logits"], fde, "C") -
                                      objective(output["mode_logits"], poisoned_labels, "C")).abs().max())
        assert poisoned_loss_change > 0
        neighbor_altered = tuple(value.clone() if value is not None else None for value in args)
        neighbor_altered[0][:, 1:] = neighbor_altered[0][:, 1:].flip(0)
        neighbor_altered[1].mul_(1.5)
        different_neighbors = model(*neighbor_altered)
        assert all(torch.equal(output[key], different_neighbors[key]) for key in output)
    result = dict(Model="Matched-NG-C", Status="PASS" if passed else "FAIL", Updates=300,
                  InitialLoss=initial, FinalLoss=final, Reduction=reduction, Threshold=.8,
                  PredictorGradientCount=0, FinalGradientNorm=norm, InitialStateSHA256=initial_state,
                  FullTrainingUsesTinyWeights=False, TotalParameters=24001, TrainableParameters=24001,
                  ActiveAdapterParameters=16576, Seconds=time.monotonic() - started)
    dump("01_preflight/stage14b_tiny_curve.csv", curve)
    dump("01_preflight/stage14b_tiny_results.csv", [result])
    dump("01_preflight/stage14b_tiny_gradient_parameters.csv", [
        dict(Parameter=name, MaximumGradientNorm=value, ActiveAdapter=name.startswith("own_adapter."),
             SharedSoftmaxOffset=name == "head.2.bias") for name, value in gradient_norms.items()])
    atomic_json(dest, dict(Status="PASS" if passed else "FAIL", FormalTrainingPermitted=passed,
                          Model="Matched-NG-C", Result=result, SameTargetsAndUpdatesAsStage11B=True,
                          LossAndLabels="exact original normalized C", AdapterTensorsHaveRealNonzeroGradient=adapter_norms,
                          ForwardGradientParameters=gradient_norms, SharedScoreBiasNote=capacity["SharedScalarBiasNote"],
                          PredictorGradientCount=0, PredictorStateUnchanged=True, CandidateIdentity="PASS",
                          CandidateCoordinateMaxDiff=0., PostTinyNeighborInputChangeMaxDiff=0.,
                          PostTinyLabelPoisonForwardMaxDiff=0., PostTinyLabelPoisonSeparateLossChange=poisoned_loss_change,
                          NoTinyCheckpointCreated=True, FullTrainingUsesTinyWeights=False,
                          FailureAction="STOP_WITHOUT_TUNING_STRUCTURAL_CHANGE_OR_FORMAL_TRAINING",
                          CapacityPreflightSHA256=sha256(ROOT / "01_preflight/stage14b_capacity_preflight.json"),
                          ProtocolSHA256=sha256(PROTOCOL), RegistrationSHA256=sha256(REG)))
    verify(history=True)
    print("STAGE14B_MATCHED_C_TINY", result, flush=True)
    if not passed:
        print("STOP_STAGE14B_TINY_GATE_FAILED", flush=True)
        return
    print("STAGE14B_TINY_PASS_FORMAL_GATE_READY", flush=True)


if __name__ == "__main__": main()
