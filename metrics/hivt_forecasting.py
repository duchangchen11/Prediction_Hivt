"""HiVT/Argoverse best-FDE-mode metrics with full/partial horizons separated."""
import torch

MR_THRESHOLD = 2.0


def multimodal_errors(prediction, target, mask, target_mask):
    if prediction.ndim != 4 or prediction.shape[2:] != target.shape[1:] or prediction.shape[0] != target.shape[0]:
        raise ValueError("Prediction [N,K,T,2], target [N,T,2] required")
    if mask.shape != target.shape[:2] or mask.dtype != torch.bool:
        raise ValueError("Boolean future mask [N,T] required")
    n, k, t, _ = prediction.shape
    if not torch.isfinite(prediction).all() or not torch.isfinite(target[mask]).all():
        raise ValueError("Non-finite predictions / valid future targets")
    errors = prediction.new_zeros((n, k, t))
    expanded = mask[:, None].expand(n, k, t)
    actual = target[:, None].expand_as(prediction)
    errors[expanded] = torch.linalg.vector_norm(prediction[expanded]-actual[expanded], dim=-1)
    valid_count = mask.sum(dim=-1)
    last = torch.where(mask, torch.arange(t, device=mask.device), -1).max(dim=-1).values
    fde_by_mode = errors.gather(2, last.clamp(min=0)[:, None, None].expand(n, k, 1)).squeeze(-1)
    best = fde_by_mode.argmin(dim=1)
    ade_by_mode = errors.sum(dim=-1)/valid_count.clamp(min=1)[:, None]
    rows = torch.arange(n, device=mask.device)
    min_fde = fde_by_mode[rows, best]
    return {"minADE_K": ade_by_mode[rows, best], "minFDE_K": min_fde, "MR_K": (min_fde > MR_THRESHOLD).float(),
            "independent_minADE_K": ade_by_mode.min(dim=1).values, "best_mode": best,
            "full_horizon": target_mask & mask.all(dim=1),
            "partial_future": target_mask & (valid_count > 0) & ~mask.all(dim=1),
            "valid_steps": valid_count, "last_valid_index": last}


def metric_records(errors):
    result = []
    for partition in ("full_horizon", "partial_future"):
        indices = torch.where(errors[partition])[0].tolist()
        for index in indices:
            result.append({"horizon": partition, "node": index,
                           **{key: float(errors[key][index]) for key in ("minADE_K", "minFDE_K", "MR_K", "independent_minADE_K")}})
    return result


def aggregate(records):
    result = {}
    for horizon in ("full_horizon", "partial_future"):
        eligible = [r for r in records if r["horizon"] == horizon]
        result[horizon] = {"count": len(eligible), **{name: sum(r[name] for r in eligible)/len(eligible) if eligible else None
                                                for name in ("minADE_K", "minFDE_K", "MR_K", "independent_minADE_K")}}
    return result
