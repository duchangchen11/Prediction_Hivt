"""ADE averages valid future points per agent, then agents equally.

FDE uses each eligible actor's last valid future annotation; also report the
fixed-horizon FDE using only actors observed at the final requested frame.
"""
import torch


def displacement_errors(prediction, target, future_mask, prediction_mask=None):
    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[-1] != 2:
        raise ValueError("Prediction and target must be [N, Tf, 2]")
    if future_mask.shape != target.shape[:2] or future_mask.dtype != torch.bool:
        raise ValueError("future_mask must be boolean [N, Tf]")
    mask = future_mask.clone()
    if prediction_mask is not None:
        if prediction_mask.shape != (len(target),) or prediction_mask.dtype != torch.bool:
            raise ValueError("prediction_mask must be boolean [N]")
        mask &= prediction_mask[:, None]
    # Index before subtracting: padded values (even NaN in another source) do
    # not participate. In our saved windows padding is finite zero + false mask.
    if not torch.isfinite(prediction[mask]).all() or not torch.isfinite(target[mask]).all():
        raise ValueError("Non-finite valid future positions")
    errors = target.new_zeros(mask.shape)
    errors[mask] = torch.linalg.vector_norm(prediction[mask] - target[mask], dim=-1)
    counts = mask.sum(dim=1)
    valid = counts > 0
    ade = errors.sum(dim=1) / counts.clamp(min=1)
    indices = torch.arange(mask.shape[1], device=mask.device).expand_as(mask)
    last = torch.where(mask, indices, -1).max(dim=1).values
    rows = torch.arange(len(target), device=target.device)
    fde = errors[rows, last.clamp(min=0)]
    return {"ade": ade, "fde": fde, "valid": valid, "valid_points": counts,
            "fde_index": last, "fixed_horizon_valid": mask[:, -1],
            "fixed_horizon_fde": errors[:, -1]}
