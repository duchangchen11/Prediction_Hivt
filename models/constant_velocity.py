"""One deterministic CV prediction per agent, using past observations only."""
import torch


def constant_velocity(agent_pos, history_mask, history_times, future_times):
    if agent_pos.ndim != 3 or agent_pos.shape[-1] != 2:
        raise ValueError("agent_pos must be [N, Th, 2]")
    n, th, _ = agent_pos.shape
    if history_mask.shape != (n, th) or history_times.shape != (th,) or future_times.ndim != 1:
        raise ValueError("Mask / timestamps shape mismatch")
    if history_mask.dtype != torch.bool:
        raise ValueError("history_mask must be boolean")
    if not torch.isfinite(agent_pos[history_mask]).all() or not torch.isfinite(history_times).all() or not torch.isfinite(future_times).all():
        raise ValueError("Non-finite valid history or times")
    if not (torch.diff(history_times) > 0).all() or not (torch.diff(future_times) > 0).all():
        raise ValueError("Timestamps must increase strictly")
    if len(future_times) == 0 or future_times[0] <= history_times[-1]:
        raise ValueError("Future times must be later than history")
    prediction = agent_pos.new_zeros((n, len(future_times), 2))
    valid_prediction = torch.zeros(n, dtype=torch.bool, device=agent_pos.device)
    velocity = agent_pos.new_zeros((n, 2))
    for i in range(n):
        valid = torch.where(history_mask[i])[0]
        if len(valid) < 2:
            continue  # excluded from metrics; never fabricate a zero-speed actor
        previous, latest = valid[-2], valid[-1]
        dt = history_times[latest] - history_times[previous]
        velocity[i] = (agent_pos[i, latest] - agent_pos[i, previous]) / dt.to(agent_pos.dtype)
        elapsed = (future_times - history_times[latest]).to(device=agent_pos.device, dtype=agent_pos.dtype)
        prediction[i] = agent_pos[i, latest] + elapsed[:, None] * velocity[i]
        valid_prediction[i] = True
    return prediction, valid_prediction, velocity
