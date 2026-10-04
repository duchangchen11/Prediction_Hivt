"""Explicit optimization-only losses; all HiVT modules/forward stay inherited."""
import torch

from .hivt_nuscenes import HiVTNuScenesVehicle


class HiVTLossRecovery(HiVTNuScenesVehicle):
    def recovery_loss(self, output, data, phase="original_nll", b_max=None):
        raw = output["raw_prediction"]
        target = torch.bmm(data.y, output["rotation"])
        mask = data.future_mask & data.target_mask[:, None]
        distance = torch.linalg.vector_norm(raw[..., :2]-target[None], dim=-1)
        best = (distance*mask[None]).sum(dim=-1).argmin(dim=0)
        chosen = raw[best, torch.arange(data.num_nodes, device=target.device)]
        location_error = (chosen[..., :2][mask]-target[mask]).abs().mean()
        if phase == "fixed_scale":
            # Keep the exact upstream detached-soft-target mode classification.
            official = super().loss(output, data)
            classification = official["classification_loss"]
            values = {"loss": location_error+classification, "regression_loss": location_error,
                      "classification_loss": classification}
            effective = torch.ones_like(chosen[..., 2:])
        elif phase in ("original_nll", "bounded_scale"):
            adjusted = output
            if phase == "bounded_scale":
                if b_max not in (2., 4., 8.):
                    raise ValueError("Only preregistered b_max=2/4/8 are permitted")
                adjusted = dict(output, raw_prediction=torch.cat([raw[..., :2], raw[..., 2:].clamp(min=1e-3, max=b_max)], dim=-1))
            values = super().loss(adjusted, data)
            effective = chosen[..., 2:] if phase == "original_nll" else chosen[..., 2:].clamp(min=1e-3, max=b_max)
        else:
            raise ValueError(phase)
        # Report NLL separately: fixed-scale regression intentionally omits log(2).
        nll = (torch.log(2*effective[mask])+(chosen[..., :2][mask]-target[mask]).abs()/effective[mask]).mean()
        return {**values, "location_error": location_error, "NLL": nll}
