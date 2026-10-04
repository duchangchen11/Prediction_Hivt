"""Native training wrapper around the official HiVT-64 components/losses."""
import torch
from torch import nn
from torch.nn import functional as F

from .hivt_runtime.local_encoder import LocalEncoder
from .hivt_runtime.global_interactor import GlobalInteractor
from .hivt_runtime.decoder import MLPDecoder
from .hivt_runtime.laplace_nll_loss import LaplaceNLLLoss
from .hivt_runtime.soft_target_cross_entropy_loss import SoftTargetCrossEntropyLoss


class HiVTNuScenesVehicle(nn.Module):
    def __init__(self, historical_steps=5, future_steps=12, num_modes=6, embed_dim=64,
                 num_heads=8, dropout=.1, num_temporal_layers=4, num_global_layers=3, local_radius=50.):
        super().__init__()
        self.historical_steps, self.future_steps, self.num_modes = historical_steps, future_steps, num_modes
        self.local_encoder = LocalEncoder(historical_steps, 2, 2, embed_dim, num_heads, dropout, num_temporal_layers, local_radius, parallel=False)
        self.global_interactor = GlobalInteractor(historical_steps, embed_dim, 2, num_modes, num_heads, num_global_layers, dropout, rotate=True)
        self.decoder = MLPDecoder(embed_dim, embed_dim, future_steps, num_modes, uncertain=True)
        self.reg_loss = LaplaceNLLLoss(reduction="mean")
        self.cls_loss = SoftTargetCrossEntropyLoss(reduction="mean")

    def forward(self, data):
        # Upstream mutates y; use a fresh graph to prevent double target rotation.
        working = data.clone()
        s, c = torch.sin(working.rotate_angles), torch.cos(working.rotate_angles)
        rotation = torch.stack([c, -s, s, c], dim=-1).reshape(-1, 2, 2)
        working.rotate_mat = rotation
        local = self.local_encoder(working)
        global_embed = self.global_interactor(working, local)
        prediction, logits = self.decoder(local, global_embed)
        return {"raw_prediction": prediction, "mode_logits": logits,
                "mode_prob": logits.softmax(dim=-1), "rotation": rotation}

    def loss(self, output, data):
        target = torch.bmm(data.y, output["rotation"])
        mask = data.future_mask & data.target_mask[:, None]
        steps = mask.sum(dim=1)
        eligible = steps > 0
        if not eligible.any():
            raise ValueError("Batch has no eligible vehicle target")
        prediction = output["raw_prediction"]
        l2 = (torch.linalg.vector_norm(prediction[..., :2]-target[None], dim=-1)*mask[None]).sum(dim=-1)
        best = l2.argmin(dim=0)
        chosen = prediction[best, torch.arange(data.num_nodes, device=target.device)]
        regression = self.reg_loss(chosen[mask], target[mask])
        soft_target = F.softmax(-l2[:, eligible]/steps[eligible], dim=0).T.detach()
        classification = self.cls_loss(output["mode_logits"][eligible], soft_target)
        return {"loss": regression+classification, "regression_loss": regression, "classification_loss": classification}

    def ego_predictions(self, output, data):
        # [K,N,T,2] agent-frame offsets -> [N,K,T,2] t0-ego positions.
        local = output["raw_prediction"][..., :2].permute(1, 0, 2, 3)
        return torch.matmul(local, output["rotation"].transpose(-1, -2)[:, None])+data.positions[:, self.historical_steps-1, None, None]

    def optimizer(self, lr=5e-4, weight_decay=1e-4):
        # Same named-module decay partition as upstream configure_optimizers.
        decay, no_decay = set(), set()
        whitelist = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.MultiheadAttention, nn.LSTM, nn.GRU)
        blacklist = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.Embedding)
        for module_name, module in self.named_modules():
            for name, _ in module.named_parameters():
                key = f"{module_name}.{name}" if module_name else name
                if "bias" in name:
                    no_decay.add(key)
                elif "weight" in name:
                    if isinstance(module, whitelist): decay.add(key)
                    elif isinstance(module, blacklist): no_decay.add(key)
                else:
                    no_decay.add(key)
        parameters = dict(self.named_parameters())
        assert not decay & no_decay and set(parameters) == decay | no_decay
        return torch.optim.AdamW([{"params": [parameters[k] for k in sorted(decay)], "weight_decay": weight_decay},
                                 {"params": [parameters[k] for k in sorted(no_decay)], "weight_decay": 0.}], lr=lr)
