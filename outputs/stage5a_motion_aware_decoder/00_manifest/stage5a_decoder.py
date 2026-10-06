"""Two learned residual experts between the original aggr_embed and loc/scale."""
import torch
from torch import nn
from torch.nn import functional as F
from models.hivt_runtime.decoder import MLPDecoder

FEATURE_NAMES = ('type_vehicle', 'type_pedestrian', 'type_bicycle',
                 'log1p_recent_displacement', 'log1p_net_displacement', 'log1p_path_length')


def condition_features(history, history_padding, agent_type):
    """Only observed history; bridge padding gaps in chronological order."""
    if history.ndim != 3 or history.shape[1:] != (5, 2):
        raise ValueError('history must be [N,5,2]')
    if history_padding.shape != history.shape[:2] or history_padding.dtype != torch.bool:
        raise ValueError('history_padding must be bool [N,5]')
    if agent_type.shape != history.shape[:1] or agent_type.dtype != torch.long:
        raise ValueError('agent_type must be int64 [N]')
    if agent_type.numel() and (int(agent_type.min()) < 0 or int(agent_type.max()) > 2):
        raise ValueError('Only V=0, P=1, B=2 are supported')
    valid = ~history_padding
    previous = history.new_zeros((len(history), 2))
    first = previous.clone()
    seen = torch.zeros(len(history), dtype=torch.bool, device=history.device)
    recent = history.new_zeros(len(history))
    path = recent.clone()
    for t in range(5):
        point = history[:, t]
        pair = valid[:, t] & seen
        distance = torch.linalg.vector_norm(point - previous, dim=-1)
        recent = torch.where(pair, distance, recent)
        path = path + torch.where(pair, distance, torch.zeros_like(distance))
        first = torch.where((valid[:, t] & ~seen)[:, None], point, first)
        previous = torch.where(valid[:, t, None], point, previous)
        seen = seen | valid[:, t]
    enough = valid.sum(-1) >= 2
    net = torch.where(enough, torch.linalg.vector_norm(previous-first, dim=-1), torch.zeros_like(path))
    motion = torch.stack((recent, net, path), dim=-1).log1p()
    one_hot = F.one_hot(agent_type, num_classes=3).to(history.dtype)
    return torch.cat((one_hot, motion), dim=-1)


class MotionAwareResidualDecoder(MLPDecoder):
    def __init__(self, local_channels=64, global_channels=64, future_steps=12,
                 num_modes=6, uncertain=True, min_scale=1e-3):
        if (local_channels, global_channels, num_modes) != (64, 64, 6):
            raise ValueError('Frozen HiVT-64 / K=6 architecture required')
        super().__init__(local_channels, global_channels, future_steps, num_modes, uncertain, min_scale)
        self.router = nn.Sequential(nn.Linear(6, 16), nn.ReLU(), nn.Linear(16, 2))
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(64, 16), nn.ReLU(), nn.Linear(16, 64)) for _ in range(2)])
        for expert in self.experts:
            nn.init.zeros_(expert[-1].weight)
            nn.init.zeros_(expert[-1].bias)
        nn.init.zeros_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)
        self.record_observation = False
        self.observation = None

    def route(self, condition):
        return self.router(condition).softmax(dim=-1)

    def residual(self, hidden, routing):
        expert_outputs = torch.stack([expert(hidden) for expert in self.experts], dim=-2)
        delta = (expert_outputs * routing[None, :, :, None]).sum(dim=-2)
        return delta, expert_outputs

    def forward(self, local_embed, global_embed, condition):
        # The original pi head sees the original embeddings, never the residual.
        pi = self.pi(torch.cat((local_embed.expand(self.num_modes, *local_embed.shape),
                                global_embed), dim=-1)).squeeze(-1).t()
        out = self.aggr_embed(torch.cat((global_embed, local_embed.expand(self.num_modes, *local_embed.shape)), dim=-1))
        routing = self.route(condition)
        delta, expert_outputs = self.residual(out, routing)
        adapted_out = out + delta
        loc = self.loc(adapted_out).view(self.num_modes, -1, self.future_steps, 2)
        if self.record_observation:
            self.observation = {'condition': condition.detach(), 'routing': routing.detach(),
                                'hidden': out.detach(), 'expert_outputs': expert_outputs.detach(),
                                'residual': delta.detach()}
        if self.uncertain:
            scale = F.elu_(self.scale(adapted_out), alpha=1.0).view(self.num_modes, -1, self.future_steps, 2) + 1.0
            scale = scale + self.min_scale
            return torch.cat((loc, scale), dim=-1), pi
        return loc, pi
