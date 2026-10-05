"""One-variable HiVT ablation: add three actor-type vectors after LocalEncoder."""
import torch
from torch import nn
from models.hivt_loss_recovery import HiVTLossRecovery

class HiVTTypeEmbedding(HiVTLossRecovery):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.type_embedding = nn.Embedding(num_embeddings=3, embedding_dim=64)

    @staticmethod
    def validate_actor_types(agent_type, actor_count):
        if agent_type.dtype != torch.long or agent_type.ndim != 1 or agent_type.numel() != actor_count:
            raise ValueError('agent_type must be int64 [number of actors]')
        if actor_count and (int(agent_type.min()) < 0 or int(agent_type.max()) > 2):
            raise ValueError('Only vehicle=0, pedestrian=1, bicycle=2 are permitted')

    def fuse_actor_types(self, local, agent_type):
        self.validate_actor_types(agent_type, local.shape[0])
        if local.ndim != 2 or local.shape[1] != 64:
            raise ValueError('Local actor embedding must have shape [N_actor,64]')
        return local + self.type_embedding(agent_type)

    def forward(self, data):
        # Match the inherited forward exactly, with one additive fusion line.
        working = data.clone()
        self.validate_actor_types(working.agent_type, working.num_nodes)
        s, c = torch.sin(working.rotate_angles), torch.cos(working.rotate_angles)
        rotation = torch.stack([c, -s, s, c], dim=-1).reshape(-1, 2, 2)
        working.rotate_mat = rotation
        local = self.local_encoder(working)
        local = self.fuse_actor_types(local, working.agent_type)
        global_embed = self.global_interactor(working, local)
        prediction, logits = self.decoder(local, global_embed)
        return {'raw_prediction': prediction, 'mode_logits': logits,
                'mode_prob': logits.softmax(dim=-1), 'rotation': rotation}
