"""Unmodified Stage3B interaction backbone; isolated residual MLP decoder."""
import torch
from stage3b_model import HiVTTypeEmbedding
from stage5a_decoder import MotionAwareResidualDecoder, condition_features


class HiVTMotionAwareDecoder(HiVTTypeEmbedding):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.decoder = MotionAwareResidualDecoder(future_steps=self.future_steps, num_modes=self.num_modes)

    def forward(self, data):
        working = data.clone()
        self.validate_actor_types(working.agent_type, working.num_nodes)
        s, c = torch.sin(working.rotate_angles), torch.cos(working.rotate_angles)
        rotation = torch.stack([c, -s, s, c], dim=-1).reshape(-1, 2, 2)
        working.rotate_mat = rotation
        local = self.local_encoder(working)
        local = self.fuse_actor_types(local, working.agent_type)
        global_embed = self.global_interactor(working, local)
        condition = condition_features(working.positions[:, :5], working.padding_mask[:, :5], working.agent_type)
        prediction, logits = self.decoder(local, global_embed, condition)
        return {'raw_prediction': prediction, 'mode_logits': logits,
                'mode_prob': logits.softmax(dim=-1), 'rotation': rotation}
