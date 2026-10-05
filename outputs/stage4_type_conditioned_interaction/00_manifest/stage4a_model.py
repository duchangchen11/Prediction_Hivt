"""Stage3B plus the single preregistered type-conditioned relation-bias adapter."""
from stage3b_model import HiVTTypeEmbedding
from stage4a_global_interactor import TypeConditionedGlobalInteractor


class HiVTTypeConditionedInteraction(HiVTTypeEmbedding):
    def __init__(self, pair_embedding_dim=16, relation_bias_hidden_dim=32, **kwargs):
        super().__init__(**kwargs)
        self.global_interactor = TypeConditionedGlobalInteractor(
            historical_steps=kwargs['historical_steps'],
            embed_dim=kwargs['embed_dim'], edge_dim=2,
            num_modes=kwargs['num_modes'], num_heads=kwargs['num_heads'],
            num_layers=kwargs['num_global_layers'], dropout=kwargs['dropout'],
            rotate=True, local_radius=kwargs['local_radius'],
            pair_embedding_dim=pair_embedding_dim,
            relation_bias_hidden_dim=relation_bias_hidden_dim)
        # This is deliberately last, after every construction/init operation.
        self.global_interactor.reset_neutral_bias()

    # The unchanged inherited forward retains LocalEncoder, TypeEmbedding,
    # decoder inputs/outputs, actor validation and all loss/optimizer behavior.
