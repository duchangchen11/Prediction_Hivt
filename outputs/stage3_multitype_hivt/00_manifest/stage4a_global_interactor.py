"""Directed type-pair logit bias; frozen HiVT topology and values are preserved.

The shared GlobalInteractor and GlobalInteractorLayer logic derives from the
Apache-2.0 HiVT implementation (Copyright 2022 Zikang Zhou). Only the additive
attention-logit term and its small shared relation network are new.
"""
from typing import Optional

import torch
from torch import nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor, Size
from torch_geometric.utils import softmax, subgraph

from models.hivt_runtime.global_interactor import GlobalInteractor
from models.hivt_runtime.utils import init_weights


PAIR_LABELS = ('V<-V', 'V<-P', 'V<-B', 'P<-V', 'P<-P', 'P<-B',
               'B<-V', 'B<-P', 'B<-B')


class TypeConditionedGlobalInteractorLayer(MessagePassing):
    """Original layer with exactly one pre-softmax addition.

    A complete MessagePassing subclass keeps PyG's generated propagate
    signature aware of relation_bias without mutable per-forward attributes.
    """

    def __init__(self, embed_dim: int, num_heads: int = 8,
                 dropout: float = 0.1, **kwargs) -> None:
        super().__init__(aggr='add', node_dim=0, **kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.lin_q_node = nn.Linear(embed_dim, embed_dim)
        self.lin_k_node = nn.Linear(embed_dim, embed_dim)
        self.lin_k_edge = nn.Linear(embed_dim, embed_dim)
        self.lin_v_node = nn.Linear(embed_dim, embed_dim)
        self.lin_v_edge = nn.Linear(embed_dim, embed_dim)
        self.lin_self = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.lin_ih = nn.Linear(embed_dim, embed_dim)
        self.lin_hh = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout))

    def forward(self, x: torch.Tensor, edge_index: Adj,
                edge_attr: torch.Tensor, relation_bias: torch.Tensor,
                size: Size = None) -> torch.Tensor:
        x = x + self._mha_block(self.norm1(x), edge_index, edge_attr,
                                relation_bias, size)
        x = x + self._ff_block(self.norm2(x))
        return x

    def message(self, x_i: torch.Tensor, x_j: torch.Tensor,
                edge_attr: torch.Tensor, relation_bias: torch.Tensor,
                index: torch.Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> torch.Tensor:
        query = self.lin_q_node(x_i).view(-1, self.num_heads,
                                         self.embed_dim // self.num_heads)
        key_node = self.lin_k_node(x_j).view(-1, self.num_heads,
                                            self.embed_dim // self.num_heads)
        key_edge = self.lin_k_edge(edge_attr).view(-1, self.num_heads,
                                                 self.embed_dim // self.num_heads)
        value_node = self.lin_v_node(x_j).view(-1, self.num_heads,
                                              self.embed_dim // self.num_heads)
        value_edge = self.lin_v_edge(edge_attr).view(-1, self.num_heads,
                                                   self.embed_dim // self.num_heads)
        scale = (self.embed_dim // self.num_heads) ** 0.5
        alpha = (query * (key_node + key_edge)).sum(dim=-1) / scale
        alpha = alpha + relation_bias
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = self.attn_drop(alpha)
        return (value_node + value_edge) * alpha.unsqueeze(-1)

    def update(self, inputs: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        inputs = inputs.view(-1, self.embed_dim)
        gate = torch.sigmoid(self.lin_ih(inputs) + self.lin_hh(x))
        return inputs + gate * (self.lin_self(x) - inputs)

    def _mha_block(self, x: torch.Tensor, edge_index: Adj,
                   edge_attr: torch.Tensor, relation_bias: torch.Tensor,
                   size: Size) -> torch.Tensor:
        x = self.out_proj(self.propagate(edge_index=edge_index, x=x,
                                         edge_attr=edge_attr,
                                         relation_bias=relation_bias, size=size))
        return self.proj_drop(x)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class TypeConditionedGlobalInteractor(GlobalInteractor):
    def __init__(self, historical_steps: int, embed_dim: int, edge_dim: int,
                 num_modes: int = 6, num_heads: int = 8, num_layers: int = 3,
                 dropout: float = 0.1, rotate: bool = True,
                 local_radius: float = 50.0, pair_embedding_dim: int = 16,
                 relation_bias_hidden_dim: int = 32) -> None:
        if pair_embedding_dim != 16 or relation_bias_hidden_dim != 32:
            raise ValueError('Stage4A fixes pair_embedding_dim=16 and hidden_dim=32')
        if local_radius != 50.0 or num_layers != 3 or num_heads != 8:
            raise ValueError('Stage4A freezes radius=50m, layers=3, heads=8')
        super().__init__(historical_steps, embed_dim, edge_dim, num_modes,
                         num_heads, num_layers, dropout, rotate)
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.local_radius = local_radius
        self.global_interactor_layers = nn.ModuleList(
            [TypeConditionedGlobalInteractorLayer(embed_dim, num_heads, dropout)
             for _ in range(num_layers)])
        self.global_interactor_layers.apply(init_weights)
        self.pair_embedding = nn.Embedding(9, pair_embedding_dim)
        self.relation_mlp = nn.Sequential(
            nn.Linear(pair_embedding_dim + 4, relation_bias_hidden_dim),
            nn.ReLU(), nn.Linear(relation_bias_hidden_dim, num_layers * num_heads))
        self.relation_mlp.apply(init_weights)
        self.reset_neutral_bias()
        # Diagnostic capture is opt-in and does not add synchronization to training.
        self.record_relation_bias = False
        self.relation_bias_observation = None

    def reset_neutral_bias(self) -> None:
        nn.init.zeros_(self.relation_mlp[-1].weight)
        nn.init.zeros_(self.relation_mlp[-1].bias)

    @staticmethod
    def directed_pair_ids(agent_type: torch.Tensor,
                          edge_index: torch.Tensor) -> torch.Tensor:
        # PyG source=j on row 0, target=i on row 1. No topology mutation.
        return 3 * agent_type[edge_index[1]] + agent_type[edge_index[0]]

    def _edge_relations(self, data):
        edge_index, _ = subgraph(
            subset=~data['padding_mask'][:, self.historical_steps - 1],
            edge_index=data.edge_index)
        rel_pos = (data['positions'][edge_index[0], self.historical_steps - 1]
                   - data['positions'][edge_index[1], self.historical_steps - 1])
        if data['rotate_mat'] is not None:
            rel_pos = torch.bmm(rel_pos.unsqueeze(-2),
                                data['rotate_mat'][edge_index[1]]).squeeze(-2)
        rel_theta = (data['rotate_angles'][edge_index[0]]
                     - data['rotate_angles'][edge_index[1]])
        heading = torch.stack((torch.cos(rel_theta), torch.sin(rel_theta)), dim=-1)
        return edge_index, rel_pos, heading

    def _relation_features(self, data, edge_index, rel_pos, heading):
        pair_ids = self.directed_pair_ids(data.agent_type, edge_index)
        features = torch.cat((self.pair_embedding(pair_ids),
                              rel_pos / self.local_radius, heading), dim=-1)
        bias = self.relation_mlp(features).reshape(
            -1, self.num_layers, self.num_heads)
        return pair_ids, features, bias

    def relation_bias(self, data):
        """Recompute diagnostic inputs using exactly the training edge convention."""
        edges, rel_pos, heading = self._edge_relations(data)
        pairs, features, bias = self._relation_features(data, edges, rel_pos, heading)
        return {'edge_index': edges, 'pair_ids': pairs,
                'relation_features': features, 'bias': bias}

    def forward(self, data, local_embed: torch.Tensor) -> torch.Tensor:
        edge_index, rel_pos, heading = self._edge_relations(data)
        if data['rotate_mat'] is None:
            rel_embed = self.rel_embed(rel_pos)
        else:
            rel_embed = self.rel_embed([rel_pos, heading])
        pair_ids, features, bias = self._relation_features(
            data, edge_index, rel_pos, heading)
        if self.record_relation_bias:
            self.relation_bias_observation = {
                'edge_index': edge_index.detach(), 'pair_ids': pair_ids.detach(),
                'relation_features': features.detach(), 'bias': bias.detach()}
        x = local_embed
        for layer_index, layer in enumerate(self.global_interactor_layers):
            x = layer(x, edge_index, rel_embed, bias[:, layer_index, :])
        x = self.norm(x)
        x = self.multihead_proj(x).view(-1, self.num_modes, self.embed_dim)
        return x.transpose(0, 1)
