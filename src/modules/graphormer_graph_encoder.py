# from https://github.com/microsoft/Graphormer

from typing import Optional, Tuple, Union, List, Any

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import LayerNorm

from src.modules.multihead_attention import MultiheadAttention
from src.modules.graphormer_layers import GraphNodeFeature, GraphAttnBias
from src.modules.graphormer_graph_encoder_layer import GraphormerGraphEncoderLayer


class GraphormerGraphEncoder(nn.Module):
    def __init__(
            self,
            # < graph
            node_feature_dim: int,
            num_nodes: int,
            num_in_degree: int,
            num_out_degree: int,
            num_edges: int,
            num_spatial: int,
            num_edge_dis: int,
            edge_type: str,
            multi_hop_max_dist: int,
            static_graph: bool = True,
            graph_token=False,
            start_conv=True,
            centrality_encoding: bool = True,
            attention_bias=True,
            edge_features=False,
            old_config=False,
            # >
            # transformer
            num_encoder_layers: int = 12,
            embedding_dim: int = 768,
            ffn_embedding_dim: int = 768,
            num_attention_heads: int = 32,
            dropout: float = 0.1,
            attention_dropout: float = 0.1,
            activation_dropout: float = 0.1,
            encoder_normalize_before: bool = True,
            pre_layernorm: bool = True,
            apply_graphormer_init: bool = False,
            activation_fn: str = "gelu",
            embed_scale: float = None,
            freeze_embeddings: bool = False,
            n_trans_layers_to_freeze: int = 0,
            export: bool = False,
            traceable: bool = False,
    ) -> None:

        super().__init__()
        self.static_graph = static_graph
        self.graph_token = graph_token

        self.dropout_module = nn.Dropout(dropout)
        self.embedding_dim = embedding_dim
        self.embed_scale = embed_scale
        self.traceable = traceable
        self.edge_features = edge_features
        self.apply_graphormer_init = apply_graphormer_init

        self.start_conv = start_conv
        self.centrality_encoding = centrality_encoding
        self.graph_node_feature = GraphNodeFeature(
            node_feature_dim=node_feature_dim,
            num_heads=num_attention_heads,
            num_nodes=num_nodes,
            num_in_degree=num_in_degree,
            num_out_degree=num_out_degree,
            hidden_dim=embedding_dim,
            start_conv=self.start_conv,
            centrality_encoding=self.centrality_encoding,
            act_fn=activation_fn,
            old_config=old_config,
        )

        self.attention_bias = attention_bias
        if self.attention_bias:
            self.graph_attn_bias = GraphAttnBias(
                num_heads=num_attention_heads,
                num_edges=num_edges,
                num_spatial=num_spatial,
                num_edge_dis=num_edge_dis,
                edge_type=edge_type,
                multi_hop_max_dist=multi_hop_max_dist,
                graph_token=graph_token,
                edge_features=edge_features
            )

        if encoder_normalize_before:
            self.emb_layer_norm = LayerNorm(self.embedding_dim, eps=1e-8)
        else:
            self.emb_layer_norm = None

        self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                GraphormerGraphEncoderLayer(
                    embedding_dim=embedding_dim,
                    ffn_embedding_dim=ffn_embedding_dim,
                    num_attention_heads=num_attention_heads,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    activation_dropout=activation_dropout,
                    activation_fn=activation_fn,
                    export=export,
                    pre_layernorm=pre_layernorm,
                )
                for _ in range(num_encoder_layers)
            ]
        )

        def freeze_module_params(m):
            if m is not None:
                for p in m.parameters():
                    p.requires_grad = False

        if freeze_embeddings:
            raise NotImplementedError("Freezing embeddings is not implemented yet.")

        for layer in range(n_trans_layers_to_freeze):
            freeze_module_params(self.layers[layer])

    def compute_attn_bias(self, batched_data):
        attn_bias = self.graph_attn_bias(batched_data)
        return attn_bias

    def compute_mods(self, batched_data, token_embeddings=None, perturb=None):
        x, in_degree, out_degree = (
            batched_data["x"],
            batched_data["in_degree"],
            batched_data["out_degree"],
        )

        if token_embeddings is not None:
            x = self.graph_node_feature(token_embeddings, in_degree, out_degree)
        else:
            x = self.graph_node_feature(x, in_degree, out_degree)

        del batched_data["in_degree"], batched_data["out_degree"], batched_data['edge_index']
        del in_degree, out_degree
        if perturb is not None:
            # ic(torch.mean(torch.abs(x[:, 1, :])))
            # ic(torch.mean(torch.abs(perturb)))
            x[:, 1:, :] += perturb

        # x: B x T x C

        # TODO: consider attn bias edge encoding for different types brain connectivity
        #  for PET, using structural connectivity is appropriate since the a-beta and tau travel through,
        #  for MRI, using functional connectivity is more appropriate,
        #  for protein expression, perhaps another type of connectivity?
        # shape [n_graph, n_head, n_node + 1, n_node + 1]
        attn_bias = None
        if self.attention_bias:
            attn_bias = self.graph_attn_bias(batched_data)
        return x, attn_bias

    def forward_transformer_layers(self, x, padding_mask, attn_bias=None, attn_mask=None, last_state_only=True):
        # B x T x C -> T x B x C
        x = x.contiguous().transpose(0, 1)

        inner_states = []
        if not last_state_only:
            inner_states.append(x)

        for layer in self.layers:
            x, _ = layer(
                x,
                self_attn_padding_mask=padding_mask,
                self_attn_mask=attn_mask,
                self_attn_bias=attn_bias,
            )
            if not last_state_only:
                inner_states.append(x)

        if last_state_only:
            inner_states = [x]
        return inner_states

    def forward(
            self,
            x,
            attn_bias=None,
            # perturb=None,
            last_state_only: bool = True,
            # token_embeddings: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ) -> Union[Tensor, list[torch.Tensor]]:
        if attn_bias is None:
            assert not self.attention_bias, 'missing graph attention bias'

        # compute padding mask. This is needed for multi-head attention
        B, T, D = x.shape

        padding_mask = None
        # what's the point of adding graph token mask? we want to attend to special tokens
        # padding_mask = (x[:, :, 0]).eq(0)  # B x T x 1
        # bug in the original code corrected below:
        # padding_mask = torch.all(x[:,:,].eq(0), dim=-1)
        # if self.graph_token:
        #     padding_mask_graph_tok = torch.zeros(
        #         N, 1, device=padding_mask.device, dtype=padding_mask.dtype
        #     )
        #     padding_mask = torch.cat((padding_mask_graph_tok, padding_mask), dim=1)
        # else:
        #     padding_mask_cls = torch.zeros(
        #         N, 1, device=padding_mask.device, dtype=padding_mask.dtype
        #     )
        #     padding_mask = torch.cat((padding_mask_cls, padding_mask), dim=1)
        # B x (T+1) x 1

        if self.embed_scale is not None:
            x = x * self.embed_scale
        if self.emb_layer_norm is not None:
            x = self.emb_layer_norm(x)
        x = self.dropout_module(x)

        inner_states = self.forward_transformer_layers(
            x=x,
            padding_mask=padding_mask,
            attn_bias=attn_bias,
            attn_mask=attn_mask,
            last_state_only=last_state_only,
        )

        if self.traceable:
            return torch.stack(inner_states)
        else:
            return inner_states
