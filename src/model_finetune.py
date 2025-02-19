import math
from functools import partial
from typing import List

from src.modules import graphormer_graph_encoder
from src.utils.log import master_print as print

import torch
import torch.nn as nn


class CausalConv2d(nn.Conv2d):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            dilation=1,
            batch_norm=True,
    ):
        padding_t = dilation * (kernel_size[1] - 1)
        padding_v = dilation * (kernel_size[0] - 1) // 2
        super(CausalConv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=(padding_v, padding_t),  # center padding for V dimension
            dilation=dilation,
            bias=False if batch_norm else True,
        )
        self.causal_padding_t = padding_t
        if batch_norm:
            self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = super(CausalConv2d, self).forward(x)
        x = x[:, :, :, :-self.causal_padding_t]  # remove trailing padding in T dimension
        if hasattr(self, 'batch_norm'):
            x = self.batch_norm(x)
        return x


class GraphEncoder(nn.Module):
    def __init__(
            self,
            # < graph args
            node_feature_dim: int = 1,
            pred_node_dim: int = 1,
            num_nodes: int = 512 * 9,
            num_edges: int = 512 * 3,
            num_in_degree: int = 512,
            num_out_degree: int = 512,
            num_spatial: int = 512,
            num_edge_dis: int = 128,
            edge_type: str = 'multi_hop',
            multi_hop_max_dist: int = 5,
            static_graph=True,
            # >
            # < transformer args
            old_config=False,
            graph_token=True,
            cls_token=False,
            sep_pos_embed=True,
            ablate_pos_embed=False,
            attention_bias=True,
            centrality_encoding=True,
            num_heads=16,
            encoder_embed_dim=1024,
            encoder_depth=24,
            pre_layernorm=True,
            norm_layer=nn.LayerNorm,
            dropout=0.1,
            n_hist=16,
            trunc_init=False,
            act_fn='gelu',
            # >
            **kwargs
    ):
        super().__init__()
        if not static_graph:
            raise NotImplementedError()
        self.node_feature_dim = node_feature_dim
        self.pred_node_dim = pred_node_dim

        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.num_in_degree = num_in_degree
        self.num_out_degree = num_out_degree
        self.num_spatial = num_spatial
        self.num_edge_dis = num_edge_dis
        self.edge_type = edge_type
        self.multi_hop_max_dist = multi_hop_max_dist
        self.static_graph = static_graph

        self.sep_pos_embed = sep_pos_embed
        self.ablate_pos_embed = ablate_pos_embed
        self.attention_bias = attention_bias
        self.centrality_encoding = centrality_encoding

        self.graph_token = graph_token
        self.cls_token = cls_token

        self.encoder_embed_dim = encoder_embed_dim
        self.trunc_init = trunc_init
        self.act_fn = act_fn
        self.hist_t_dim = n_hist
        self.num_heads = num_heads
        self.dropout = dropout

        # MAE pretrained encoder
        if self.graph_token:
            self.graph_token_embed = nn.Parameter(torch.zeros(1, 1, 1, encoder_embed_dim))
            if self.attention_bias:
                self.graph_token_virtual_distance = nn.Embedding(1, num_heads)
        if self.cls_token:
            assert self.graph_token is False
            self.cls_token_embed = nn.Parameter(torch.zeros(1, 1, encoder_embed_dim))
            if self.attention_bias:
                self.cls_token_virtual_distance = nn.Embedding(1, num_heads)

        if not self.ablate_pos_embed:
            if sep_pos_embed:
                self.pos_embed_time = nn.Parameter(
                    torch.zeros(1, n_hist, encoder_embed_dim),
                )
                self.pos_embed_space = nn.Parameter(
                    torch.zeros(1, num_nodes, encoder_embed_dim),
                )
                if self.graph_token or self.cls_token:
                    self.pos_embed_cls = nn.Parameter(torch.zeros(1, 1, encoder_embed_dim))
            else:
                if self.graph_token:
                    self.pos_embed = nn.Parameter(
                        torch.zeros(1, n_hist * (num_nodes + 1), encoder_embed_dim),
                    )
                elif self.cls_token:
                    self.pos_embed = nn.Parameter(
                        torch.zeros(1, n_hist * num_nodes + 1, encoder_embed_dim),
                    )
                else:
                    self.pos_embed = nn.Parameter(
                        torch.zeros(1, n_hist * num_nodes, encoder_embed_dim),
                    )

        self.pre_layernorm = pre_layernorm

        self.blocks = graphormer_graph_encoder.GraphormerGraphEncoder(
            # for graphormer
            node_feature_dim=node_feature_dim,
            num_nodes=num_nodes,
            num_in_degree=num_in_degree,
            num_out_degree=num_out_degree,
            num_edges=num_edges,
            num_spatial=num_spatial,
            num_edge_dis=num_edge_dis,
            edge_type=edge_type,
            multi_hop_max_dist=multi_hop_max_dist,
            # for transformer blocks
            num_encoder_layers=encoder_depth,
            embedding_dim=encoder_embed_dim,
            ffn_embedding_dim=encoder_embed_dim * 4,
            num_attention_heads=num_heads,
            static_graph=self.static_graph,
            graph_token=self.graph_token,
            start_conv=True,
            old_config=old_config,
            centrality_encoding=self.centrality_encoding,
            attention_bias=self.attention_bias,
            pre_layernorm=pre_layernorm,
            activation_fn=act_fn,
            dropout=dropout,
        )
        self.norm = norm_layer(encoder_embed_dim)

    @torch.jit.ignore
    def no_weight_decay(self):
        if self.ablate_pos_embed:
            if self.attention_bias:
                return {
                    "cls_token_embed",
                    "cls_token_virtual_distance.weight",
                    "graph_token_embed",
                    "graph_token_virtual_distance.weight",
                }
            return {
                "cls_token_embed",
                "graph_token_embed",
            }
        if self.sep_pos_embed:
            if self.attention_bias:
                return {
                    "cls_token_embed",
                    "cls_token_virtual_distance.weight",
                    "graph_token_embed",
                    "graph_token_virtual_distance.weight",
                    "pos_embed_time",
                    "pos_embed_space",
                    "pos_embed_cls" if self.graph_token or self.cls_token else ''
                                                                               ""
                }
            return {
                "cls_token_embed",
                "graph_token_embed",
                "pos_embed_time",
                "pos_embed_space",
                "pos_embed_cls" if self.graph_token or self.cls_token else ''
                                                                           ""
            }
        if self.attention_bias:
            return {
                "cls_token_embed",
                "cls_token_virtual_distance.weight",
                "graph_token_embed",
                "graph_token_virtual_distance.weight",
                "pos_embed",
            }
        return {
            "cls_token_embed",
            "graph_token_embed",
            "pos_embed",
        }

    def initialize_weights(self):
        if not self.ablate_pos_embed:
            if self.sep_pos_embed:
                torch.nn.init.trunc_normal_(self.pos_embed_time, std=0.02)
                torch.nn.init.trunc_normal_(self.pos_embed_space, std=0.02)
                if self.graph_token or self.cls_token:
                    torch.nn.init.trunc_normal_(self.pos_embed_cls, std=0.02)
            else:
                torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)

        if self.centrality_encoding:
            torch.nn.init.trunc_normal_(
                self.blocks.graph_node_feature.in_degree_encoder.weight,
                mean=0.0,
                std=0.02
            )
            torch.nn.init.trunc_normal_(
                self.blocks.graph_node_feature.out_degree_encoder.weight,
                mean=0.0,
                std=0.02
            )
        if self.attention_bias:
            torch.nn.init.trunc_normal_(
                self.blocks.graph_attn_bias.spatial_pos_encoder.weight,
                mean=0.0,
                std=0.02
            )
        if self.blocks.edge_features:
            torch.nn.init.trunc_normal_(
                self.blocks.graph_attn_bias.edge_encoder.weight,
                mean=0.0,
                std=0.02
            )
            if self.blocks.graph_attn_bias.edge_type == "multi_hop":
                torch.nn.init.trunc_normal_(
                    self.blocks.graph_attn_bias.edge_dis_encoder.weight,
                    mean=0.0,
                    std=0.02
                )
        if self.graph_token:
            torch.nn.init.trunc_normal_(self.graph_token_embed, std=0.02)
            if self.attention_bias:
                torch.nn.init.trunc_normal_(self.graph_token_virtual_distance.weight, std=0.02)
        elif self.cls_token:
            torch.nn.init.trunc_normal_(self.cls_token_embed, std=0.02)
            if self.attention_bias:
                torch.nn.init.trunc_normal_(self.cls_token_virtual_distance.weight, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            if self.trunc_init:
                nn.init.trunc_normal_(m.weight, std=0.02)
            else:
                torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            if self.trunc_init:
                nn.init.trunc_normal_(m.weight, std=0.02)
            else:
                torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def add_token_distance(self, cls: bool, attn_bias: torch.Tensor, device: torch.device):
        if not self.attention_bias:
            return None
        # correct attn_bias for added tokens
        N, n_heads, L, _ = attn_bias.shape
        if cls:
            # fix attn bias shape based on added cls token
            attn_bias = torch.cat(
                [torch.zeros(N, n_heads, 1, L, device=device), attn_bias],
                dim=2
            )
            attn_bias = torch.cat(
                [torch.zeros(N, n_heads, L + 1, 1, device=device), attn_bias],
                dim=3
            )
            # add virtual distance attn bias here for VNode
            t = self.cls_token_virtual_distance.weight.repeat(N, 1).contiguous().view(-1, n_heads, 1)
            attn_bias[:, :, 1:, 0] = attn_bias.clone()[:, :, 1:, 0] + t
            attn_bias[:, :, 0, :] = attn_bias.clone()[:, :, 0, :] + t
        else:
            t = self.graph_token_virtual_distance.weight.repeat(N, 1).contiguous().view(-1, n_heads, 1)
            attn_bias[:, :, 1:, 0] = attn_bias[:, :, 1:, 0] + t
            attn_bias[:, :, 0, :] = attn_bias[:, :, 0, :] + t
        return attn_bias

    def get_graph_rep(self, x, x_shape):
        # isolate graph tokens from x
        N, T, V, D = x_shape

        if self.graph_token:
            x = x.contiguous().view(N, T, -1, D)
            graph_rep = x[:, :, :1, :]
            x = x[:, :, 1:, :]
        elif self.cls_token:
            graph_rep = x[:, :1, :]
            x = x[:, 1:, :]
        else:
            graph_rep = None
            x = x[:, :, :]
        return graph_rep, x

    def forward_encoder(self, x: torch.Tensor, attn_bias: torch.Tensor):
        """

        :param x: time series data [Batch, Vertices, Embed_Dim, Time_step]
        :param attn_bias: attention bias computed based on graph structure and edge
        lengths [Batch, n_heads, Vertices, Vertices]
        :return: Tuple of
        [X of shape (Batch, Time_step * Vertices + added tokens, Embed_dim),
        graph_representation,
        mask from random_masking used to get binary mask,
        ids_restore to un-shuffle,
        x_shape from original input used for shaping]
        """
        if not self.attention_bias:
            del attn_bias
            attn_bias = None
        x_shape = list(x.shape)
        N, T, V, D = x_shape

        if self.graph_token:
            # 1 graph token for each time step,
            # masks based on each time_step, ids_keep shape [n_batches, T, V*mask_ratio]

            # repeating since graph tokens stay the same through time dimension
            graph_token_feature = self.graph_token_embed.expand(N, T, -1, -1)
            x = torch.cat([graph_token_feature, x], dim=2)
            x = x.contiguous().view(N, -1, D)

            attn_bias = self.add_token_distance(
                cls=False,
                attn_bias=attn_bias,
                device=x.device
            )
            attn_bias = attn_bias.repeat(1, 1, T, T) if self.attention_bias else None

            n_tokens = x.shape[1]
        else:
            # 1 graph token for all time steps (cls token) or no graph token
            x = x.contiguous().view(N, -1, D)

            # concatenate the cls_token
            if self.cls_token:
                cls_embed = self.cls_token_embed
                cls_embeds = cls_embed.contiguous().expand(N, -1, -1)
                x = torch.cat((cls_embeds, x), dim=1)

            attn_bias = attn_bias.repeat(1, 1, T, T) if self.attention_bias else None  # expand the time dimensions

            n_tokens = x.shape[1]
        if self.cls_token:
            attn_bias = self.add_token_distance(
                cls=True,
                attn_bias=attn_bias,
                device=x.device
            )

        if not self.ablate_pos_embed:
            if self.sep_pos_embed:
                pos_embed = self.pos_embed_space.repeat(
                    1, T, 1
                ) + torch.repeat_interleave(
                    self.pos_embed_time,
                    V,
                    dim=1
                )
                pos_embed = pos_embed.expand(N, -1, -1)
                if self.graph_token:
                    pos_embed_token = self.pos_embed_cls.expand(N, T, -1, -1)
                    pos_embed = torch.cat(
                        [
                            pos_embed_token,
                            pos_embed.contiguous().view(N, T, V, D),
                        ],
                        dim=2
                    )
                elif self.cls_token:
                    pos_embed_token = self.pos_embed_cls.expand(N, -1, -1)
                    pos_embed = torch.cat(
                        [pos_embed_token, pos_embed],
                        dim=1
                    )
            else:
                pos_embed = self.pos_embed[:, :, :].expand(N, -1, -1)

            x = x + pos_embed.contiguous().view(N, -1, D)

        # apply Transformer blocks
        inner_states = self.blocks(x, attn_bias)
        x = inner_states[-1].contiguous().transpose(0, 1)

        graph_rep, x = self.get_graph_rep(x, x_shape)
        x = x.contiguous().view(N, -1, D)
        return x, graph_rep, x_shape


# uses MAE_ST implementation, see https://github.com/facebookresearch/mae_st/blob/main/models_vit.py
class GraphEncoderMLP(GraphEncoder):
    """Graphformer Encoder for Classification Tasks"""

    def __init__(
            self,
            pred_per_time_step=True,
            pred_num_classes: int = 3,
            mlp_pred_dropout: float = 0.1,
            class_init_prob: List[float] = None,
            max_pooling=True,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if class_init_prob is None:
            class_init_prob = [1 / pred_num_classes] * pred_num_classes

        assert len(class_init_prob) == pred_num_classes
        assert math.isclose(sum(class_init_prob), 1.0), 'class probabilities must sum to 1'
        self.max_pooling = max_pooling
        self.pred_per_time_step = pred_per_time_step
        self.mlp_pred_dropout = nn.Dropout(mlp_pred_dropout)
        self.head = nn.Linear(self.encoder_embed_dim, pred_num_classes)

        self.initialize_weights()
        torch.nn.init.normal_(self.head.weight, std=2e-5)
        bias_values = torch.log(torch.FloatTensor(class_init_prob))
        self.head.bias.data = bias_values
        print("model initialized")

    def forward(self, batched_data):
        # loss is computed during training loop

        # compute attention biases and node centrality encodings
        x, attn_bias = self.blocks.compute_mods(batched_data)
        x, _, x_shape = self.forward_encoder(x, attn_bias)
        # x = self.norm(x)
        x = x.contiguous().view(x_shape[0], self.hist_t_dim, self.num_nodes, -1)
        if not self.pred_per_time_step:  # if predicting for the entire time series
            if self.max_pooling:
                x = x.max(dim=1).values  # global pool, [N, T, V, D] -> [N, V, D]
                x = x.max(dim=1).values  # global pool, [N, V, D] -> [N, D]
            else:
                x = x.mean(dim=1)
                x = x.mean(dim=1)
        else:  # if predicting stage per time step
            if self.max_pooling:
                x = x.max(dim=2).values  # global pool, [N, T, V, D] -> [N, T, D]
            else:
                x = x.mean(dim=2)  # global pool, [N, T, V, D] -> [N, T, D]
        x = self.norm(x)
        x = self.mlp_pred_dropout(x)
        x = self.head(x)  # -> [N, T, class_prob] if pred_per_T else [N, class_prob]
        return x


# uses convolution for time series prediction from encoder only representations,
# see https://github.com/nnzhan/Graph-WaveNet/blob/master/model.py
class GraphEncoderPred(GraphEncoder):
    """Autoencoder with Graphformer backbone for Time Series Prediction"""

    def __init__(
            self,
            n_pred=12,
            end_channel=512,
            use_conv=True,
            batch_norm=True,
            mlp_pred_dropout=0.5,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.n_pred = n_pred
        self.end_channel = end_channel
        self.use_conv = use_conv
        self.activation = nn.GELU() if self.act_fn == 'gelu' else nn.ReLU()
        old_config = kwargs.get('old_config', False)

        if self.n_pred > self.hist_t_dim:
            self.fc_project = nn.Linear(
                self.hist_t_dim * self.encoder_embed_dim,
                self.n_pred * end_channel
            )
            if batch_norm:
                self.layer_norm = nn.LayerNorm(end_channel, eps=1e-8)
            self.end_conv_2 = nn.Conv2d(
                in_channels=end_channel,
                out_channels=self.pred_node_dim,
                kernel_size=(1, 1),
                bias=True
            )
        else:
            if self.use_conv:
                if old_config:
                    self.end_conv_1 = nn.Conv2d(
                        in_channels=self.encoder_embed_dim,
                        out_channels=end_channel,
                        # kernel_size=(1, self.hist_t_dim - self.n_pred + 1),
                        kernel_size=(1, 1),
                        bias=True,
                    )
                else:
                    self.end_conv_1 = nn.Conv2d(
                        in_channels=self.encoder_embed_dim,
                        out_channels=end_channel,
                        # kernel_size=(1, self.hist_t_dim - self.n_pred + 1),
                        kernel_size=(1, 1),
                        bias=False if batch_norm else True,
                    )
                    if batch_norm:
                        self.batch_norm = nn.BatchNorm2d(end_channel)
                self.end_conv_2 = nn.Conv2d(
                    in_channels=end_channel,
                    out_channels=self.pred_node_dim,
                    kernel_size=(1, 1),
                    bias=True
                )
            else:
                # self.fc_his = nn.Sequential(
                #     nn.Linear(self.hist_t_dim, self.n_pred),
                #     nn.GELU()
                # )
                # self.mlp_pred_dropout = nn.Dropout(mlp_pred_dropout)
                # self.fc_channel = nn.Linear(
                #     self.encoder_embed_dim,
                #     self.pred_node_dim
                # )
                self.fc_channel = nn.Linear(self.encoder_embed_dim, self.pred_node_dim)
                self.mlp_pred_dropout = nn.Dropout(mlp_pred_dropout)
                self.fc_his = nn.Linear(
                    self.hist_t_dim,
                    self.n_pred
                )

        self.initialize_weights()
        print("model initialized")

    def forward_pred(self, x, x_shape):
        N, T, V, D = x_shape

        if self.n_pred > self.hist_t_dim:
            x = x.contiguous().view(N, V, T, D).view(N, V, T * D)  # [N, T, V, D] -> [N, V, T*D]
            x = self.fc_project(x)
            x = x.view(N, V, self.n_pred, self.end_channel)
            if hasattr(self, 'layer_norm'):
                x = self.layer_norm(x)
            x = self.activation(x)
            x = x.contiguous().transpose(1, 3)
            x = self.end_conv_2(x)  # [N, end, pred, V] -> [N, D, pred, V]
            x = x.contiguous().permute(0, 2, 3, 1)  # [N, D, pred, V] -> [N, pred, V, D]
        else:
            x = x.contiguous().view(N, T, V, D).transpose(1, 3)  # [N, T, V, D] -> [N, D, V, T]
            # predictor projects from decoder_embed_dim to node values
            if self.use_conv:

                x = self.end_conv_1(x)  # [N, D, V, T] -> [N, end, V, pred]
                if hasattr(self, 'batch_norm'):
                    x = self.batch_norm(x)
                x = self.activation(x)
                x = self.end_conv_2(x)  # [N, end, V, pred] -> [N, D, V, pred]
                x = x.transpose(1, 3)
                # # reshape output: [N, D, V, pred] -> [N, V*future_time, out]
                # x = x.contiguous().view(N, V * self.n_pred, self.pred_node_dim)
            else:
                x = self.fc_his(x).transpose(1, 3)  # [N, D, V, T] -> [N, pred, V, D]
                x = self.activation(x)
                x = self.mlp_pred_dropout(x)
                x = self.fc_channel(x)  # [N, pred, V, D] -> [N, pred, V, out]
        # return x.contiguous().view(N, self.n_pred * V, self.pred_node_dim)
        return x.contiguous().view(N, 12 * V, self.pred_node_dim)

    def forward_loss(self, pred, y):
        """
        y: [N, P, V, D]
        pred: [N, T*V, D]
        mask: [N, T, V], 0 is keep, 1 is remove,
        """
        N, P, V, D = y.shape
        assert V == self.num_nodes, P == self.n_pred
        y = y.contiguous().view(N, -1, D)

        loss = (pred - y) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per node
        loss = loss.mean()
        return loss

    def forward(self, batched_data):
        # compute attention biases and node centrality encodings
        x, attn_bias = self.blocks.compute_mods(batched_data)
        latent, _, x_shape = self.forward_encoder(x, attn_bias)
        latent = self.norm(latent)
        pred = self.forward_pred(latent, x_shape)  # [N, L, D]
        return pred


class GraphEncoderCausalPred(GraphEncoder):
    """Autoencoder with Graphormer backbone for Time Series Prediction"""

    def __init__(
            self,
            n_pred=12,
            end_channel=512,
            use_conv=True,
            kernel_size=3,
            batch_norm=True,
            mlp_pred_dropout=0.5,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.n_pred = n_pred
        self.end_channel = end_channel
        self.use_conv = use_conv
        self.activation = nn.GELU() if self.act_fn == 'gelu' else nn.ReLU()
        if not use_conv:
            raise ValueError('graph_causal_pred has to have --use_conv')

        self.dilated_conv = nn.ModuleList([
            CausalConv2d(
                in_channels=self.encoder_embed_dim if i == 0 else end_channel,
                out_channels=end_channel,
                kernel_size=(kernel_size, kernel_size),
                dilation=2 ** (i + 1),  # increase dilation exponentially
                batch_norm=batch_norm,
            )
            for i in range(3)  # three layers with increasing dilation
        ])
        if self.n_pred > self.hist_t_dim:
            self.fc_project = nn.Linear(
                self.hist_t_dim * end_channel,
                self.n_pred * (end_channel // 2)
            )
            if batch_norm:
                self.layer_norm = nn.LayerNorm(end_channel // 2, eps=1e-8)
            self.end_conv_2 = nn.Conv2d(
                in_channels=(end_channel // 2),
                out_channels=self.pred_node_dim,
                kernel_size=(1, 1),
                bias=True
            )
        else:
            self.end_conv_1 = nn.Conv2d(
                in_channels=self.end_channel,
                out_channels=end_channel,
                # kernel_size=(1, self.hist_t_dim - self.n_pred + 1),
                kernel_size=(1, 1),
                bias=False if batch_norm else True
            )
            if batch_norm:
                self.batch_norm = nn.BatchNorm2d(end_channel)
            self.end_conv_2 = nn.Conv2d(
                in_channels=end_channel,
                out_channels=self.pred_node_dim,
                kernel_size=(1, 1),
                bias=True
            )

        self.initialize_weights()
        print("model initialized")

    def forward_pred(self, x, x_shape):
        N, T, V, D = x_shape

        if self.n_pred > self.hist_t_dim:
            x = x.contiguous().view(N, V, T, D).transpose(1, 3)  # [N, V, T, D] -> [N, D, T, V]
            for conv_layer in self.dilated_conv:  # [N, D, T, V] -> [N, end_channel, T, V]
                x = conv_layer(x)  # apply each convolution layer
                x = self.activation(x)
            x = x.transpose(1, 3).contiguous().view(N, V, -1)  # [N, end_channel, T, V] -> [N, V, T*end]
            x = self.fc_project(x)  # [N, V, T*end] -> [N, V, T*end//2]
            x = x.view(N, V, self.n_pred, -1)
            if hasattr(self, 'layer_norm'):
                x = self.layer_norm(x)
            x = self.activation(x)
            x = x.contiguous().transpose(1, 3)  # [N, V, T*end//2, D] -> [N, D, T*end//2, V]
            x = self.end_conv_2(x)  # [N, end//2, P, V] -> [N, D, P, V]
            x = x.contiguous().permute(0, 2, 3, 1)
        else:
            x = x.contiguous().view(N, T, V, D).transpose(1, 3)  # [N, T, V, D] -> [N, D, V, T]
            for conv_layer in self.dilated_conv:  # [N, D, V, T] -> [N, end_channel, V, T]
                x = conv_layer(x)  # apply each convolution layer
                x = self.activation(x)
            x = self.end_conv_1(x)  # [N, end, V, T] -> [N, end, V, pred]
            if hasattr(self, 'batch_norm'):
                x = self.batch_norm(x)
            x = self.activation(x)
            x = self.end_conv_2(x)  # [N, end, V, pred] -> [N, D, V, pred]
            x = x.transpose(1, 3)
        return x.contiguous().view(N, 12 * V, self.pred_node_dim)

    def forward_loss(self, pred, y):
        """
        y: [N, P, V, D]
        pred: [N, T*V, D]
        mask: [N, T, V], 0 is keep, 1 is remove,
        """
        N, P, V, D = y.shape
        assert V == self.num_nodes, P == self.n_pred
        y = y.contiguous().view(N, -1, D)

        loss = (pred - y) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per node
        loss = loss.mean()
        return loss

    def forward(self, batched_data):
        # compute attention biases and node centrality encodings
        x, attn_bias = self.blocks.compute_mods(batched_data)
        latent, _, x_shape = self.forward_encoder(x, attn_bias)
        latent = self.norm(latent)
        pred = self.forward_pred(latent, x_shape)  # [N, L, D]
        return pred


# uses Seq2Seq, adds masked self-attention and cross encoder-decoder attention?
def graph_mlp_micro(**kwargs):
    model = GraphEncoderMLP(
        encoder_embed_dim=64,
        encoder_depth=6,
        num_heads=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-8),
        **kwargs,
    )
    return model


def graph_mlp_mini(**kwargs):
    model = GraphEncoderMLP(
        encoder_embed_dim=128,
        encoder_depth=6,
        num_heads=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-8),
        **kwargs,
    )
    return model


def graph_mlp_small(**kwargs):
    model = GraphEncoderMLP(
        encoder_embed_dim=192,
        encoder_depth=8,
        num_heads=6,
        norm_layer=partial(nn.LayerNorm, eps=1e-8),
        **kwargs,
    )
    return model


def graph_pred_micro(**kwargs):
    model = GraphEncoderPred(
        encoder_embed_dim=64,
        encoder_depth=6,
        num_heads=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-8),
        **kwargs,
    )
    return model


def graph_pred_mini(**kwargs):
    model = GraphEncoderPred(
        encoder_embed_dim=128,
        encoder_depth=6,
        num_heads=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-8),
        **kwargs,
    )
    return model


def graph_pred_small(**kwargs):
    model = GraphEncoderPred(
        encoder_embed_dim=192,
        encoder_depth=8,
        num_heads=6,
        norm_layer=partial(nn.LayerNorm, eps=1e-8),
        **kwargs,
    )
    return model


def graph_causal_pred_micro(**kwargs):
    model = GraphEncoderCausalPred(
        encoder_embed_dim=64,
        encoder_depth=6,
        num_heads=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-8),
        **kwargs,
    )
    return model


def graph_causal_pred_mini(**kwargs):
    model = GraphEncoderCausalPred(
        encoder_embed_dim=128,
        encoder_depth=6,
        num_heads=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-8),
        **kwargs,
    )
    return model


def graph_causal_pred_small(**kwargs):
    model = GraphEncoderCausalPred(
        encoder_embed_dim=192,
        encoder_depth=8,
        num_heads=6,
        norm_layer=partial(nn.LayerNorm, eps=1e-8),
        **kwargs,
    )
    return model

# def graph_causal_pred_med(**kwargs):
#     model = GraphEncoderCausalPred(
#         encoder_embed_dim=384,
#         encoder_depth=10,
#         num_heads=8,
#         norm_layer=partial(nn.LayerNorm, eps=1e-8),
#         **kwargs,
#     )
#     return model
#
#
# def graph_causal_pred_big(**kwargs):
#     model = GraphEncoderCausalPred(
#         encoder_embed_dim=768,
#         encoder_depth=12,
#         num_heads=12,
#         norm_layer=partial(nn.LayerNorm, eps=1e-8),
#         **kwargs,
#     )
#     return model
#
#
# def graph_causal_pred_large(**kwargs):
#     model = GraphEncoderCausalPred(
#         encoder_embed_dim=1024,
#         encoder_depth=24,
#         num_heads=16,
#         norm_layer=partial(nn.LayerNorm, eps=1e-8),
#         **kwargs,
#     )
#     return model
#
#
# def graph_causal_pred_xl(**kwargs):
#     model = GraphEncoderCausalPred(
#         encoder_embed_dim=1280,
#         encoder_depth=32,
#         num_heads=16,
#         norm_layer=partial(nn.LayerNorm, eps=1e-8),
#         **kwargs,
#     )
#     return model
