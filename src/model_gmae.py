from functools import partial

from fairseq import utils

import torch
import torch.nn as nn
from src.modules import graphormer_graph_encoder
from src.utils.log import master_print as print


class MaskedGraphAutoEncoder(nn.Module):
    """Masked AutoEncoder with Graphformer backbone"""

    def __init__(
            self,
            # < graph args
            node_feature_dim: int = 1,
            pred_node_dim: int = 1,
            num_nodes: int = 512 * 9,
            num_edges: int = 512 * 3,
            num_in_degree: int = 512,
            num_out_degree: int = 512,
            num_spatial: int = 512,  # longest edge distance between nodes, spatial encoding in graphormer
            num_edge_dis: int = 128,
            edge_type: str = 'multi_hop',
            multi_hop_max_dist: int = 5,
            static_graph=True,
            edge_features=False,
            n_hist=12,
            # >

            # < transformer args
            graph_token=True,
            cls_token=False,
            sep_pos_embed=True,
            attention_bias=True,
            centrality_encoding=True,
            num_heads=16,
            encoder_embed_dim=1024,
            encoder_depth=24,
            decoder_embed_dim=512,
            decoder_depth=8,
            dropout=0.1,
            pre_layernorm=True,
            norm_layer=nn.LayerNorm,
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
        self.decoder_embed_dim = decoder_embed_dim
        self.trunc_init = trunc_init
        self.n_hist = n_hist
        self.static_graph = static_graph
        self.num_nodes = num_nodes

        self.cls_token = cls_token
        self.decoder_cls_token = cls_token
        self.graph_token = graph_token
        self.decoder_graph_token = graph_token

        self.sep_pos_embed = sep_pos_embed
        self.attention_bias = attention_bias
        self.centrality_encoding = centrality_encoding
        self.act_fn = act_fn
        act_function = utils.get_activation_fn(self.act_fn)
        self.activation = act_function() if self.act_fn == 'swish' else act_function

        # encoder inits
        if self.graph_token:
            self.graph_token_embed = nn.Parameter(torch.zeros(1, 1, 1, encoder_embed_dim))
            self.decoder_graph_token_embed = nn.Parameter(torch.zeros(1, 1, 1, decoder_embed_dim))
            self.graph_token_virtual_distance = nn.Embedding(1, num_heads)
        if self.cls_token:
            assert self.graph_token is False, "cannot have both tokens"
            self.cls_token_embed = nn.Parameter(torch.zeros(1, 1, encoder_embed_dim))
            self.decoder_cls_token_embed = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
            self.cls_token_virtual_distance = nn.Embedding(1, num_heads)

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
            static_graph=self.static_graph,
            edge_features=edge_features,
            # for transformer blocks
            num_encoder_layers=encoder_depth,
            embedding_dim=encoder_embed_dim,
            ffn_embedding_dim=encoder_embed_dim * 4,
            num_attention_heads=num_heads,
            graph_token=self.graph_token,
            start_conv=True,
            centrality_encoding=self.centrality_encoding,
            attention_bias=self.attention_bias,
            pre_layernorm=pre_layernorm,
            activation_fn=act_fn,
            dropout=dropout,
        )
        self.norm = norm_layer(encoder_embed_dim)  # final encoder norm layer

        # decoder inits
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)

        self.decoder_blocks = graphormer_graph_encoder.GraphormerGraphEncoder(
            node_feature_dim=node_feature_dim,
            num_nodes=num_nodes,
            num_in_degree=num_in_degree,
            num_out_degree=num_out_degree,
            num_edges=num_edges,
            num_spatial=num_spatial,
            num_edge_dis=num_edge_dis,
            edge_type=edge_type,
            multi_hop_max_dist=multi_hop_max_dist,
            num_encoder_layers=decoder_depth,
            embedding_dim=decoder_embed_dim,
            ffn_embedding_dim=decoder_embed_dim * 4,
            num_attention_heads=num_heads,
            static_graph=self.static_graph,
            graph_token=self.graph_token,
            start_conv=False,
            centrality_encoding=self.centrality_encoding,
            attention_bias=False,
            pre_layernorm=pre_layernorm,
            activation_fn=act_fn,
            dropout=dropout,
        )
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_conv_1 = nn.Conv2d(
            in_channels=decoder_embed_dim,
            out_channels=decoder_embed_dim // 2,
            # kernel_size=(1, self.hist_t_dim - self.n_pred + 1),
            kernel_size=(1, 1),
            bias=False
        )
        self.decoder_batch_norm = nn.BatchNorm2d(decoder_embed_dim // 2)
        self.decoder_conv_2 = nn.Conv2d(
            in_channels=decoder_embed_dim // 2,
            out_channels=pred_node_dim,
            kernel_size=(1, 1),
            bias=True
        )

        if sep_pos_embed:
            self.pos_embed_time = nn.Parameter(
                torch.zeros(1, n_hist, encoder_embed_dim),
            )
            self.pos_embed_space = nn.Parameter(
                torch.zeros(1, num_nodes, encoder_embed_dim),
            )

            self.decoder_pos_embed_time = nn.Parameter(
                torch.zeros(1, n_hist, decoder_embed_dim),
            )
            self.decoder_pos_embed_space = nn.Parameter(
                torch.zeros(1, num_nodes, decoder_embed_dim),
            )
            if self.graph_token or self.cls_token:
                self.pos_embed_cls = nn.Parameter(torch.zeros(1, 1, encoder_embed_dim))
                self.decoder_pos_embed_cls = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        else:
            if self.graph_token:
                self.pos_embed = nn.Parameter(
                    torch.zeros(1, n_hist * (num_nodes + 1), encoder_embed_dim),
                )
                self.decoder_pos_embed = nn.Parameter(
                    torch.zeros(1, n_hist * (num_nodes + 1), decoder_embed_dim),
                )
            elif self.cls_token:
                self.pos_embed = nn.Parameter(
                    torch.zeros(1, n_hist * num_nodes + 1, encoder_embed_dim),
                )
                self.decoder_pos_embed = nn.Parameter(
                    torch.zeros(1, n_hist * num_nodes + 1, decoder_embed_dim),
                )
            else:
                self.pos_embed = nn.Parameter(
                    torch.zeros(1, n_hist * num_nodes, encoder_embed_dim),
                )
                self.decoder_pos_embed = nn.Parameter(
                    torch.zeros(1, n_hist * num_nodes, decoder_embed_dim),
                )

        self.initialize_weights()

        print("model initialized")

    def initialize_weights(self):
        if self.sep_pos_embed:
            torch.nn.init.trunc_normal_(self.pos_embed_time, std=0.02)
            torch.nn.init.trunc_normal_(self.pos_embed_space, std=0.02)
            torch.nn.init.trunc_normal_(self.decoder_pos_embed_time, std=0.02)
            torch.nn.init.trunc_normal_(self.decoder_pos_embed_space, std=0.02)
            if self.graph_token or self.cls_token:
                torch.nn.init.trunc_normal_(self.pos_embed_cls, std=0.02)
                torch.nn.init.trunc_normal_(self.decoder_pos_embed_cls, std=0.02)
        else:
            torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)
            torch.nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)

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
            torch.nn.init.trunc_normal_(self.graph_token_virtual_distance.weight, std=0.02)
            torch.nn.init.trunc_normal_(self.decoder_graph_token_embed, std=0.02)
        elif self.cls_token:
            torch.nn.init.trunc_normal_(self.cls_token_embed, std=0.02)
            torch.nn.init.trunc_normal_(self.cls_token_virtual_distance.weight, std=0.02)
            torch.nn.init.trunc_normal_(self.decoder_cls_token_embed, std=0.02)
        if self.trunc_init:
            torch.nn.init.trunc_normal_(self.mask_token, std=0.02)
        else:
            torch.nn.init.normal_(self.mask_token, std=0.02)

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

    def shuffle_attn_bias(self, attn_bias, ids_keep, x_shape, n_token):
        if not self.attention_bias:
            return None
        N, T, V, D = x_shape
        _, n_heads, n_nodes, _ = attn_bias.shape

        # reorder attn_bias by how x was shuffled by ids_keep,
        # see test_attn_bias_mask in test_gmae.py
        if self.graph_token:  # ids_keep has shape [N, n_tokens]
            ids_keep_ = ids_keep.detach().clone()
            ids_keep_ = ids_keep_ + 1

            # add shuffle indices for the graph tokens such that they are always kept
            ids_keep_ = torch.cat(
                tensors=[
                    torch.zeros(N, T, 1, dtype=ids_keep_.dtype, device=ids_keep_.device),
                    ids_keep_
                ],
                dim=2
            )
            assert ids_keep_.shape[1] * ids_keep_.shape[2] == n_token

            attn_bias = attn_bias.unsqueeze(2).expand(-1, -1, T, -1, -1).clone()  # expand the time dimensions
            # retrieve the attn bias vectors for each shuffled node at each time step
            # each vector contains all the nodes, shape is [n_batches, n_heads, T, nodes, n_tokens]
            attn_bias = torch.gather(
                attn_bias,
                dim=3,
                index=ids_keep_.unsqueeze(-1).unsqueeze(1).expand(-1, n_heads, -1, -1, V + 1)
            )

            # retrieve the node-node attn bias at each time step
            attn_bias = attn_bias.unsqueeze(-2).expand(-1, -1, -1, -1, T, -1).clone()
            attn_bias = torch.gather(
                attn_bias,
                dim=-1,
                # [n_batches, n_heads, T, n_nodes, T, n_nodes]
                index=ids_keep_.unsqueeze(1).unsqueeze(2).unsqueeze(1).
                expand(-1, n_heads, T, ids_keep_.shape[2], -1, -1)
            )
            # torch.all(torch.tensor([[attn_bias[0][0][1][163*i + j].eq(attn_bias_[0][0][124][ids_keep[0][i][j]]) for j in range(163)] for i in range(16)]))
        else:  # ids_keep has shape [N, n_tokens]
            attn_bias = attn_bias.repeat(1, 1, T, T)  # expand the time dimensions

            # retrieve the attn bias vectors for each shuffled node at each time step
            # each vector contains all the nodes, shape is [n_batches, n_heads, T, nodes, n_tokens]
            attn_bias = torch.gather(
                attn_bias,
                dim=2,
                index=ids_keep.unsqueeze(-1).unsqueeze(1).expand(-1, n_heads, -1, T * V)
            )
            # retrieve the node-node attn bias at each time step
            attn_bias = torch.gather(
                attn_bias,
                dim=3,
                index=ids_keep.unsqueeze(1).unsqueeze(2).expand(-1, n_heads, ids_keep.shape[1], -1)
            )
            n_token = n_token - 1 if self.cls_token else n_token
        return attn_bias.contiguous().view(N, n_heads, n_token, n_token)

    def random_masking(self, x, mask_ratio: float):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        Based on the sampled noise per batch, we align it with the input to rank and shuffle tokens.
        If the masking ratio is 50%, then the first 50% of the sorted tokens are kept.
        Since this does not preserve the order of the input,
        we keep track of how the processed input can be unshuffled and restored for calculating training loss.
        x: [N, n_tokens, D] if no graph token at every time step, otherwise [N, T, V, D]
        outputs:
         x_masked: [N, T*V*mask_ratio, D],
         mask,
         ids_restore,
         ids_keep (shape [N, T*V*(1-mask_ratio)] if no graph token, [N, T, V*(1-mask_ratio)] otherwise)
        """

        if self.graph_token:
            N, T, V, D = x.shape  # batch, time, nodes, dim
            len_keep = int(V * (1 - mask_ratio))
            noise = torch.rand(N, T, V, device=x.device)
            # sort noise for each sample
            ids_shuffle = torch.argsort(
                noise, dim=2
            )  # ascend: small is keep, large is remove
            # this is the rank of the noise
            ids_restore = torch.argsort(ids_shuffle, dim=2)
            # keep the first subset
            ids_keep = ids_shuffle[:, :, :len_keep]
            # select the original xs that corresponds to small noise
            # note that order is not preserved
            x_masked = torch.gather(x, dim=2,
                                    index=ids_keep.unsqueeze(-1).expand(-1, -1, -1, D))  # match the last D dim
            mask = torch.ones([N, T, V], device=x.device)
            mask[:, :, len_keep] = 0
            # unshuffle to get the binary mask
            mask = torch.gather(mask, dim=2, index=ids_restore)
        else:  # cls_token masking
            N, L, D = x.shape  # batch, length, dim
            len_keep = int(L * (1 - mask_ratio))

            noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

            # sort noise for each sample
            ids_shuffle = torch.argsort(
                noise, dim=1
            )  # ascend: small is keep, large is remove
            ids_restore = torch.argsort(ids_shuffle, dim=1)

            # keep the first subset
            ids_keep = ids_shuffle[:, :len_keep]
            # select the original xs that corresponds to small noise
            # note that order is not preserved
            x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))

            # generate the binary mask: 0 is keep, 1 is removed
            mask = torch.ones([N, L], device=x.device)
            mask[:, :len_keep] = 0
            # unshuffle to get the binary mask
            mask = torch.gather(mask, dim=1, index=ids_restore)

        del noise, ids_shuffle
        return x_masked, mask, ids_restore, ids_keep

    def add_token_distance(self, cls, attn_bias: torch.Tensor, device: torch.device):
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
            t = self.cls_token_virtual_distance.weight.repeat(N, 1).view(-1, n_heads, 1)
            attn_bias[:, :, 1:, 0] = attn_bias[:, :, 1:, 0].clone() + t
            attn_bias[:, :, 0, :] = attn_bias[:, :, 0, :].clone() + t
        else:
            t = self.graph_token_virtual_distance.weight.repeat(N, 1).view(-1, n_heads, 1)
            attn_bias[:, :, 1:, 0] = attn_bias[:, :, 1:, 0].clone() + t
            attn_bias[:, :, 0, :] = attn_bias[:, :, 0, :].clone() + t
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

    def forward_encoder(self, x: torch.Tensor, attn_bias: torch.Tensor, mask_ratio: float = 0.5):
        """

        :param x: time series data [Batch, Time_step, Vertices, Embed_Dim]
        :param attn_bias: attention bias computed based on graph structure and edge
        lengths [Batch, n_heads, Vertices, Vertices]
        :param mask_ratio: how much masking is dones on nodes, time and space agnostic
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

        # mask tokens, add cls_tokens, and get positional embeddings based on the masking indices
        if self.graph_token:
            # 1 graph token for each time step, similar to segment token in NLP
            # masks based on each time_step, ids_keep shape: [n_batches, T, V*mask_ratio]
            x, mask, ids_restore, ids_keep = self.random_masking(x, mask_ratio)

            # repeating since graph tokens stay the same through time dimension
            graph_token_feature = self.graph_token_embed.expand(N, T, -1, -1)
            x = torch.cat([graph_token_feature, x], dim=2)
            x = x.contiguous().view(N, -1, D)

            n_tokens = x.shape[1]
            assert n_tokens == T * (int(V * (1 - mask_ratio)) + 1), \
                f'{n_tokens} does not match {T} * ({V} * (1-{mask_ratio})+1)'

            if self.sep_pos_embed:
                pos_embed = self.pos_embed_space.repeat(
                    1, T, 1
                ) + torch.repeat_interleave(
                    self.pos_embed_time,
                    V,
                    dim=1
                )
                pos_embed = pos_embed.contiguous().expand(N, -1, -1).view(N, T, V, D)
                pos_embed_token = self.pos_embed_cls.expand(N, T, -1, -1)
            else:
                pos_embed = self.pos_embed.contiguous().expand(N, -1, -1).view(N, T, V + 1, D)
                pos_embed = pos_embed[:, :, 1:, :]
                pos_embed_token = pos_embed[:, :, :1, :]

            pos_embed_x = torch.gather(
                pos_embed,
                dim=2,
                index=ids_keep.unsqueeze(-1).expand(-1, -1, -1, D)
            )  # gather the positional embeddings for unmasked tokens
            pos_embed = torch.cat(
                [pos_embed_token, pos_embed_x],
                dim=2
            )  # add positional embedding for graph tokens
            attn_bias = self.add_token_distance(
                cls=False,
                attn_bias=attn_bias,
                device=x.device
            )
        else:
            # 1 graph token for all time steps (cls token) or no graph token
            x = x.contiguous().view(N, -1, D)  # flatten the time dimension
            # masking: length -> length * mask_ratio
            x, mask, ids_restore, ids_keep = self.random_masking(x, mask_ratio)
            x = x.contiguous().view(N, -1, D)

            # concatenate the cls_token
            if self.cls_token:
                cls_token_feature = self.cls_token_embed
                cls_token_feature = cls_token_feature.expand(N, -1, -1)
                x = torch.cat((cls_token_feature, x), dim=1)
                cls_ind = 1
            else:
                cls_ind = 0

            n_tokens = x.shape[1]
            if self.cls_token:
                assert n_tokens == int(T * V * (1 - mask_ratio)) + 1, \
                    f'{n_tokens} does not match ({T} * {V} * 1- {mask_ratio} + 1) '
            else:
                assert n_tokens == int(T * V * (1 - mask_ratio)), \
                    f'{n_tokens} does not match ({T} * {V} * 1-{mask_ratio}) '

            if self.sep_pos_embed:
                pos_embed = self.pos_embed_space.repeat(
                    1, T, 1
                ) + torch.repeat_interleave(
                    self.pos_embed_time,
                    V,
                    dim=1
                )
                pos_embed = pos_embed.expand(N, -1, -1)
                if self.cls_token:
                    pos_embed_token = self.pos_embed_cls.expand(N, -1, -1)
            else:
                pos_embed = self.pos_embed[:, cls_ind:, :].expand(N, -1, -1)
                pos_embed_token = self.pos_embed[:, :1, :].expand(N, -1, -1)
            pos_embed = torch.gather(
                pos_embed,
                dim=1,
                index=ids_keep.unsqueeze(-1).expand(-1, -1, pos_embed.shape[2]),
            )
            if self.cls_token:
                pos_embed = torch.cat(
                    [pos_embed_token, pos_embed],
                    1,
                )  # add pos_embed for cls_token

        x = x + pos_embed.contiguous().view(N, -1, D)

        # retrieve attn_bias for unmasked tokens based on indices provided by ids_keep
        attn_bias = self.shuffle_attn_bias(attn_bias, ids_keep, x_shape, n_tokens) if self.attention_bias else None

        if self.cls_token:
            attn_bias = self.add_token_distance(
                cls=True,
                attn_bias=attn_bias,
                device=x.device
            )

        # apply Transformer blocks
        inner_states = self.blocks(x, attn_bias)
        x = inner_states[-1].contiguous().transpose(0, 1)
        x = self.norm(x)
        # TODO: add eigenvector PE
        graph_rep, x = self.get_graph_rep(x, x_shape)
        x = x.contiguous().view(N, -1, D)
        return x, graph_rep, mask, ids_restore, x_shape

    def forward_decoder(self, x, ids_restore, x_shape, attn_bias: torch.Tensor):
        N, T, V, D = x_shape
        if not self.attention_bias:
            del attn_bias
            attn_bias = None
        # embed tokens
        x = self.decoder_embed(x)
        D = x.shape[-1]
        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(N, T * V - x.shape[1], 1)
        x = torch.cat([x[:, :, :], mask_tokens], dim=1)  # no cls token

        # unshuffle and recover the original input
        if self.decoder_graph_token:
            assert ids_restore.shape[1] * ids_restore.shape[2] == T * V
            x = x.contiguous().view(N, T, V, D)
            x = torch.gather(
                x, dim=2, index=ids_restore.unsqueeze(-1).expand(-1, -1, -1, D)
            )  # unshuffle
            graph_token_feature = self.decoder_graph_token_embed.expand(N, T, -1, -1)
            x = torch.cat([graph_token_feature, x], dim=2)
            x = x.contiguous().view(N, -1, D)

            attn_bias = self.add_token_distance(
                cls=False,
                attn_bias=attn_bias,
                device=x.device
            )
            attn_bias = attn_bias.repeat(1, 1, T, T) if self.attention_bias else None
        elif self.decoder_cls_token:
            assert ids_restore.shape[-1] == T * V
            x = x.contiguous().view(N, T * V, D)
            x = torch.gather(
                x, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, D)
            )
            cls_embed = self.decoder_cls_token_embed
            cls_embeds = cls_embed.expand(N, -1, -1)
            x = torch.cat((cls_embeds, x), dim=1)
            x = x.contiguous().view(N, T * V + 1, D)

            # fix attn_bias
            attn_bias = attn_bias.repeat(1, 1, T, T) if self.attention_bias else None
            attn_bias = self.add_token_distance(
                cls=True,
                attn_bias=attn_bias,
                device=x.device
            )
        else:
            x = x.contiguous().view(N, T * V, D)
            x = torch.gather(
                x, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, D)
            )
            x = x.contiguous().view(N, T * V, D)
            attn_bias = attn_bias.repeat(1, 1, T, T) if self.attention_bias else None

        if self.sep_pos_embed:
            decoder_pos_embed = self.decoder_pos_embed_space.repeat(
                1, T, 1
            ) + torch.repeat_interleave(
                self.decoder_pos_embed_time,
                V,
                dim=1
            )
            decoder_pos_embed = decoder_pos_embed.expand(N, -1, -1)
            if self.graph_token:
                decoder_pos_embed_token = self.decoder_pos_embed_cls.expand(N, T, -1, -1)
                decoder_pos_embed = torch.cat(
                    [
                        decoder_pos_embed_token,
                        decoder_pos_embed.contiguous().view(N, T, V, D),
                    ],
                    dim=2
                )
            elif self.cls_token:
                decoder_pos_embed_token = self.decoder_pos_embed_cls.expand(N, -1, -1)
                decoder_pos_embed = torch.cat(
                    [decoder_pos_embed_token, decoder_pos_embed],
                    dim=1
                )
        else:
            decoder_pos_embed = self.decoder_pos_embed[:, :, :].expand(N, -1, -1)
        # add pos embed
        x = x + decoder_pos_embed.contiguous().view(N, -1, D)

        # apply Transformer blocks with attn bias
        inner_states = self.decoder_blocks(x, attn_bias)
        x = inner_states[-1].contiguous().transpose(0, 1)
        x = self.decoder_norm(x)
        # [N, T * num_nodes, decoder_embed_dim]
        # predictor projects from decoder_embed_dim to node values

        _, x = self.get_graph_rep(x, (N, T, V, self.decoder_embed_dim))

        x = x.contiguous().view(N, T, V, -1).transpose(1, 3)  # [N, T, V, D] -> [N, D, V, T]
        x = self.decoder_conv_1(x)  # [N, D, V, T] -> [N, end, V, T]
        x = self.decoder_batch_norm(x)
        x = self.activation(x)
        x = self.decoder_conv_2(x)  # [N, end, V, T] -> [N, D, V, T]
        x = x.transpose(1, 3)  # [N, D, V, T] -> [N, T, V, D]

        x = x.contiguous().view(N, -1, self.pred_node_dim)
        return x

    def forward_loss(self, orig_x, pred, mask, scaler):
        """
        orig_x: [N, T, V, D]
        pred: [N, T*V, D]
        mask: [N, T, V], 0 is keep, 1 is remove,
        """
        N, T, V, D = orig_x.shape
        assert V == self.num_nodes, T == self.n_hist

        if self.node_feature_dim != self.pred_node_dim:
            if self.pred_node_dim == 1:  # only calculate loss on sensor data
                orig_x = orig_x[..., [0]]

        if scaler is not None:
            orig_x = scaler.inverse_transform(orig_x)
            pred = scaler.inverse_transform(pred)

        orig_x = orig_x.contiguous().view(N, -1, self.pred_node_dim)
        loss = (pred - orig_x) ** 2
        loss = loss.mean(dim=-1)  # [N, n_tokens], mean loss per node

        mask = mask.view(loss.shape)
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed nodes
        return loss

    def forward(self, batched_data, mask_ratio=0.75):
        # compute attention biases and node centrality encodings
        x, attn_bias = self.blocks.compute_mods(batched_data)
        latent, _, mask, ids_restore, x_shape = self.forward_encoder(
            x,
            attn_bias.clone() if self.attention_bias else None,
            mask_ratio
        )

        # decoder to recover masked nodes
        pred = self.forward_decoder(latent, ids_restore, x_shape, attn_bias)  # [N, n_tokens, D]
        loss = self.forward_loss(batched_data['x'], pred, mask, batched_data['scaler'])
        return loss, pred, mask


def mae_graph_debug(**kwargs):
    model = MaskedGraphAutoEncoder(
        encoder_embed_dim=16,
        encoder_depth=12,
        num_heads=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-8),
        **kwargs,
    )
    return model


def mae_graph_mini(**kwargs):
    model = MaskedGraphAutoEncoder(
        encoder_embed_dim=128,
        encoder_depth=6,
        num_heads=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-8),
        **kwargs,
    )
    return model


def mae_graph_small(**kwargs):
    model = MaskedGraphAutoEncoder(
        encoder_embed_dim=192,
        encoder_depth=8,
        num_heads=6,
        norm_layer=partial(nn.LayerNorm, eps=1e-8),
        **kwargs,
    )
    return model

#
# def mae_graph_med(**kwargs):
#     model = MaskedGraphAutoEncoder(
#         encoder_embed_dim=384,
#         encoder_depth=10,
#         num_heads=8,
#         norm_layer=partial(nn.LayerNorm, eps=1e-8),
#         **kwargs,
#     )
#     return model
#
#
# def mae_graph_big(**kwargs):
#     model = MaskedGraphAutoEncoder(
#         encoder_embed_dim=768,
#         encoder_depth=12,
#         num_heads=12,
#         norm_layer=partial(nn.LayerNorm, eps=1e-8),
#         **kwargs,
#     )
#     return model
#
#
# def mae_graph_large(**kwargs):
#     model = MaskedGraphAutoEncoder(
#         encoder_embed_dim=1024,
#         encoder_depth=24,
#         num_heads=16,
#         norm_layer=partial(nn.LayerNorm, eps=1e-8),
#         **kwargs,
#     )
#     return model
#
#
# def mae_graph_xl(**kwargs):
#     model = MaskedGraphAutoEncoder(
#         encoder_embed_dim=1280,
#         encoder_depth=32,
#         num_heads=16,
#         norm_layer=partial(nn.LayerNorm, eps=1e-8),
#         **kwargs,
#     )
#     return model
