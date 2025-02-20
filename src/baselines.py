import math
from functools import partial
from typing import List

import torch
import torch.nn as nn

import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv, GraphConv, SAGEConv, ChebConv, GATConv,
    SGConv, GatedGraphConv,
    global_max_pool, global_mean_pool,
    TopKPooling
)

from src.modules.baseline import STConv, DCRNN_Layer

GRAPH_CONV_DICT = {
    'gcn': GCNConv,  # Graph Convolutional Network
    # 'graphsage': SAGEConv,  # GraphSAGE
    'cheb': partial(ChebConv, K=3),  # Chebyshev Convolution
    # 'gat': GATConv,  # Graph Attention Network
    'sg': SGConv,  # Simplified Graph Convolution
    'graphconv': GraphConv,  # Graph Convolution (Kipf & Welling)
    'gated': GatedGraphConv,  # Gated Graph Convolution
}


class GCNMLP(nn.Module):
    def __init__(
            self,
            node_feature_dim: int = 1,
            pred_node_dim: int = 1,
            num_nodes: int = 512 * 9,
            pred_num_classes: int = 3,
            mlp_pred_dropout: float = 0.1,
            class_init_prob: List[float] = None,
            encoder_embed_dim: int = 512,
            encoder_depth=3,
            dropout=0.1,
            max_pooling=True,
            gcn_type='gcn',
            norm_layer=nn.LayerNorm,
            trunc_init=True,
            *args,
            **kwargs,
    ):
        super().__init__()
        if class_init_prob is None:
            class_init_prob = [1 / pred_num_classes] * pred_num_classes

        assert len(class_init_prob) == pred_num_classes
        assert math.isclose(sum(class_init_prob), 1.0), 'class probabilities must sum to 1'
        assert gcn_type in GRAPH_CONV_DICT.keys(), f'conv_type must be one of {GRAPH_CONV_DICT.keys()}'
        self.conv_type = gcn_type
        self.conv_depth = encoder_depth
        self.max_pooling = max_pooling
        self.pred_node_dim = pred_node_dim
        self.pred_num_classes = pred_num_classes
        self.class_init_prob = class_init_prob
        self.num_nodes = num_nodes
        self.node_feature_dim = node_feature_dim
        self.encoder_embed_dim = encoder_embed_dim
        self.trunc_init = trunc_init
        self.norm = norm_layer(encoder_embed_dim)
        self.mlp_pred_dropout = nn.Dropout(mlp_pred_dropout)
        self.dropout_module = nn.Dropout(dropout)
        self.head = nn.Linear(self.encoder_embed_dim, pred_num_classes)

        graph_conv = GRAPH_CONV_DICT[gcn_type]
        self.convs = torch.nn.ModuleList([
            graph_conv(
                self.node_feature_dim if i == 0 else encoder_embed_dim,
                encoder_embed_dim
            )
            for i in range(encoder_depth)
        ])
        self.initialize_weights()
        torch.nn.init.normal_(self.head.weight, std=2e-5)
        bias_values = torch.log(torch.FloatTensor(class_init_prob))
        self.head.bias.data = bias_values

    @torch.jit.ignore
    def no_weight_decay(self):
        return dict()

    def initialize_weights(self):
        # initialize nn.Linear and nn.LayerNorm
        for conv in self.convs:
            conv.reset_parameters()
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

    def forward(self, data):
        x, edge_index, edge_weight = data['x'], data['edge_index'], data['edge_weight']
        x_shape = x.shape
        N, T, V, D = x_shape
        if T > 1:
            raise ValueError('GNN classification only supports 1 time step')

        # Flatten the batch dimension, so x becomes [N*V, D]
        x = x.view(N * V, D)

        # Flatten edge_index for each graph in the batch
        edge_index = edge_index.view(2, -1)  # [2, N*E]
        edge_weight = edge_weight.view(-1)

        # Shift the edge indices to account for batch-wise node indexing
        batch_offsets = torch.arange(N, device=edge_index.device) * V
        edge_index[0] += torch.repeat_interleave(batch_offsets, edge_index.size(1) // N)
        edge_index[1] += torch.repeat_interleave(batch_offsets, edge_index.size(1) // N)

        # Create a batch tensor indicating the graph each node belongs to
        batch = torch.repeat_interleave(torch.arange(N, device=x.device), V)

        # get node representations
        # shape: [N, V, D] -> [N, V, end_channel]
        for conv in self.convs:
            x = conv(x, edge_index, edge_weight)
            x = F.relu(x)
            x = self.norm(x)
            x = self.dropout_module(x)

        if self.max_pooling:
            # Global max pooling
            x = global_max_pool(x, batch)  # [N, D]
        else:
            # Global mean pooling
            x = global_mean_pool(x, batch)  # [N, D]

        x = self.norm(x)
        x = self.mlp_pred_dropout(x)
        x = self.head(x)  # -> [N, T, class_prob] if pred_per_T else [N, class_prob]
        return x


class GCNMae(nn.Module):
    def __init__(
            self,
            node_feature_dim: int = 1,
            num_nodes: int = 68,
            encoder_embed_dim: int = 512,
            latent_dim: int = 512,
            encoder_depth=3,
            pool_ratio=0.5,
            dropout=0.1,
            gcn_type='gcn',
            norm_layer=nn.LayerNorm,
            trunc_init=True,
            *args,
            **kwargs,
    ):
        super().__init__()
        assert gcn_type in GRAPH_CONV_DICT.keys(), f'conv_type must be one of {GRAPH_CONV_DICT.keys()}'
        self.conv_type = gcn_type

        self.num_nodes = num_nodes
        self.node_feature_dim = node_feature_dim
        self.encoder_embed_dim = encoder_embed_dim
        self.latent_dim = latent_dim

        self.trunc_init = trunc_init
        self.dropout_module = nn.Dropout(dropout)

        graph_conv = GRAPH_CONV_DICT[gcn_type]
        assert (latent_dim & (latent_dim - 1) == 0), 'latent_dim must be a power of 2'
        assert (encoder_embed_dim & (encoder_embed_dim - 1) == 0), 'encoder_embed_dim must be a power of 2'
        assert (encoder_embed_dim * num_nodes) // 2 > latent_dim, \
            'latent_dim must be less than half of encoder_embed_dim'

        # Make sure final encoder_embed_dim is bigger than latent dim
        max_depth = math.log(encoder_embed_dim, 2)
        while 2 ** (max_depth - encoder_depth + 1) < latent_dim:
            encoder_depth -= 1
        self.encoder_depth = encoder_depth

        encoder_dims = [
            (encoder_embed_dim // (2 ** i), encoder_embed_dim // (2 ** (i + 1)))
            for i in range(encoder_depth - 1)
        ]
        encoder_dims = [(node_feature_dim, encoder_embed_dim)] + encoder_dims
        self.convs = torch.nn.ModuleList([
            graph_conv(encoder_dims[i][0], encoder_dims[i][1])
            for i in range(encoder_depth)
        ])
        encoder_final_dim = encoder_dims[-1][-1]
        self.encoder_final_dim = encoder_final_dim

        self.pool_ratio = pool_ratio
        self.pooling = TopKPooling(encoder_dims[-1][-1], ratio=pool_ratio)
        seq_dim = encoder_final_dim * math.ceil(num_nodes * pool_ratio)
        self.seq_dim = seq_dim

        self.head = nn.Linear(seq_dim, latent_dim)
        self.norm = norm_layer(latent_dim)

        output_dim = node_feature_dim * num_nodes
        self.decoder_head = nn.Linear(latent_dim, output_dim)
        self.decoder_norm = norm_layer(output_dim)
        self.decoder_convs = torch.nn.ModuleList([
            graph_conv(node_feature_dim, node_feature_dim)
            for _ in range(math.floor(encoder_depth // 2))
        ])
        self.initialize_weights()

    @torch.jit.ignore
    def no_weight_decay(self):
        return dict()

    def initialize_weights(self):
        # initialize nn.Linear and nn.LayerNorm
        for conv in self.convs:
            conv.reset_parameters()
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

    def forward(self, data):
        x, edge_index, edge_weight = data['x'], data['edge_index'], data['edge_attr']
        x_shape = x.shape
        N, T, V, D = x_shape
        if T > 1:
            raise ValueError('GNN only supports 1 time step')

        # Flatten the batch dimension, so x becomes [N*V, D]
        x = x.view(N * V, D)

        # Flatten edge_index for each graph in the batch
        edge_index = edge_index.view(2, -1)  # [2, N*E]
        edge_weight = edge_weight.view(-1)

        # Shift the edge indices to account for batch-wise node indexing
        batch_offsets = torch.arange(N, device=edge_index.device) * V
        edge_index[0] += torch.repeat_interleave(batch_offsets, edge_index.size(1) // N)
        edge_index[1] += torch.repeat_interleave(batch_offsets, edge_index.size(1) // N)

        # Create a batch tensor indicating the graph each node belongs to
        batch = torch.repeat_interleave(torch.arange(N, device=x.device), V)

        # get node representations
        # shape: [N, V, D] -> [N, V, end_channel]
        for conv in self.convs:
            x = conv(x, edge_index, edge_weight)
            x = F.relu(x)
            x = self.dropout_module(x)

        x, _, _, _, _, _ = self.pooling(x, edge_index, edge_weight, batch)
        x = x.contiguous().view(N, -1)
        x = self.head(x)
        x = self.norm(x)

        x = F.relu(self.decoder_head(x))
        x = self.decoder_norm(x)
        x = x.view(N * V, D)
        for conv in self.decoder_convs:
            x = conv(x, edge_index, edge_weight)
            x = F.relu(x)
        return x.view(N, V, D)

    def forward_loss(self, data, criterion=torch.nn.MSELoss()):
        pred = self.forward(data)
        y = data['x']
        N, T, V, D = y.shape
        loss = criterion(pred, y.view(N, -1))
        return loss


class TimeSeriesPred(nn.Module):
    def __init__(
            self,
            num_nodes: int = 68,
            node_feature_dim: int = 1,
            pred_node_dim: int = 1,
            n_hist=1,
            n_pred=2,
            end_channel=512,
            norm_layer=nn.LayerNorm,
            batch_norm=True,
            act_fn='relu',
            encoder_embed_dim=1024,
            mlp_pred_dropout=0.5,
            *args,
            **kwargs,
    ):
        super().__init__()
        self.node_feature_dim = node_feature_dim
        self.num_nodes = num_nodes
        self.pred_node_dim = pred_node_dim

        self.hist_t_dim = n_hist
        self.n_pred = n_pred
        self.end_channel = end_channel
        self.batch_norm = batch_norm
        self.encoder_embed_dim = encoder_embed_dim
        self.mlp_pred_dropout = mlp_pred_dropout

        self.activation = nn.GELU() if act_fn == 'gelu' else nn.ReLU()
        self.norm = norm_layer(encoder_embed_dim)
        if self.n_pred > self.hist_t_dim:
            self.fc_project = nn.Linear(
                self.hist_t_dim * encoder_embed_dim,
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
            self.end_conv_1 = nn.Conv2d(
                in_channels=self.encoder_embed_dim,
                out_channels=end_channel,
                kernel_size=(1, self.hist_t_dim - self.n_pred + 1),
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

    @staticmethod
    def process_static_edge_info(edge_index, edge_attr):
        # make sure edge_index and edge_attr across batches are the same
        if not (len(edge_attr.shape) > 1 and len(edge_index.shape) > 2):
            return edge_index, edge_attr
        batch_size = edge_index.shape[0]
        assert all(torch.all(edge_index[0] == edge_index[i]) for i in range(batch_size - 1))
        assert all(torch.all(edge_attr[0] == edge_attr[i]) for i in range(batch_size - 1))

        return edge_index[0], edge_attr[0]

    def forward_encoder(self, batched_data):
        raise NotImplementedError

    def forward_end_conv(self, x, x_shape):
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
            x = self.end_conv_1(x)  # [N, D, V, T] -> [N, end, V, pred]
            if hasattr(self, 'batch_norm'):
                x = self.batch_norm(x)
            x = self.activation(x)
            x = self.end_conv_2(x)  # [N, end, V, pred] -> [N, D, V, pred]
            x = x.transpose(1, 3)
            # # reshape output: [N, D, V, pred] -> [N, V*future_time, out]
            # x = x.contiguous().view(N, V * self.n_pred, self.pred_node_dim)
        return x.contiguous().view(N, self.n_pred * V, self.pred_node_dim)

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


class DCRNN(TimeSeriesPred):
    def __init__(
            self,
            encoder_depth: int = 3,
            K: int = 3,
            bias: bool = True,
            dropout: float = 0.1,
            trunc_init=True,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.trunc_init = trunc_init
        self.dcrnn_layers = nn.ModuleList([
            DCRNN_Layer(
                in_channels=int(self.hist_t_dim * self.node_feature_dim)
                if i == 0 else (self.hist_t_dim * self.encoder_embed_dim),
                out_channels=(self.hist_t_dim * self.encoder_embed_dim),
                K=K,
                bias=bias,
            )
            for i in range(encoder_depth)
        ])
        self.dropout_module = nn.Dropout(dropout)
        self.initialize_weights()

    @torch.jit.ignore
    def no_weight_decay(self):
        return dict()

    def initialize_weights(self):
        for layer in self.dcrnn_layers:
            layer.reset_parameters()
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def forward_encoder(self, batched_data):
        x = batched_data['x']
        edge_index, edge_weight = batched_data['edge_index'], batched_data['edge_attr']
        edge_index, edge_weight = self.process_static_edge_info(edge_index, edge_weight)
        x_shape = x.shape
        N, T, V, D = x_shape

        # Flatten the time and feature dimension, so x becomes [N, V, T*D]
        x = x.view(N, V, T * D)
        for layer in self.dcrnn_layers:
            x = layer(x, edge_index, edge_weight)
            x = self.activation(x)
            x = self.dropout_module(x)
        x = x.contiguous().view(N, T, V, self.encoder_embed_dim)
        return x

    def forward(self, batched_data):
        x = self.forward_encoder(batched_data)
        x = self.norm(x)
        pred = self.forward_end_conv(x, x.shape)
        return pred


class STGCN(TimeSeriesPred):
    def __init__(
            self,
            encoder_depth: int = 3,
            kernel_size: int = 1,
            K: int = 3,
            dropout: float = 0.1,
            trunc_init=True,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.trunc_init = trunc_init
        self.dropout_module = nn.Dropout(dropout)
        self.stconv_layers = nn.ModuleList([
            STConv(
                num_nodes=self.num_nodes,
                in_channels=self.node_feature_dim if i == 0 else self.encoder_embed_dim,
                hidden_channels=self.encoder_embed_dim * 2,
                out_channels=self.encoder_embed_dim,
                kernel_size=kernel_size,
                K=K,
            )
            for i in range(encoder_depth)
        ])
        self.initialize_weights()

    @torch.jit.ignore
    def no_weight_decay(self):
        return dict()

    def initialize_weights(self):
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def forward_encoder(self, batched_data):
        x = batched_data['x']
        edge_index, edge_weight = batched_data['edge_index'], batched_data['edge_attr']
        edge_index, edge_weight = self.process_static_edge_info(edge_index, edge_weight)
        x_shape = x.shape
        N, T, V, D = x_shape

        for layer in self.stconv_layers:
            x = layer(x, edge_index, edge_weight)
            x = self.activation(x)
            x = self.dropout_module(x)
        x = x.contiguous().view(N, T, V, self.encoder_embed_dim)
        return x

    def forward(self, batched_data):
        x = self.forward_encoder(batched_data)
        x = self.norm(x)
        pred = self.forward_end_conv(x, x.shape)
        return pred


def gae_micro(**kwargs):
    model = GCNMae(
        encoder_embed_dim=64,
        latent_dim=8,
        encoder_depth=6,
        pool_ratio=0.7,
        norm_layer=partial(nn.LayerNorm, eps=1e-8),
        **kwargs
    )
    return model


def gae_mini(**kwargs):
    model = GCNMae(
        encoder_embed_dim=128,
        latent_dim=8,
        encoder_depth=6,
        pool_ratio=0.7,
        norm_layer=partial(nn.LayerNorm, eps=1e-8),
        **kwargs
    )
    return model


def gae_small(**kwargs):
    model = GCNMae(
        encoder_embed_dim=256,
        latent_dim=8,
        encoder_depth=6,
        pool_ratio=0.7,
        norm_layer=partial(nn.LayerNorm, eps=1e-8),
        **kwargs
    )
    return model


def gnn_mlp_mini(**kwargs):
    model = GCNMLP(
        encoder_embed_dim=128,
        encoder_depth=6,
        norm_layer=partial(nn.LayerNorm, eps=1e-8),
        **kwargs
    )
    return model


def gnn_mlp_small(**kwargs):
    model = GCNMLP(
        encoder_embed_dim=192,
        encoder_depth=8,
        norm_layer=partial(nn.LayerNorm, eps=1e-8),
        **kwargs
    )
    return model


def gnn_mlp_med(**kwargs):
    model = GCNMLP(
        encoder_embed_dim=384,
        encoder_depth=10,
        norm_layer=partial(nn.LayerNorm, eps=1e-8),
        **kwargs
    )
    return model


def DCRNN_mini(**kwargs):
    model = DCRNN(
        encoder_embed_dim=128,
        encoder_depth=6,
        norm_layer=partial(nn.LayerNorm, eps=1e-8),
        **kwargs
    )
    return model


def DCRNN_small(**kwargs):
    model = DCRNN(
        encoder_embed_dim=192,
        encoder_depth=8,
        norm_layer=partial(nn.LayerNorm, eps=1e-8),
        **kwargs
    )
    return model


def DCRNN_med(**kwargs):
    model = DCRNN(
        encoder_embed_dim=384,
        encoder_depth=10,
        norm_layer=partial(nn.LayerNorm, eps=1e-8),
        **kwargs
    )
    return model


def STGCN_mini(**kwargs):
    model = STGCN(
        encoder_embed_dim=128,
        encoder_depth=6,
        norm_layer=partial(nn.LayerNorm, eps=1e-8),
        **kwargs
    )
    return model


def STGCN_small(**kwargs):
    model = STGCN(
        encoder_embed_dim=192,
        encoder_depth=8,
        norm_layer=partial(nn.LayerNorm, eps=1e-8),
        **kwargs
    )
    return model


def STGCN_med(**kwargs):
    model = STGCN(
        encoder_embed_dim=384,
        encoder_depth=10,
        norm_layer=partial(nn.LayerNorm, eps=1e-8),
        **kwargs
    )
    return model
