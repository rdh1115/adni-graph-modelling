import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from typing import Union, Optional, Callable


# from PyTorch Geometric Temporal code
# https://pytorch-geometric-temporal.readthedocs.io/en/latest/_modules/torch_geometric_temporal/nn/attention/gman.html#GMAN
# conv2d_ and FC are for converting traffic data with d node feature to the specified embedding dim
class Conv2D(nn.Module):
    r"""An implementation of the 2D-convolution block.
    For details see this paper: `"GMAN: A Graph Multi-Attention Network for Traffic Prediction."
    <https://arxiv.org/pdf/1911.08415.pdf>`

    Args:
        input_dims (int): Dimension of input.
        output_dims (int): Dimension of output.
        kernel_size (tuple or list): Size of the convolution kernel.
        stride (tuple or list, optional): Convolution strides, default (1,1).
        use_bias (bool, optional): Whether to use bias, default is True.
        activation (Callable, optional): Activation function, default is torch.nn.functional.gelu.
        bn_decay (float, optional): Batch normalization momentum, default is None.
    """

    def __init__(
            self,
            input_dims: int,
            output_dims: int,
            kernel_size: Union[tuple, list],
            stride: Union[tuple, list] = (1, 1),
            use_bias: bool = False,
            activation: Optional[Callable[[torch.FloatTensor], torch.FloatTensor]] = F.gelu,
    ):
        super(Conv2D, self).__init__()
        self.activation = activation
        self.conv = nn.Conv2d(
            input_dims,
            output_dims,
            kernel_size,
            stride=stride,
            padding=0,
            bias=use_bias,
        )
        self.batch_norm = nn.BatchNorm2d(output_dims)

    def forward(self, x):
        """
                Making a forward pass of the 2D-convolution block.

                Arg types:
                    * **x** (torch float tensor) - Input with shape (batch_size, input_dims, num_nodes, num_his).

                Return types:
                    * **x** (torch float Tensor) - Output with shape (batch_size, output_dims, num_nodes, num_his).
        """
        x = self.conv(x)
        x = self.batch_norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class FC(nn.Module):
    def __init__(self, input_dims, units, activations, use_bias):
        super(FC, self).__init__()
        if isinstance(units, int):
            units = [units]
            input_dims = [input_dims]
            activations = [activations]
        elif isinstance(units, tuple):
            units = list(units)
            input_dims = list(input_dims)
            activations = list(activations)
        assert type(units) == list
        self.convs = nn.ModuleList(
            [
                Conv2D(
                    input_dims=input_dim,
                    output_dims=num_unit,
                    kernel_size=[1, 1],
                    stride=[1, 1],
                    use_bias=use_bias,
                    activation=activation,
                )
                for input_dim, num_unit, activation in zip(
                input_dims, units, activations
            )
            ]
        )

    def forward(self, x):
        """
                Making a forward pass of the fully-connected layer.

                Arg types:
                    * **x** (torch float) - Input, with shape (batch_size, num_his, num_nodes, input_dim[0]).

                Return types:
                    * **x** (torch float) - Output, with shape (batch_size, num_his, num_nodes, units[-1]).
        """
        x = x.contiguous().permute(0, 3, 2, 1)
        for conv in self.convs:
            x = conv(x)
        x = x.contiguous().permute(0, 3, 2, 1)
        return x


class GraphNodeFeature(nn.Module):
    """
    Compute node features for each node in the graph.
    """

    def __init__(
            self,
            node_feature_dim,
            num_heads,
            num_nodes,
            num_in_degree,
            num_out_degree,
            hidden_dim,
            start_conv=False,
            centrality_encoding=True,
            graph_token=False,
            act_fn='gelu',
            old_config=False,
    ):
        super(GraphNodeFeature, self).__init__()
        self.graph_token = graph_token
        self.num_heads = num_heads
        self.num_nodes = num_nodes
        self.start_conv = start_conv
        self.centrality_encoding = centrality_encoding
        act_function = utils.get_activation_fn(act_fn)
        self.activation = act_function() if act_fn == 'swish' else act_function
        if self.start_conv:  # convert to hidden_dim
            if old_config:
                self.fc = FC(
                    input_dims=[node_feature_dim, hidden_dim],
                    units=[hidden_dim, hidden_dim],
                    activations=[self.activation, None],
                    use_bias=True,
                    # bn_decay=bn_decay
                )
            else:
                self.fc = FC(
                    input_dims=[node_feature_dim, hidden_dim],
                    units=[hidden_dim, hidden_dim],
                    activations=[self.activation, None],
                    use_bias=False,
                    # bn_decay=bn_decay
                )
            if centrality_encoding:
                self.in_degree_encoder = nn.Embedding(
                    num_in_degree + 1, hidden_dim, padding_idx=0
                )
                self.out_degree_encoder = nn.Embedding(
                    num_out_degree + 1, hidden_dim, padding_idx=0
                )

    def forward(self, x, in_degree, out_degree):
        # N, T, V, D = x.shape
        # node feature + graph token
        # only for integer node features
        # node_feature = self.atom_encoder(x).sum(dim=-2)  # [n_graph, n_node, n_hidden]

        # if self.flag and perturb is not None:
        #     node_feature += perturb

        if self.start_conv:
            x = self.fc(x)  # for node values that are not one hot
        if self.centrality_encoding:
            centrality_encodings = (
                    self.in_degree_encoder(in_degree)
                    + self.out_degree_encoder(out_degree)
            )
            x = x + centrality_encodings.unsqueeze(1)
        return x


class GraphAttnBias(nn.Module):
    """
    Compute attention bias for each head.
    """

    def __init__(
            self,
            num_heads,
            num_edges,
            num_spatial,
            num_edge_dis,
            edge_type,
            multi_hop_max_dist,
            graph_token,
            edge_features
    ):
        super(GraphAttnBias, self).__init__()
        self.num_heads = num_heads
        self.multi_hop_max_dist = multi_hop_max_dist
        self.graph_token = graph_token
        self.edge_features = edge_features

        if self.edge_features:
            # edge_encoder and edge_dis_encoder only used when there are edge features
            # edge_encoder is 512*max_SPD, which is then avged
            self.edge_encoder = nn.Embedding(
                num_edges + 1, num_heads, padding_idx=0
            )
            self.edge_type = edge_type
            if self.edge_type == "multi_hop":
                self.edge_dis_encoder = nn.Embedding(
                    num_edge_dis * num_heads * num_heads, 1
                )
        # embedding based on distance (eq. 6 in the Graphormer paper: b_{phi(vi,vj)},
        # phi in this case is the shortest path)
        self.spatial_pos_encoder = nn.Embedding(num_spatial + 1, num_heads, padding_idx=0)

    def forward(self, batched_data):
        # compute the attention bias for the graph
        attn_bias, spatial_pos, x = (
            batched_data["attn_bias"],
            batched_data["spatial_pos"],
            batched_data["x"],
        )
        # in_degree, out_degree = batched_data.in_degree, batched_data.in_degree
        edge_input, attn_edge_type = (
            batched_data["edge_input"],
            batched_data["attn_edge_type"],
        )
        """
        edge_input is for edge encoding,
        has shape [N, V, V, longest SPD, n_feature],
        see /data/utils.preprocess_item
        """

        N, T, V, D = x.shape
        graph_attn_bias = attn_bias.clone()
        graph_attn_bias = graph_attn_bias.unsqueeze(1).repeat(
            1, self.num_heads, 1, 1
        )  # [N, n_head, V+1, V+1]
        if self.graph_token:
            g_idx = 1
        else:
            g_idx = 0
        # spatial pos
        # [N, V, V, n_head] -> [N, n_head, V, V]
        spatial_pos_bias = self.spatial_pos_encoder(spatial_pos).contiguous().permute(0, 3, 1, 2)
        graph_attn_bias[:, :, g_idx:, g_idx:] = graph_attn_bias[:, :, g_idx:, g_idx:] + spatial_pos_bias

        if edge_input and attn_edge_type:
            assert self.edge_features
            # edge feature
            if self.edge_type == "multi_hop":
                spatial_pos_ = spatial_pos.clone()
                spatial_pos_[spatial_pos_ == 0] = 1  # set pad to 1
                # set 1 to 1, x > 1 to x - 1
                spatial_pos_ = torch.where(spatial_pos_ > 1, spatial_pos_ - 1, spatial_pos_)
                if self.multi_hop_max_dist > 0:
                    spatial_pos_ = spatial_pos_.clamp(0, self.multi_hop_max_dist)
                    edge_input = edge_input[:, :, :, : self.multi_hop_max_dist, :]
                # [N, V, V, max_dist, n_head]
                edge_input = self.edge_encoder(edge_input).mean(-2)
                max_dist = edge_input.size(-2)
                # [max_dist, N*V*V, n_head]
                edge_input_flat = edge_input.permute(3, 0, 1, 2, 4).reshape(
                    max_dist, -1, self.num_heads
                )
                # [num_edge_dis, num_heads, num_heads]
                edge_dis_encoder_flat = self.edge_dis_encoder.weight.reshape(
                    -1, self.num_heads, self.num_heads
                )[:max_dist, :, :]
                # [max_dist, N*V*V, num_heads]
                edge_input_flat = torch.bmm(
                    edge_input_flat,
                    edge_dis_encoder_flat,
                )
                edge_input = edge_input_flat.reshape(
                    max_dist, N, V, V, self.num_heads
                ).permute(1, 2, 3, 0, 4)
                edge_input = (
                        edge_input.sum(-2) / (spatial_pos_.float().unsqueeze(-1))
                ).permute(0, 3, 1, 2)
            else:
                # [N, V, V, n_head] -> [N, n_head, V, V]
                edge_input = self.edge_encoder(attn_edge_type).mean(-2).permute(0, 3, 1, 2)

            graph_attn_bias[:, :, g_idx:, g_idx:] = graph_attn_bias[:, :, g_idx:, g_idx:] + edge_input

        del edge_input, attn_edge_type, batched_data['edge_input'], batched_data['attn_edge_type'], spatial_pos, \
            batched_data['spatial_pos']

        # why do this?
        graph_attn_bias = graph_attn_bias + attn_bias.unsqueeze(1)  # reset

        return graph_attn_bias
