# reference torch_geometric temporal
# very inefficient, loops over batch

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv, MessagePassing
from torch_geometric.utils import to_dense_adj, dense_to_sparse


class DConv(MessagePassing):
    def __init__(self, in_channels, out_channels, K, bias=True):
        super(DConv, self).__init__(aggr="add", flow="source_to_target")
        assert K > 0
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = torch.nn.Parameter(torch.Tensor(2, K, in_channels, out_channels))

        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        torch.nn.init.zeros_(self.bias)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def compute_graph_stats(self, X_shape, edge_index, edge_weight, device):
        batch_size, num_nodes, D = X_shape
        # Compute adjacency matrix only once, shared across all batches
        adj_mat = to_dense_adj(edge_index, edge_attr=edge_weight)
        adj_mat = adj_mat.view(num_nodes, num_nodes)  # Same adjacency for all batches

        # Compute degree matrices for outgoing and incoming edges
        deg_out = torch.matmul(adj_mat, torch.ones(size=(num_nodes, 1)).to(device)).flatten()
        deg_in = torch.matmul(torch.ones(size=(1, num_nodes)).to(device), adj_mat).flatten()

        # Inverse of degrees with small epsilon for numerical stability
        deg_out_inv = torch.reciprocal(deg_out + 1e-8)
        deg_in_inv = torch.reciprocal(deg_in + 1e-8)

        # Use the same edge_index for all batches, compute norms
        row, col = edge_index
        norm_out = deg_out_inv[row]
        norm_in = deg_in_inv[col]

        reverse_edge_index = adj_mat.transpose(0, 1)
        reverse_edge_index, _ = dense_to_sparse(reverse_edge_index)
        return adj_mat, deg_out, deg_in, norm_out, norm_in, reverse_edge_index

    def forward(
            self,
            X: torch.FloatTensor,
            edge_index: torch.LongTensor,
            edge_weight: torch.FloatTensor,
    ) -> torch.FloatTensor:
        # X: [batch_size, num_nodes, num_features]
        batch_size, num_nodes, D = X.size()
        (
            _, _, _,
            norm_out, norm_in, reverse_edge_index
        ) = self.compute_graph_stats(
            X.shape, edge_index, edge_weight, X.device
        )
        # Diffusion convolution computations with shared graph structure
        Tx_0 = X
        Tx_1 = X
        H = torch.matmul(Tx_0, (self.weight[0])[0]) + torch.matmul(Tx_0, (self.weight[1])[0])

        if self.weight.size(1) > 1:
            Tx_1_o = self.propagate(edge_index, x=X, norm=norm_out, size=None)
            Tx_1_i = self.propagate(reverse_edge_index, x=X, norm=norm_in, size=None)
            H = (
                    H
                    + torch.matmul(Tx_1_o, (self.weight[0])[1])
                    + torch.matmul(Tx_1_i, (self.weight[1])[1])
            )

        for k in range(2, self.weight.size(1)):
            Tx_2_o = self.propagate(edge_index, x=Tx_1_o, norm=norm_out, size=None)
            Tx_2_o = 2.0 * Tx_2_o - Tx_0
            Tx_2_i = self.propagate(reverse_edge_index, x=Tx_1_i, norm=norm_in, size=None)
            Tx_2_i = 2.0 * Tx_2_i - Tx_0
            H = (
                    H
                    + torch.matmul(Tx_2_o, (self.weight[0])[k])
                    + torch.matmul(Tx_2_i, (self.weight[1])[k])
            )
            Tx_0, Tx_1_o, Tx_1_i = Tx_1, Tx_2_o, Tx_2_i

        if self.bias is not None:
            H += self.bias

        return H


class DCRNN_Layer(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, K: int, bias: bool = True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.bias = bias

        self._create_parameters_and_layers()

    def reset_parameters(self):
        self.conv_x_z.reset_parameters()
        self.conv_x_r.reset_parameters()
        self.conv_x_h.reset_parameters()

    def _create_update_gate_parameters_and_layers(self):
        self.conv_x_z = DConv(
            in_channels=self.in_channels + self.out_channels,
            out_channels=self.out_channels,
            K=self.K,
            bias=self.bias,
        )

    def _create_reset_gate_parameters_and_layers(self):
        self.conv_x_r = DConv(
            in_channels=self.in_channels + self.out_channels,
            out_channels=self.out_channels,
            K=self.K,
            bias=self.bias,
        )

    def _create_candidate_state_parameters_and_layers(self):
        self.conv_x_h = DConv(
            in_channels=self.in_channels + self.out_channels,
            out_channels=self.out_channels,
            K=self.K,
            bias=self.bias,
        )

    def _create_parameters_and_layers(self):
        self._create_update_gate_parameters_and_layers()
        self._create_reset_gate_parameters_and_layers()
        self._create_candidate_state_parameters_and_layers()

    def _set_hidden_state(self, X, H):
        if H is None:
            H = torch.zeros(X.shape[0], X.shape[1], self.out_channels).to(X.device)
        return H

    def _calculate_update_gate(self, X, edge_index, edge_weight, H):
        Z = torch.cat([X, H], dim=-1)
        Z = self.conv_x_z(Z, edge_index, edge_weight)
        Z = torch.sigmoid(Z)
        return Z

    def _calculate_reset_gate(self, X, edge_index, edge_weight, H):
        R = torch.cat([X, H], dim=-1)
        R = self.conv_x_r(R, edge_index, edge_weight)
        R = torch.sigmoid(R)
        return R

    def _calculate_candidate_state(self, X, edge_index, edge_weight, H, R):
        H_tilde = torch.cat([X, H * R], dim=-1)
        H_tilde = self.conv_x_h(H_tilde, edge_index, edge_weight)
        H_tilde = torch.tanh(H_tilde)
        return H_tilde

    def _calculate_hidden_state(self, Z, H, H_tilde):
        H = Z * H + (1 - Z) * H_tilde
        return H

    def forward(
            self,
            X: torch.FloatTensor,
            edge_index: torch.LongTensor,
            edge_weight: torch.FloatTensor = None,
            H: torch.FloatTensor = None,
    ) -> torch.FloatTensor:
        # X: [batch_size, num_nodes, num_features]
        H = self._set_hidden_state(X, H)
        Z = self._calculate_update_gate(X, edge_index, edge_weight, H)
        R = self._calculate_reset_gate(X, edge_index, edge_weight, H)
        H_tilde = self._calculate_candidate_state(X, edge_index, edge_weight, H, R)
        H = self._calculate_hidden_state(Z, H, H_tilde)
        return H


class TemporalConv(nn.Module):
    r"""Temporal convolution block applied to nodes in the STGCN Layer
    For details see: `"Spatio-Temporal Graph Convolutional Networks:
    A Deep Learning Framework for Traffic Forecasting."
    <https://arxiv.org/abs/1709.04875>`_ Based off the temporal convolution
     introduced in "Convolutional Sequence to Sequence Learning"  <https://arxiv.org/abs/1709.04875>`_

    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        kernel_size (int): Convolutional kernel size.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super(TemporalConv, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv_2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv_3 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))

    def forward(self, X: torch.FloatTensor) -> torch.FloatTensor:
        """Forward pass through temporal convolution block.

        Arg types:
            * **X** (torch.FloatTensor) -  Input data of shape
                (batch_size, input_time_steps, num_nodes, in_channels).

        Return types:
            * **H** (torch.FloatTensor) - Output data of shape
                (batch_size, in_channels, num_nodes, input_time_steps).
        """
        X = X.permute(0, 3, 2, 1)
        P = self.conv_1(X)
        Q = torch.sigmoid(self.conv_2(X))
        PQ = P * Q
        H = F.relu(PQ + self.conv_3(X))
        H = H.permute(0, 3, 2, 1)
        return H


class STConv(nn.Module):
    r"""Spatio-temporal convolution block using ChebConv Graph Convolutions.
    For details see: `"Spatio-Temporal Graph Convolutional Networks:
    A Deep Learning Framework for Traffic Forecasting"
    <https://arxiv.org/abs/1709.04875>`_

    NB. The ST-Conv block contains two temporal convolutions (TemporalConv)
    with kernel size k. Hence for an input sequence of length m,
    the output sequence will be length m-2(k-1).

    Args:
        in_channels (int): Number of input features.
        hidden_channels (int): Number of hidden units output by graph convolution block
        out_channels (int): Number of output features.
        kernel_size (int): Size of the kernel considered.
        K (int): Chebyshev filter size :math:`K`.
        normalization (str, optional): The normalization scheme for the graph
            Laplacian (default: :obj:`"sym"`):

            1. :obj:`None`: No normalization
            :math:`\mathbf{L} = \mathbf{D} - \mathbf{A}`

            2. :obj:`"sym"`: Symmetric normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{A}
            \mathbf{D}^{-1/2}`

            3. :obj:`"rw"`: Random-walk normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1} \mathbf{A}`

            You need to pass :obj:`lambda_max` to the :meth:`forward` method of
            this operator in case the normalization is non-symmetric.
            :obj:`\lambda_max` should be a :class:`torch.Tensor` of size
            :obj:`[num_graphs]` in a mini-batch scenario and a
            scalar/zero-dimensional tensor when operating on single graphs.
            You can pre-compute :obj:`lambda_max` via the
            :class:`torch_geometric.transforms.LaplacianLambdaMax` transform.
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)

    """

    def __init__(
            self,
            num_nodes: int,
            in_channels: int,
            hidden_channels: int,
            out_channels: int,
            kernel_size: int,
            K: int,
            normalization: str = "sym",
            bias: bool = True,
    ):
        super(STConv, self).__init__()
        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.K = K
        self.normalization = normalization
        self.bias = bias

        self._temporal_conv1 = TemporalConv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
        )

        self._graph_conv = ChebConv(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            K=K,
            normalization=normalization,
            bias=bias,
        )

        self._temporal_conv2 = TemporalConv(
            in_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
        )

        self._batch_norm = nn.BatchNorm2d(num_nodes)

    def forward(
            self,
            X: torch.FloatTensor,
            edge_index: torch.LongTensor,
            edge_weight: torch.FloatTensor = None,
    ) -> torch.FloatTensor:

        r"""Forward pass. If edge weights are not present the forward pass
        defaults to an unweighted graph.

        Arg types:
            * **X** (PyTorch FloatTensor) - Sequence of node features of shape (Batch size X Input time steps X Num nodes X In channels).
            * **edge_index** (PyTorch LongTensor) - Graph edge indices.
            * **edge_weight** (PyTorch LongTensor, optional)- Edge weight vector.

        Return types:
            * **T** (PyTorch FloatTensor) - Sequence of node features.
        """
        T_0 = self._temporal_conv1(X)
        T = torch.zeros_like(T_0).to(T_0.device)
        for b in range(T_0.size(0)):
            for t in range(T_0.size(1)):
                T[b][t] = self._graph_conv(T_0[b][t], edge_index, edge_weight)

        T = F.relu(T)
        T = self._temporal_conv2(T)
        T = T.permute(0, 2, 1, 3)
        T = self._batch_norm(T)
        T = T.permute(0, 2, 1, 3)
        return T
