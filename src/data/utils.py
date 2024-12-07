from typing import Iterable

import torch
import numpy as np
import pyximport
from sklearn.model_selection import train_test_split
import json

pyximport.install(setup_args={"include_dirs": np.get_include()})
from src.data.algos import gen_edge_input, floyd_warshall
from torch_geometric.data import Data


class Scaler:
    def transform(self, data):
        raise NotImplementedError

    def inverse_transform(self, data, args=None):
        raise NotImplementedError

    def to_device(self, device):
        raise NotImplementedError


class StandardScaler(Scaler):
    """
    z-score norm the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data, args=None):
        data_shape = data.shape
        D = data_shape[-1]

        if args:
            filter_list = args.filter_list
            pred_dim_idx = get_target_brain_arr_indices(D, filter_list, args.include_pet_volume)
            return (data * self.std[pred_dim_idx]) + self.mean[pred_dim_idx]
        return (data * self.std) + self.mean

    def __str__(self):
        return f"StandardScaler(mean={self.mean}, std={self.std})"

    def to_device(self, device):
        for attr in ['mean', 'std']:
            attr_val = getattr(self, attr)
            if isinstance(attr_val, np.ndarray):
                setattr(self, attr, torch.tensor(attr_val, dtype=torch.float, device=device))
            elif isinstance(attr_val, torch.Tensor):
                setattr(self, attr, attr_val.to(device))
            elif isinstance(attr_val, float) or isinstance(attr_val, int):
                setattr(self, attr, torch.tensor(attr_val, device=device))
            else:
                raise NotImplementedError('scaler attributes should be torch.Tensor or np.ndarray or float/int')
        return self


class MinMaxScaler(Scaler):
    """
    Min-Max normalization for the input.
    """

    def __init__(self, data_min, data_max, feature_range=(0, 1)):
        self.data_min = data_min
        self.data_max = data_max
        self.feature_range = feature_range
        self.min_val, self.max_val = feature_range

    def transform(self, data):
        """
        Scales the input data using Min-Max normalization.
        """
        return (data - self.data_min) / (self.data_max - self.data_min) * (self.max_val - self.min_val) + self.min_val

    def inverse_transform(self, data, args=None):
        """
        Reverts the scaled data back to the original range.
        """
        data_shape = data.shape
        D = data_shape[-1]

        if args:
            filter_list = args.filter_list
            pred_dim_idx = get_target_brain_arr_indices(D, filter_list, args.include_pet_volume)
            return (data - self.min_val) / (self.max_val - self.min_val) * (
                    self.data_max[pred_dim_idx] - self.data_min[pred_dim_idx]) + self.data_min[pred_dim_idx]
        return (data - self.min_val) / (self.max_val - self.min_val) * (self.data_max - self.data_min) + self.data_min

    def __str__(self):
        return f"MinMaxScaler(data_min={self.data_min}, data_max={self.data_max}, feature_range={self.feature_range})"

    def to_device(self, device):
        for attr in ['data_min', 'data_max', 'min_val', 'max_val']:
            attr_val = getattr(self, attr)
            if isinstance(attr_val, np.ndarray):
                setattr(self, attr, torch.tensor(attr_val, dtype=torch.float, device=device))
            elif isinstance(attr_val, torch.Tensor):
                setattr(self, attr, attr_val.to(device))
            elif isinstance(attr_val, float) or isinstance(attr_val, int):
                setattr(self, attr, torch.tensor(attr_val, device=device))
            else:
                raise NotImplementedError('scaler attributes should be torch.Tensor or np.ndarray or float/int')
        return self


def get_target_brain_arr_indices(num_feature, filter_list, include_pet_volume):
    """
    the input to the model could be extra information used for prediction
    e.g. the target for PET is always the PET SUVR values, but
    the input to the model could also include the MRI volume closest to the PET scan
    the target for MRI is the volume and thickness average
    if MRI SA and TA are included in the input, then there is no PET volume
    :param x_shape:
    :param filter_list:
    :param include_pet_volume:
    :return:
    """

    if filter_list[0]:
        return [i for i in range(num_feature)]
    else:
        if include_pet_volume:
            if num_feature == 4:  # input includes [SUVR, volume, SUVR, volume]
                return [0, 2]
            elif num_feature == 2:  # input includes [SUVR, volume]
                return [0]
            else:
                raise ValueError('brain node dimension unrecognized input shape')
        else:
            return [i for i in range(num_feature)]


def normalize(train_x, val_x, test_x, train_y, val_y, test_y):
    mean, std = train_x.mean(axis=(0, 1)), train_x.std(axis=(0, 1))

    scaler = StandardScaler(np.array(mean), np.array(std))
    train_x = scaler.transform(train_x)
    val_x = scaler.transform(val_x)
    test_x = scaler.transform(test_x)
    return train_x, val_x, test_x, train_y, val_y, test_y, scaler


def convert_to_single_emb(x, offset=512):
    feature_num = x.size(1) if len(x.size()) > 1 else 1
    feature_offset = 1 + torch.arange(0, feature_num * offset, offset, dtype=torch.long)
    x = x + feature_offset
    return x


def pad_1d_unsqueeze(x, padlen):
    x = x + 1  # pad id = 0
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen], dtype=x.dtype)
        new_x[:xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_2d_unsqueeze(x, padlen):
    x = x + 1  # pad id = 0
    xlen, xdim = x.size()
    if xlen < padlen:
        new_x = x.new_zeros([padlen, xdim], dtype=x.dtype)
        new_x[:xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_xs(x: torch.Tensor, padlen):
    x = x + 1
    V, D, T = x.shape
    if V < padlen:
        new_x = x.new_zeros([V, D, padlen], dtype=x.dtype)
        new_x[:V, :, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_3d_unsqueeze(x, padlen1, padlen2, padlen3):
    x = x + 1
    xlen1, xlen2, xlen3, xlen4 = x.size()
    if xlen1 < padlen1 or xlen2 < padlen2 or xlen3 < padlen3:
        new_x = x.new_zeros([padlen1, padlen2, padlen3, xlen4], dtype=x.dtype)
        new_x[:xlen1, :xlen2, :xlen3, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_spatial_pos_unsqueeze(x, padlen):
    x = x + 1
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype)
        new_x[:xlen, :xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_attn_bias_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype).fill_(float("-inf"))
        new_x[:xlen, :xlen] = x
        new_x[xlen:, :xlen] = 0
        x = new_x
    return x.unsqueeze(0)


def pad_edge_type_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen, x.size(-1)], dtype=x.dtype)
        new_x[:xlen, :xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def preprocess_item(item: Data, graph_token: bool):
    # see pyg data doc
    # x is node feature matrix with shape [n_nodes, node_feature_dim]
    # edge_attr same
    edge_attr, edge_index, x = item.edge_attr, item.edge_index, item.x
    assert len(x.shape) == 3
    t, N_nodes, feature_dim = x.shape

    # node adj matrix [N, N] bool
    adj = torch.zeros([N_nodes, N_nodes], dtype=torch.bool)
    adj[edge_index[0, :], edge_index[1, :]] = True
    weighted_adj = adj.detach().clone()

    # edge features
    if edge_attr is not None:
        # if the edge features are 1-hot encoding
        # (in the case of different edge types e.g. different molecular bonds)
        if torch.all(sum([edge_attr == i for i in [1, 0]]).bool()):
            edge_attr = edge_attr[:, None]
            attn_edge_type = torch.zeros([N_nodes, N_nodes, edge_attr.size(-1)], dtype=torch.long)
            # stop using convert_to_single, which converts to ints
            attn_edge_type[edge_index[0, :], edge_index[1, :]] = convert_to_single_emb(edge_attr.long()) + 1
        else:
            # in this case, edge_attr is edge distance
            assert torch.all(edge_attr < 15), 'Distances should be normalized'
            weighted_adj = weighted_adj.float()
            weighted_adj[edge_index[0, :], edge_index[1, :]] = edge_attr
            attn_edge_type = []
    else:
        attn_edge_type = []

    # path i,j entry stores the intermediate node used to reach i,j
    shortest_path_result, path = floyd_warshall(weighted_adj.numpy())
    max_dist = np.amax(shortest_path_result)

    if attn_edge_type:
        # collect edge attributes along the shortest paths
        edge_input = gen_edge_input(max_dist, path, attn_edge_type.numpy())
        item.edge_input = torch.from_numpy(edge_input).long()
    else:
        edge_input = []
        item.edge_input = edge_input

    # spatial pos is [n_node, n_node], the shortest path between nodes
    # used in spatial encoding attention bias where b_(vi, vj) is learnable scalar indexed by shortest path
    spatial_pos = torch.from_numpy(shortest_path_result).long()
    if graph_token:
        attn_bias = torch.zeros([N_nodes + 1, N_nodes + 1], dtype=torch.float)
    else:
        attn_bias = torch.zeros([N_nodes, N_nodes], dtype=torch.float)

    # combine
    item.x = x
    item.adj = adj
    item.attn_bias = attn_bias
    item.attn_edge_type = attn_edge_type
    item.spatial_pos = spatial_pos
    item.in_degree = adj.long().sum(dim=1).view(-1)
    item.out_degree = adj.long().sum(dim=0).view(-1)

    return item


def collator(items, max_node=512, multi_hop_max_dist=20, spatial_pos_max=20, graph_token=True, scaler=None):
    """
    get the attributes from a list of Data objects and stack them such that
    the output is a dictionary of tensors with the same shape,
    keys are the attribute names, values are the stacked tensors
    :param items:
    :param max_node:
    :param multi_hop_max_dist:
    :param spatial_pos_max:
    :param graph_token:
    :param scaler:
    :return:
    """
    items = [item for item in items if item is not None and item.x.size(0) <= max_node]

    items = [
        (
            item.idx,
            item.attn_bias,
            item.attn_edge_type,
            item.spatial_pos,
            item.in_degree,
            item.out_degree,
            item.x,
            item.edge_index,
            item.edge_attr,
            item.edge_input[:, :, :multi_hop_max_dist, :] if item.edge_input else item.edge_input,
            item.y,
            item.additional_features
        )
        for item in items
    ]
    (
        idxs,
        attn_biases,
        attn_edge_types,
        spatial_poses,
        in_degrees,
        out_degrees,
        xs,
        edge_indices,
        edge_attrs,
        edge_inputs,
        ys,
        add_features
    ) = zip(*items)

    if all(len(i) == 0 for i in edge_inputs):
        edge_inputs = None
    if all(len(i) == 0 for i in attn_edge_types):
        attn_edge_types = None

    for idx, _ in enumerate(attn_biases):
        if graph_token:
            attn_biases[idx][1:, 1:][spatial_poses[idx] >= spatial_pos_max] = float("-inf")
        else:
            attn_biases[idx][:, :][spatial_poses[idx] >= spatial_pos_max] = float('-inf')
    # check if the graphs are static
    it = iter(xs)
    the_len = len(next(it))
    # if graphs are not static
    if not all(len(l) == the_len for l in it):
        # pad to max_node_num and max_dist
        max_node_num = max(i.size(0) for i in xs)
        max_dist = max(i.size(-2) for i in edge_inputs)
        y = torch.stack(ys)
        # TODO: fix x padding for dynamic graphs
        x = torch.cat([pad_xs(i, max_node_num) for i in xs])

        edge_input = torch.cat(
            [pad_3d_unsqueeze(i, max_node_num, max_node_num, max_dist) for i in edge_inputs]
        ) if edge_inputs else None
        attn_edge_type = torch.cat(
            [pad_edge_type_unsqueeze(i, max_node_num) for i in attn_edge_types]
        ) if attn_edge_types else None

        attn_bias = torch.cat(
            [pad_attn_bias_unsqueeze(i, max_node_num + 1) for i in attn_biases]
        )

        spatial_pos = torch.cat(
            [pad_spatial_pos_unsqueeze(i, max_node_num) for i in spatial_poses]
        )
        in_degree = torch.cat([pad_1d_unsqueeze(i, max_node_num) for i in in_degrees])
    else:
        # just stack
        y = torch.stack(ys)
        x = torch.stack(xs)
        edge_indices = torch.stack(edge_indices) if edge_indices else None
        edge_attrs = torch.stack(edge_attrs) if edge_attrs else None
        edge_input = torch.stack(edge_inputs) if edge_inputs else None
        attn_edge_type = torch.stack(attn_edge_types) if attn_edge_types else None
        attn_bias = torch.stack(attn_biases)
        spatial_pos = torch.stack(spatial_poses) + 1
        in_degree = torch.stack(in_degrees) + 1

    return dict(
        idx=torch.LongTensor(idxs),
        attn_bias=attn_bias,
        attn_edge_type=attn_edge_type,
        spatial_pos=spatial_pos,
        in_degree=in_degree,
        out_degree=in_degree,  # for undirected graph
        edge_index=edge_indices,
        edge_attr=edge_attrs,
        x=x,
        edge_input=edge_input,
        y=y,
        add_features=add_features,
        scaler=scaler
    )


def generate_regression_task_df(df, n_hist, n_pred, add_time_in_day=True, add_day_in_week=False):
    '''

    :param df: dataframe with all sensor data with shape[T, V]
    :param n_hist: number of observed time points
    :param n_pred: time points to be predicted
    :param add_time_in_day: whether to add time in day information to the traffic volume sensor data
    :param add_day_in_week: whether to add time in week information to the traffic volume sensor data
    :return: features and targets of shape [num datapoints, time slice, num_nodes, d]
    '''
    features, targets = [], []
    T, V = df.shape

    data = np.expand_dims(df.values, axis=-1)
    data_list = [data]
    if not df.index.values.dtype == np.dtype('<M8[ns]'):
        add_time_in_day = False
        add_day_in_week = False

    if add_time_in_day:
        time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
        time_in_day = np.tile(time_ind, [1, V, 1]).transpose((2, 1, 0))
        data_list.append(time_in_day)
    if add_day_in_week:
        day_in_week = np.zeros(shape=(T, V, 7))
        day_in_week[np.arange(T), :, df.index.dayofweek] = 1
        data_list.append(day_in_week)
    data = np.concatenate(data_list, axis=-1)

    indices = [
        (i, i + (n_hist + n_pred))
        for i in range(T - (n_hist + n_pred) + 1)
    ]

    for i, j in indices:
        features.append(data[i: i + n_hist, ...])
        targets.append(data[i + n_hist: j, ...])
    features = np.stack(features, axis=0)
    targets = np.stack(targets, axis=0)
    return features, targets


def generate_regression_task(X, n_hist, n_pred):
    features, targets = [], []
    T, V, D = X.shape
    indices = [
        (i, i + (n_hist + n_pred))
        for i in range(T - (n_hist + n_pred) + 1)
    ]

    for i, j in indices:
        features.append(X[i: i + n_hist, ...])
        targets.append(X[i + n_hist: j, ...])
    features = np.stack(features, axis=0)
    targets = np.stack(targets, axis=0)
    return features, targets


def generate_regression_brain(subject_d, n_hist, n_pred):
    X = subject_d['arr']
    T, V, D = X.shape
    indices = [
        (i, i + (n_hist + n_pred))
        for i in range(T - (n_hist + n_pred) + 1)
    ]
    add_feature_keys = [k for k in subject_d.keys() if k != 'arr' and isinstance(subject_d[k], Iterable)]
    assert all(len(subject_d[k]) == len(subject_d['arr']) for k in add_feature_keys)

    add_features = {k: list() for k in add_feature_keys}
    add_targets = {k + '_target': list() for k in add_feature_keys}
    for i, j in indices:
        for k in add_feature_keys:
            v = subject_d[k]
            add_features[k].append(v[i: i + n_hist])
            add_targets[k + '_target'].append(v[i + n_hist: j])
    for k, v in add_features.items():
        add_features[k] = np.stack(v, axis=0)
    for k, v in add_targets.items():
        add_targets[k] = np.stack(v, axis=0)
    return add_features, add_targets


def generate_split(X, y, split_ratio, stratify_feature=None):
    num_data = len(X)

    test_split, valid_split = split_ratio
    test_split, valid_split = test_split / 100, valid_split / 100
    print(
        f"creating train/valid/test datasets, ratio: "
        f"{1.0 - test_split - valid_split:.1f}/{valid_split:.1f}/{test_split:.1f}"
    )
    valid_split = valid_split / (1.0 - test_split)

    shuffle = True
    if stratify_feature is None:
        def stratify_feature(x):
            return max([DX_DICT[dx] for dx in x['DX']])  # only consider the worst diagnosis
    ys = np.array([stratify_feature(d) for rid, d in X.items()])
    train_valid_idx, test_idx = train_test_split(
        np.arange(num_data),
        test_size=test_split,
        shuffle=shuffle,
        stratify=ys
    )
    train_idx, valid_idx = train_test_split(
        train_valid_idx,
        test_size=valid_split,
        shuffle=shuffle,
        stratify=ys[train_valid_idx] if ys is not None else None
    )
    train_x, val_x, test_x = dict(), dict(), dict()
    for i, (rid_x, d_x) in enumerate(X.items()):
        if i in train_idx:
            train_x[rid_x] = d_x
        elif i in valid_idx:
            val_x[rid_x] = d_x
        elif i in test_idx:
            test_x[rid_x] = d_x
        else:
            raise RuntimeError('splitting went wrong')
    return (train_x, val_x, test_x, None, None, None, None), train_idx, valid_idx, test_idx


def save_np_dict(d, idx, fp):
    tmp = {
        int(rid):
            {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in d.items()
            }
        for rid, d in d.items()
    }
    tmp['idx'] = idx.tolist()
    with open(fp, 'w', encoding='utf-8') as f:
        json.dump(tmp, f)
    return


def load_np_dict(fp):
    with open(fp, 'r', encoding='utf-8') as f:
        content = f.read().strip()  # Strip any leading/trailing whitespace
        subject_dict = json.loads(content)
    tmp = subject_dict.pop('idx')
    subject_dict = {
        int(rid):
            {
                k: np.array(v) if isinstance(v, list) else v
                for k, v in d.items()
            }
        for rid, d in subject_dict.items()
    }
    subject_dict['idx'] = tmp
    return subject_dict


DX_DICT = {
    'CN': 0,
    'MCI': 1,
    'Dementia': 2
}
