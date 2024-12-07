import torch
import numpy as np

from typing import List, Union, Iterable
from torch_geometric.data import Data

Edge_Index = Union[np.ndarray, None]
Edge_Weight = Union[np.ndarray, None]
Node_Features = List[Union[np.ndarray, None]]
Targets = List[Union[np.ndarray, None]]
Additional_Features = List[np.ndarray]


class StaticGraphTemporalSignal:
    r"""A data iterator object to contain a static graph with a dynamically
    changing constant time difference temporal feature set (multiple signals).
    The node labels (target) are also temporal. The iterator returns a single
    constant time difference temporal snapshot for a time period (e.g. day or week).
    This single temporal snapshot is a Pytorch Geometric Data object. Between two
    temporal snapshots the features and optionally passed attributes might change.
    However, the underlying graph is the same.

    Args:
        edge_index (Numpy array): Index tensor of edges.
        edge_weight (Numpy array): Edge weight tensor.
        features (List of Numpy arrays): List of node feature tensors.
        targets (List of Numpy arrays): List of node label (target) tensors.
        **kwargs (optional List of Numpy arrays): List of additional attributes.
    """

    def __init__(
            self,
            edge_index: Edge_Index,
            edge_weight: Edge_Weight,
            features: Node_Features,
            targets: Targets,
            **kwargs: Additional_Features
    ):
        self.edge_index = edge_index if not isinstance(edge_index, np.ndarray) else edge_index
        self.edge_weight = edge_weight if not isinstance(edge_weight, np.ndarray) else edge_weight
        self.features = np.array(features) if not isinstance(features, np.ndarray) else features
        self.targets = np.array(targets) if not isinstance(targets, np.ndarray) else targets
        self.additional_feature_keys = []
        for key, value in kwargs.items():
            if isinstance(value, Iterable):
                value = np.array(value) if not isinstance(value, np.ndarray) else value
                setattr(self, key, value)
                self.additional_feature_keys.append(key)
        self._check_temporal_consistency()
        self._set_snapshot_count()

    def _check_temporal_consistency(self):
        assert len(self.features) == len(
            self.targets
        ), "Temporal dimension inconsistency between features and targets."
        for key in self.additional_feature_keys:
            assert len(self.targets) == len(
                getattr(self, key)
            ), (f"Temporal dimension inconsistency for {key}. "
                f"Expected {len(self.targets)} but got {len(getattr(self, key))}.")

    def _set_snapshot_count(self):
        self.snapshot_count = len(self.features)

    def _get_edge_index(self):
        if self.edge_index is None:
            return self.edge_index
        else:
            return torch.LongTensor(self.edge_index)

    def _get_edge_weight(self):
        if self.edge_weight is None:
            return self.edge_weight
        else:
            return torch.FloatTensor(self.edge_weight)

    def _get_features(self, time_index: int):
        if self.features[time_index] is None:
            return self.features[time_index]
        else:
            return torch.FloatTensor(self.features[time_index])

    def _get_target(self, time_index: int):
        if self.targets[time_index] is None:
            return self.targets[time_index]
        else:
            if self.targets[time_index].dtype.type == np.int_:
                return torch.LongTensor(self.targets[time_index])
            return torch.FloatTensor(self.targets[time_index])

    def _get_additional_feature(self, time_index: int, feature_key: str):
        feature = getattr(self, feature_key)[time_index]
        if isinstance(feature, str) or feature.dtype.type == np.str_:
            return feature
        feature = np.array(feature) if not isinstance(feature, Iterable) else feature
        return torch.from_numpy(feature)

    def _get_additional_features(self, time_index: int):
        additional_features = {
            key: self._get_additional_feature(time_index, key)
            for key in self.additional_feature_keys
        }
        return additional_features

    def __getitem__(self, time_index: Union[int, slice, np.ndarray, torch.Tensor]):
        if isinstance(time_index, slice) or isinstance(time_index, np.ndarray) or isinstance(time_index, torch.Tensor):
            snapshot = StaticGraphTemporalSignal(
                self.edge_index,
                self.edge_weight,
                self.features[time_index],
                self.targets[time_index],
                **{key: getattr(self, key)[time_index]
                   for key in self.additional_feature_keys}
            )
        else:
            x = self._get_features(time_index)
            edge_index = self._get_edge_index()
            edge_weight = self._get_edge_weight()
            y = self._get_target(time_index)
            additional_features = self._get_additional_features(time_index)

            snapshot = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_weight,
                y=y,
                additional_features=additional_features
            )
        return snapshot

    def __next__(self):
        if self.t < self.snapshot_count:
            snapshot = self[self.t]
            self.t = self.t + 1
            return snapshot
        else:
            self.t = 0
            raise StopIteration

    def __iter__(self):
        self.t = 0
        return self

    def __len__(self):
        return self.snapshot_count
