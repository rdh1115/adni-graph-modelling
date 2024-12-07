# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from typing import Optional

from torch_geometric.data import Dataset
from src.data.StaticGraphTemporalSignal import StaticGraphTemporalSignal
from sklearn.model_selection import train_test_split
import torch
import numpy as np

from src.data.utils import preprocess_item
import pyximport

pyximport.install(setup_args={'include_dirs': np.get_include()})
import copy
from functools import lru_cache


class GraphTemporalDataset(Dataset):
    def __init__(
            self,
            dataset: StaticGraphTemporalSignal = None,
            seed: int = 0,
            split_ratio=(20, 10),
            scaler=None,
            train_idx: torch.Tensor = None,
            valid_idx: torch.Tensor = None,
            test_idx: torch.Tensor = None,
            train_set=None,
            valid_set=None,
            test_set=None,
            static_graph=True,
            graph_token=True,
    ):
        super().__init__()
        self.dataset = dataset
        self.graph_token = graph_token
        self.static_graph = static_graph
        self.scaler = scaler

        if self.static_graph:
            self.preprocessed = False
            self.adj, self.attn_bias, self.attn_edge_type = None, None, None
            self.spatial_pos, self.in_degree, self.out_degree = None, None, None

        if self.dataset is not None:
            self.num_data = self.dataset.snapshot_count
        self.seed = seed

        if train_idx is None and train_set is None:
            test_split, valid_split = split_ratio
            test_split, valid_split = test_split / 100, valid_split / 100
            print(
                f"creating train/valid/test datasets, ratio: "
                f"{1.0 - test_split - valid_split:2f}/{valid_split: 2f}/{test_split: 2f}"
            )
            valid_split = valid_split / (1.0 - test_split)

            train_valid_idx, test_idx = train_test_split(
                np.arange(self.num_data),
                test_size=test_split,
                random_state=seed,
                shuffle=True
            )
            train_idx, valid_idx = train_test_split(
                train_valid_idx,
                test_size=valid_split,
                random_state=seed,
                shuffle=True
            )

            self.train_idx = torch.from_numpy(train_idx)
            self.valid_idx = torch.from_numpy(valid_idx)
            self.test_idx = torch.from_numpy(test_idx)
            self.train_data = self.index_select(self.train_idx)
            self.valid_data = self.index_select(self.valid_idx)
            self.test_data = self.index_select(self.test_idx)
            assert len(self.train_data) == len(self.train_idx)
            assert len(self.valid_data) == len(self.valid_idx)
            assert len(self.test_idx) == len(self.test_idx)
        elif train_set is not None:
            self.num_data = len(train_set) + len(valid_set) + len(test_set)
            self.train_data = self.create_subset(train_set)
            self.valid_data = self.create_subset(valid_set)
            self.test_data = self.create_subset(test_set)
            self.train_idx = None
            self.valid_idx = None
            self.test_idx = None
        else:
            self.num_data = len(train_idx) + len(valid_idx) + len(test_idx)
            self.train_idx = train_idx
            self.valid_idx = valid_idx
            self.test_idx = test_idx
            self.train_data = self.index_select(self.train_idx)
            self.valid_data = self.index_select(self.valid_idx)
            self.test_data = self.index_select(self.test_idx)

        self.__indices__ = None

    def create_subset(self, subset):
        dataset = copy.copy(self)
        dataset.dataset = subset
        dataset.num_data = len(subset)
        dataset.__indices__ = None
        dataset.train_data = None
        dataset.valid_data = None
        dataset.test_data = None
        dataset.train_idx = None
        dataset.valid_idx = None
        dataset.test_idx = None
        return dataset

    def index_select(self, idx):
        dataset = copy.copy(self)
        if isinstance(idx, slice) or isinstance(idx, np.ndarray) or isinstance(idx, torch.Tensor):
            dataset.dataset = self.dataset[idx]

        dataset.num_data = idx.shape[0]

        dataset.__indices__ = idx
        dataset.train_data = None
        dataset.valid_data = None
        dataset.test_data = None
        dataset.train_idx = None
        dataset.valid_idx = None
        dataset.test_idx = None

        assert dataset.dataset.snapshot_count == dataset.num_data
        return dataset

    def __getitem__(self, idx):
        if isinstance(idx, int):
            # only the x is different if the graph is static
            item = self.dataset[idx]
            item.idx = idx
            if not self.preprocessed:
                item = preprocess_item(item, self.graph_token)
                self.adj = item.adj
                self.edge_index = item.edge_index
                self.attn_bias = item.attn_bias
                self.attn_edge_type = item.attn_edge_type
                self.spatial_pos = item.spatial_pos
                self.in_degree = item.in_degree
                self.out_degree = item.out_degree
                self.edge_input = item.edge_input
                self.preprocessed = True
            else:
                item.edge_index = self.edge_index
                item.adj = self.adj
                item.attn_bias = self.attn_bias
                item.attn_edge_type = self.attn_edge_type
                item.spatial_pos = self.spatial_pos
                item.in_degree = self.in_degree
                item.out_degree = self.out_degree
                item.edge_input = self.edge_input
            return item
        else:
            return self.index_select(idx)

    def get(self, idx: int) -> Optional[dict]:
        return self.__getitem__(idx)

    def __len__(self):
        return self.num_data

    def len(self) -> int:
        return self.__len__()

    # def __iter__(self):
    #     self.t = 0
    #     print('iter start')
    #     return self
    #
    # def __next__(self):
    #     if self.t < len(self.dataset.features):
    #         snapshot = self[self.t]
    #         self.t += 1
    #         print('t < len', self.t)
    #         return snapshot
    #     else:
    #         self.t = 0
    #         raise StopIteration
