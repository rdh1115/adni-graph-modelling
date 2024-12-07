import numpy as np
from typing import List, Optional, Iterable

from torch_geometric.data import Dataset
from src.data.StaticGraphTemporalSignal import StaticGraphTemporalSignal
from src.data.pyg_dataset import GraphTemporalDataset


def wrap_brain_dataset(
        train_x,
        val_x,
        test_x,
        train_y,
        val_y,
        test_y,
        feature_ids,
        additional_features,
        additional_targets,
        edge_indices,
        edge_values,
        subject_info,
        scaler,
        **kwargs
) -> Optional[Dataset]:
    if not additional_features:
        # add additional features
        additional_features = list()
        if subject_info and feature_ids:
            for rids, info in zip(feature_ids, subject_info):  # iterate over [training, validation, test] datasets
                # find the additional feature keys and make a dictionary
                keys = [k for k, v in list(info.values())[0].items()
                        if isinstance(v, Iterable) and not k == 'arr']
                additional_feature = {k: list() for k in list(keys)}
                _, idx = np.unique(rids, return_index=True)
                unique_rids = rids[np.sort(idx)]
                for rid in unique_rids:
                    d = info[rid]
                    assert all(len(d[k]) == len(d['arr']) for k in keys)
                    for k, v in d.items():
                        if isinstance(v, Iterable) and not k == 'arr':
                            for i, val in enumerate(v):
                                additional_feature[k].append(val)
                additional_features.append(additional_feature)

    train_dataset = StaticGraphTemporalSignal(
        edge_index=edge_indices,
        edge_weight=edge_values,
        features=train_x,
        targets=train_y,
        rids=feature_ids[0],
        **additional_features[0] if subject_info else dict(),
        **additional_targets[0] if additional_targets else dict()
    )
    val_dataset = StaticGraphTemporalSignal(
        edge_index=edge_indices,
        edge_weight=edge_values,
        features=val_x,
        targets=val_y,
        rids=feature_ids[1],
        **additional_features[1] if subject_info else dict(),
        **additional_targets[1] if additional_targets else dict()
    )
    test_dataset = StaticGraphTemporalSignal(
        edge_index=edge_indices,
        edge_weight=edge_values,
        features=test_x,
        targets=test_y,
        rids=feature_ids[2],
        **additional_features[2] if subject_info else dict(),
        **additional_targets[2] if additional_targets else dict()
    )
    return GraphTemporalDataset(
        train_set=train_dataset,
        valid_set=val_dataset,
        test_set=test_dataset,
        scaler=scaler,
        **kwargs
    )
