import numpy as np
import torch

from src.data.get_dataset import get_dataset
from src.utils.const import ADD_FEATURE_KEYS
from src.baselines import GCNMae
from src.data.utils import DX_DICT
from src.data.utils import collator


class DictArgs:
    def __init__(self, d):
        for key, value in d.items():
            setattr(self, key, value)


def get_vis_dataset(
        dataset_dir,
        graph_token,
        n_hist,
        n_pred,
        num_visits,
        filter_list,
        filter_diagnosis,
        include_pet_volume,
        norm,
        *args,
        **kwargs,
):
    dataset_dict = get_dataset(
        dataset_name='ADNI',
        data_dir=dataset_dir,
        n_hist=n_hist,
        n_pred=n_pred,
        num_visits=num_visits,
        task='class',
        filter_list=filter_list,
        filter_diagnosis=filter_diagnosis,
        graph_token=graph_token,
        mode='finetune',
        include_pet_volume=include_pet_volume,
        norm=norm
    )
    dataset_train = dataset_dict['train_dataset']
    dataset_val = dataset_dict['valid_dataset']
    dataset_test = dataset_dict['test_dataset']

    data_sample = dataset_train[0]
    node_feature_dim = data_sample['x'].shape[-1]
    num_nodes = data_sample['adj'].shape[0]
    num_edges = len(data_sample['edge_attr'])

    # account for data.utils.collator changes
    num_spatial = torch.max(data_sample['spatial_pos']).item() + 1
    num_in_degree = torch.max(data_sample['in_degree']).item() + 1
    num_out_degree = torch.max(data_sample['out_degree']).item() + 1
    graph_info = {
        'node_feature_dim': node_feature_dim,
        'num_nodes': num_nodes,
        'num_edges': num_edges,
        'num_spatial': num_spatial,
        'num_in_degree': num_in_degree,
        'num_out_degree': num_out_degree
    }
    return dataset_train, dataset_val, dataset_test, graph_info


def prepare_data(data_loader, keys=ADD_FEATURE_KEYS):
    """
    Prepares the input and target data for sklearn.
    Flattens the input tensor and converts it into numpy arrays for sklearn compatibility.
    """
    all_samples = [batch['x'].numpy().flatten() for batch in data_loader]
    all_targets = [batch['y'].numpy().flatten() for batch in data_loader]
    add_features_dict = {
        k: [batch['additional_features'][k] for batch in data_loader] for k in keys
    }

    # Combine all batches into single arrays
    all_samples = np.stack(all_samples)
    all_targets = np.stack(all_targets).ravel()
    add_features_dict = {k: np.stack(v) for k, v in add_features_dict.items()}
    return all_samples, all_targets, add_features_dict
