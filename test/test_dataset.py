import unittest
import torch
from functools import partial

from src.data.get_dataset import get_dataset
from src.data.utils import collator


class MyTestCase(unittest.TestCase):
    def test_get_dataset(self):
        n_hist, n_pred, num_visits = 1, 0, 1
        filter_list = (1, 1, 1)
        filter_diagnosis, include_pet_volume = False, False
        norm = False
        graph_token = False
        dataset_dir = ''
        dataset_args = {
            'dataset_dir': dataset_dir,
            'n_hist': n_hist,
            'n_pred': n_pred,
            'num_visits': num_visits,
            'filter_list': filter_list,
            'filter_diagnosis': filter_diagnosis,
            'include_pet_volume': include_pet_volume,
            'norm': norm,
            'graph_token': graph_token,
        }
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
        scaler = dataset_train.scaler
        sampler_train = torch.utils.data.SequentialSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)

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
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train,
            sampler=sampler_train,
            batch_size=1,
            num_workers=0,
            pin_memory=True,
            drop_last=True,
            collate_fn=partial(
                collator,
                max_node=num_nodes,
                spatial_pos_max=num_spatial,
                graph_token=graph_token,
                scaler=scaler,
            ),
        )
        batch = next(iter(data_loader_train))
        print(batch['edge_index'].shape, batch['edge_attr'].shape)


if __name__ == '__main__':
    unittest.main()
