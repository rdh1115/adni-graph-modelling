import unittest
import torch
from functools import partial

from src.data.get_dataset import get_dataset
from src.data.utils import collator


class MyTestCase(unittest.TestCase):
    def test_get_dataset(self):
        for n_visits in [1, 3]:
            for filter_diagnosis in [True, False]:
                if n_visits == 1:
                    filter_lists = [(1, 1, 1), (1, 1, 0), (0, 1, 1), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
                else:
                    if filter_diagnosis:
                        filter_lists = [(1, 0, 0)]
                    else:
                        filter_lists = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0)]
                for filter_list in filter_lists:
                    n_hist, n_pred, num_visits = 1, 0, n_visits
                    filter_list = filter_list
                    filter_diagnosis, include_pet_volume = filter_diagnosis, False
                    norm = True
                    graph_token = False
                    dataset_dir = ''
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


if __name__ == '__main__':
    unittest.main()
