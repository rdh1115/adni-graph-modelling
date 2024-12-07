import os

from src.data.wrapper import wrap_brain_dataset
from src.data.preprocess_brain import get_raw_data_brain, get_brain_connectivity


def get_dataset(
        mode: str = 'pretrain',
        data_dir: str = None,
        dataset_name: str = 'ADNI',
        n_hist: int = 12,
        n_pred: int = 12,
        split_ratio=(20, 10),
        graph_token=True,
        seed=0,
        task='pred',
        filter_list=(True, True, False),
        include_pet_volume=False,
        filter_diagnosis=False,
        num_visits=2,
        norm=True,
):
    assert mode in [
        "pretrain",
        "finetune",
        "valid",
        "test",
        "debug"
    ]
    if n_hist > 1:
        raise ValueError('n_hist should be 1 for better biological interpretation')
    if not data_dir:
        data_dir = os.path.dirname(os.path.realpath(__file__))
        dataset_path = os.path.join(data_dir, 'brain', dataset_name)
    else:
        dataset_path = data_dir

    assert dataset_name in ['ADNI'], 'Brain Datasets are: ADNI'
    assert n_hist <= num_visits

    if task == 'pred':
        assert n_hist + n_pred == num_visits, (f'For prediction, '
                                               f'number of visits {num_visits} must be equal to '
                                               f'n_hist {n_hist} + n_pred {n_pred}')
    elif task == 'class':
        if filter_diagnosis:
            # filter_diagnosis = False
            print('****Warning****\n'
                  'Should not filter diagnosis to only AD subjects when task is classification')
    (
        train_x, val_x, test_x,
        train_y, val_y, test_y,
        feature_ids, add_features, add_targets,
        subject_dict_list, scaler, class_init_prob
    ) = get_raw_data_brain(
        dataset_path,
        split_ratio,
        n_hist,
        n_pred,
        task,
        filter_list,
        filter_diagnosis,
        include_pet_volume,
        num_visits,
        norm,
    )
    if scaler:
        print(f'Using normalization with {str(scaler)}')
    else:
        print('*** warning: no normalization! ***')
    edge_indices, edge_values = get_brain_connectivity(dataset_path)
    subject_info = None if mode == 'pretrain' else subject_dict_list
    dataset = wrap_brain_dataset(
        train_x=train_x,
        val_x=val_x,
        test_x=test_x,
        train_y=train_y,
        val_y=val_y,
        test_y=test_y,
        feature_ids=feature_ids,
        additional_features=add_features,
        additional_targets=add_targets,
        edge_indices=edge_indices.numpy(),
        edge_values=edge_values.numpy(),
        graph_token=graph_token,
        subject_info=subject_info,
        seed=seed,
        scaler=scaler
    )

    INFO = {
        'train_dataset': dataset.train_data,
        'valid_dataset': dataset.valid_data,
        'test_dataset': dataset.test_data,
        'class_init_prob': class_init_prob,
    }

    print(f' > {dataset_name} loaded!')
    print(INFO)
    print(f' > dataset info ends')
    return INFO
