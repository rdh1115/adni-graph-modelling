import os
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch_geometric.utils import dense_to_sparse

from src.data.brain.ADNI.generate_subject_csv import get_subject_dict
from src.data.utils import (
    DX_DICT,
    StandardScaler, MinMaxScaler,
    save_np_dict, load_np_dict,
    normalize, generate_split, generate_regression_brain,
    generate_regression_task
)


def get_brain_connectivity(data_dir, atlas='dk'):
    """

    :param data_dir: data directory
    :param atlas: the brain atlas
    :return: edge indices [2, number of edges], edge weight values [number of edges]
    """
    if atlas == 'dk':
        if 'subject_info' in data_dir:
            connectivity_fp = os.path.join(
                os.path.dirname(data_dir),
                'brain_connectivity.csv'
            )
        else:
            connectivity_fp = os.path.join(
                data_dir,
                'brain_connectivity.csv'
            )
        adj = pd.read_csv(connectivity_fp, header=None).to_numpy()[:68, :68]
        edge_indices, edge_values = dense_to_sparse(torch.tensor(adj))
        return edge_indices, edge_values
    else:
        raise NotImplementedError()


def get_all_subject_roi(
        data_dir,
        dataset_name,
        subject_info_fn=None
):
    dataset_path = os.path.join(data_dir, dataset_name)
    csv_fp = Path(dataset_path).glob('*.csv')
    if not csv_fp:
        raise ValueError(f'No preprocessed data found at {dataset_path}')
    subject_df = None
    if subject_info_fn:
        subject_info_fn = os.path.join(data_dir, subject_info_fn)
        subject_df = pd.read_csv(subject_info_fn)
        subject_df = subject_df[subject_df.columns[~subject_df.isnull().all()]]

    X, Y = [], {key: list() for key in subject_df.columns.to_list()}
    for fp in csv_fp:
        subject_id = fp.parts[-1].split('.')[0][2:]

        if subject_df is not None:
            subject_info = subject_df[subject_df['INDI_ID'] == int(subject_id)].to_dict('list')
            for k, v in subject_info.items():
                if k in Y:
                    Y[k] += v
        df = pd.read_csv(fp, header=None)
        X.append(df.to_numpy())
    X = np.array(X)
    assert len(set([len(v) for v in Y.values()])) == 1
    return X, Y


def get_raw_data_brain(
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
):
    # for brain data, the data is split and saved based on subjects and
    # then further processed and split into train/val/test
    # Create a list of scan types based on filter_list values
    scan_types = []
    if filter_list[0] == 1:
        scan_types.append('MRI')
    if filter_list[1] == 1:
        scan_types.append('Amyloid-Beta PET scans')
    if filter_list[2] == 1:
        scan_types.append('Tau PET scans')

    # Convert the list to a string for printing
    scan_types_str = ', '.join(scan_types)

    # Print the final statement
    print(
        f'Getting ADNI data with \n'
        f'num_visits: {num_visits}, \n'
        f'{scan_types_str}'
    )
    X_s, y_s, indices = list(), list(), list()
    filter_mri, filter_ab, filter_tau = filter_list
    suffix = ''
    suffix += f'visits{num_visits}_mri{filter_mri}_ab{filter_ab}_tau{filter_tau}_dx{filter_diagnosis}'
    if suffix in dataset_path:
        dir_path = dataset_path
        dataset_path = os.path.dirname(dataset_path)
    else:
        dir_path = os.path.join(dataset_path, f'subject_info_{suffix}')

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    for split in ['train', 'val', 'test']:
        fp = f'{split}.json'
        subject_info_fp = list(Path(dir_path).glob(fp))
        if subject_info_fp:
            print(f'Loading {fp} from {dir_path}')
            subject_dict = load_np_dict(subject_info_fp[0])
            indices.append(subject_dict.pop('idx'))
            X_s.append(subject_dict)
        else:  # read and save subject dict
            data_csv_path = os.path.join(
                dataset_path,
                'merge_process.csv'
            )
            print(f'Could not find {fp} in {dir_path}\n Generating subject info from {data_csv_path}... ')

            subject_dict = get_subject_dict(
                csv_fp=data_csv_path,
                connectivity_fp=os.path.join(
                    dataset_path,
                    'ROI_labels.csv'
                ),
                num_visits=num_visits,
                filter_mri=filter_mri,
                filter_ab=filter_ab,
                filter_tau=filter_tau,
                filter_diagnosis=filter_diagnosis,
                include_pet_volume=include_pet_volume,
            )

            (
                (train_x, val_x, test_x,
                 _, _, _, _),
                train_idx, val_idx, test_idx,
            ) = generate_split(subject_dict, None, split_ratio)
            train_fp = os.path.join(dir_path, f'train.json')
            val_fp = os.path.join(dir_path, f'val.json')
            test_fp = os.path.join(dir_path, f'test.json')
            save_np_dict(train_x, train_idx, train_fp)
            save_np_dict(val_x, val_idx, val_fp)
            save_np_dict(test_x, test_idx, test_fp)
            X_s += [train_x, val_x, test_x]
            indices += [train_idx, val_idx, test_idx]
            break

    print(f'total subjects: {len(X_s[0]) + len(X_s[1]) + len(X_s[2])}')
    print('train subjects: ', len(X_s[0]), 'val subjects: ', len(X_s[1]), 'test subjects: ', len(X_s[2]))
    X_features, feature_ids = list(), list()
    add_features_list, add_targets_list = list(), list()
    labels = list()
    class_init_prob = None
    scaler = None

    if norm and filter_list[0] == 1:  # only normalize MRI scans, PET is alreayd normalized
        tmp = list()
        # concat across subjects and normalize each vertex
        for rid, d in X_s[0].items():
            tmp.append(d['arr'])
        tmp = np.nan_to_num(np.concatenate(tmp, axis=0))[..., :2]
        data_min, data_max = tmp.min(axis=(0, 1)), tmp.max(axis=(0, 1))
        scaler = MinMaxScaler(np.array(data_min), np.array(data_max))
        # norm = False
        del tmp

    for i, x in enumerate(X_s):
        for rid, d in x.items():  # iterate over all subjects
            if norm and filter_list[0] == 1:
                d['arr'][..., :2] = scaler.transform(d['arr'][..., :2])

            if i == 0 and task == 'class':
                # all historical scans have the same labels
                # labels.append(max([DX_DICT[dx] for dx in d['DX']]))  # only the worst diagnosis is used,
                labels.append([DX_DICT[dx] for dx in d['DX']][-1])  # only the last diagnosis is used
            # if task == 'class':
            # if d['visits'] > n_hist:
            # only keep first and last + evenly spaced out middle indices
            # idx = np.round(np.linspace(0, d['visits'] - 1, n_hist)).astype(int)
            # x[rid] = {k: np.array(v)[idx] for k, v in d.items() if isinstance(v, Iterable)}

        if i == 0 and task == 'class':  # if task is classification, then use the training set to compute class prob
            labels, counts = np.unique(labels, return_counts=True)
            class_init_prob = counts / counts.sum()
            print(f'class labels: {labels}, counts: {counts}')
            print(f'initial class prob: {class_init_prob}')

        # if task is classification, then n_pred is 0,
        # and the models predicts the final diagnosis for all historical scans
        subject_X = {rid: generate_regression_task(d['arr'], n_hist, n_pred) for rid, d in x.items()}
        tmp = {rid: generate_regression_brain(d, n_hist, n_pred) for rid, d in x.items()}
        add_features = {rid: d[0] for rid, d in tmp.items()}
        add_targets = {rid: d[1] for rid, d in tmp.items()}

        features = np.nan_to_num(np.concatenate([arr[0] for arr in subject_X.values()]))
        if task == 'pred':
            targets = np.nan_to_num(np.concatenate([arr[1] for arr in subject_X.values()]))
        elif task == 'class':
            targets = np.concatenate([
                [
                    [max([DX_DICT[dx] for dx in x[rid]['DX']])] for _ in arr[0]
                ] for rid, arr in subject_X.items()
            ])
        else:
            raise ValueError(f'Unknown task: {task}')
        add_feature_keys = set.intersection(*[set(d.keys()) for d in add_features.values()])
        add_features = {
            k: np.concatenate([d[k] for d in add_features.values()])
            for k in add_feature_keys  # for each additional feature key, concatenate across subjects
        }
        add_targets = {
            k + '_target': np.concatenate([d[k + '_target'] for d in add_targets.values()])
            for k in add_feature_keys  # for each additional feature key, concatenate across subjects
        }

        X_features.append(features)
        y_s.append(targets)
        add_features_list.append(add_features)
        add_targets_list.append(add_targets)

        # get the RID corresponding to each feature vector
        idx = np.concatenate([[int(rid) for _ in arr[0]] for rid, arr in subject_X.items()])
        feature_ids.append(idx)
    for i in range(len(X_features)):
        assert X_features[i].shape[0] == y_s[i].shape[0] == feature_ids[i].shape[0]
    # if norm:
    #     (train_x, val_x, test_x,
    #      train_y, val_y, test_y, scaler) = normalize(*(X_features + y_s), dataset_type)

    train_x, val_x, test_x = X_features
    train_y, val_y, test_y = y_s
    return (train_x, val_x, test_x,
            train_y, val_y, test_y,
            feature_ids, add_features_list, add_targets_list,
            X_s, scaler, class_init_prob)
