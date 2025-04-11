from src.visualization.dataset import (
    prepare_data, get_vis_dataset, DictArgs
)
from src.utils.const import NODE_LABELS


def cluster_pipeline(
        grid_search_pipeline: callable,
        dataset_args=None,
        param_grids=None,
        seed=2025,
        n_clusters=3,
        n_splits=5,
        filter_regions=False,
        scaler='standard',
        *args,
        **kwargs,
):
    if not dataset_args:
        n_hist, n_pred, num_visits = 1, 0, 1
        filter_list = (0, 1, 0)
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
    loader_train, loader_val, loader_test, graph_info = get_vis_dataset(
        **dataset_args
    )
    edge_index, edge_weight = loader_train[0]['edge_index'], loader_train[0]['edge_attr']

    dataset_args.update(graph_info)
    dataset_args['dataset_name'] = 'ADNI'
    print(dataset_args)
    V = dataset_args['num_nodes']

    x_train, y_train, _ = prepare_data(loader_train)
    x_val, y_val, _ = prepare_data(loader_val)
    x_test, y_test, _ = prepare_data(loader_test)

    print(
        'X_train shape', x_train.shape,
        '\n X_val shape', x_val.shape,
        '\n X_test shape', x_test.shape,
    )
    if filter_regions:
        # Select major regions (left and right pairs)
        MAJOR_ROI_PAIRS = [
            (26, 60),  # Superior Frontal
            (28, 62),  # Superior Parietal
            (29, 61),  # Superior Temporal
            (9, 41),  # Lateral Occipital
            (4, 38)  # Entorhinal
        ]

        # Extract indices and names
        major_regions = [idx for pair in MAJOR_ROI_PAIRS for idx in pair]
        biomarkers = [(NODE_LABELS[l_idx], NODE_LABELS[r_idx]) for l_idx, r_idx in MAJOR_ROI_PAIRS]

        # Filter data
        x_train, x_val, x_test = x_train[..., major_regions], x_val[..., major_regions], x_test[..., major_regions]
        print(f'After filtering {len(major_regions)} biomarkers, new shape: {x_train.shape}')

    cluster_results, best_results = grid_search_pipeline(
        x_train, y_train,
        x_val, y_val,
        x_test, y_test,
        edge_index, edge_weight,
        n_clusters=n_clusters,
        seed=seed,
        *args, **kwargs,
    )
    return cluster_results, best_results, (loader_train, loader_val, loader_test), DictArgs(dataset_args)
