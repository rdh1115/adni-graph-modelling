import pandas as pd
import numpy as np
import argparse
import os


def generate_subject_info(df, add_feature_keys=None):
    subject_info = {rid: dict() for rid in df['RID'].unique()}
    if add_feature_keys is None:
        add_feature_keys = [
            'VISCODE', 'EXAMDATE', 'DXCHANGE_NEW', 'APOE4', 'DX_bl', 'DX', 'PTGENDER', 'PTEDUCAT',
            'MMSE', 'MOCA',
            'ADAS11', 'ADAS13', 'ADASQ4',
            'RAVLT_immediate', 'RAVLT_learning', 'RAVLT_forgetting', 'RAVLT_perc_forgetting',
            'EcogPtTotal', 'EcogSPTotal'
        ]
    for rid in subject_info:
        sub_df = df[df['RID'] == rid]
        for key in add_feature_keys:
            if key == 'EXAMDATE':
                dt = sub_df['EXAMDATE']
                # store another key for the time in days since the first visit
                subject_info[rid]['DELTA_DAYS'] = list((dt - dt.min()).dt.days)
                # also store the new age at each visit
                subject_info[rid]['AGE'] = list((dt - dt.min()).dt.days / 365.25 + sub_df['AGE'])
                subject_info[rid][key] = list(dt.astype(str))
            else:
                subject_info[rid][key] = list(sub_df[key])
    return subject_info


def get_roi_indices(header, roi_labels):
    return {
        roi: [header.get_loc(col) for col in header if col.endswith(roi)]
            for roi in roi_labels}


def extract_values(tmp, index, take_all=False):
    """Extracts values from the given index in tmp, handling string conversion."""
    if any(not isinstance(roi[index], str) for roi in tmp):
        return np.nan_to_num(np.array([[roi[index]] for roi in tmp]))
    if take_all:
        return np.array([np.array(roi[index][1:-1].split(), dtype=np.float64) for roi in tmp])
    else:
        return np.array([np.array(roi[index][1:-1].split()[:1], dtype=np.float64) for roi in tmp])


def concat_time(subject_df, indices, filter_mri, filter_ab, filter_tau, include_pet_volume):
    """
    Concatenates time-series data based on specified filters.
    """
    T, V = subject_df.shape[0], len(indices)
    if filter_mri:
        D = int(1 * filter_mri + filter_ab + filter_tau)
    else:
        if include_pet_volume:
            D = int(2 * filter_ab + 2 * filter_tau)
        else:
            D = int(filter_ab + filter_tau)
    info = np.empty((T, V, D), np.float64)

    for r in range(T):
        row = subject_df.iloc[r]
        tmp = [row.iloc[idx].to_numpy() for idx in indices]
        tmp_arr = np.empty((V, D), np.float64)

        col_idx = 0
        if filter_mri:
            # FreeSurfer MRI ROI: [Cortical Volume, Surface Area, Thickness avg, Thickness std]
            # default is take volume
            mri = extract_values(tmp, 0)
            tmp_arr[:, col_idx:col_idx + mri.shape[1]] = mri
            col_idx += mri.shape[1]
        if filter_ab:
            ab = extract_values(tmp, 1, take_all=(include_pet_volume and not filter_mri))
            tmp_arr[:, col_idx:col_idx + ab.shape[1]] = ab
            col_idx += ab.shape[1]
        if filter_tau:
            tau = extract_values(tmp, 2, take_all=(include_pet_volume and not filter_mri))
            tmp_arr[:, col_idx:col_idx + tau.shape[1]] = tau
        info[r] = tmp_arr
    return info


def filter_tracer_visits(df):
    fbp_filter = df[df['AMY_TRACER'] == 'FBP']
    fbb_filter = df[df['AMY_TRACER'] == 'FBB']

    return fbp_filter, fbb_filter


def filter_condition(df, conditions, mode='and'):
    if len(conditions) == 1:
        return df.loc[conditions[0]]
    if mode == 'and':
        return df.loc[np.logical_and(*conditions)]
    elif mode == 'or':
        return df.loc[np.logical_or(*conditions)]
    else:
        return df.loc[mode(*conditions)]


def get_subject_dict(
        csv_fp=None,
        connectivity_fp=None,
        num_visits=2,
        filter_mri=True,
        filter_ab=True,
        filter_tau=False,
        filter_diagnosis=False,
        include_pet_volume=False,
):
    """
    gather all subjects based on the filtering conditions

    :param csv_fp:
    :param connectivity_fp:
    :param num_visits:
    :param filter_mri:
    :param filter_ab:
    :param filter_tau:
    :param filter_diagnosis:
    :param include_pet_volume:
    :return: dictionary where keys are subject_ids and values are subject dictionaries
    """
    if csv_fp is None:
        data_dir = os.path.dirname(os.path.realpath(__file__))
        csv_fp = os.path.join(data_dir, 'merge_process.csv')
    if connectivity_fp is None:
        data_dir = os.path.dirname(os.path.realpath(__file__))
        connectivity_fp = os.path.join(data_dir, 'ROI_labels.csv')

    df = pd.read_csv(csv_fp, index_col=[0])
    ab_tracer = 'FBP'
    mri = pd.notnull(df['FS_type'])
    ab = (df['AMY_TRACER'] == ab_tracer)
    tau = (df['TAU_TRACER'] == 'FTP')

    df = df[(pd.notnull(df['DXCHANGE_NEW'])) & (df['DXCHANGE_NEW'] != -1.0)]
    df = df[(pd.notnull(df['DX'])) & (df['DX'] != 'nan')]
    if filter_diagnosis:
        df = df[df['DX'] == 'Dementia']
    roi_labels = pd.read_csv(connectivity_fp, header=None).to_numpy().flatten()
    cortical_idx = 68
    roi_indices = get_roi_indices(df.columns, roi_labels[:cortical_idx])

    filter_scans = list()
    if filter_mri:
        filter_scans += [mri]
    if filter_ab:
        filter_scans += [ab]
    if filter_tau:
        filter_scans += [tau]
    df = filter_condition(df, filter_scans)

    # order by RID and EXAMDATE
    df['EXAMDATE'] = pd.to_datetime(df['EXAMDATE'])
    df = df.sort_values(by=['RID', 'EXAMDATE'])

    subject_info = generate_subject_info(df)
    output = dict()
    for rid, d in subject_info.items():
        if len(d['VISCODE']) >= num_visits:
            subject_df = df[df['RID'] == rid]
            info_arr = concat_time(
                subject_df,
                list(roi_indices.values()),
                filter_mri, filter_ab, filter_tau,
                include_pet_volume=include_pet_volume,
            )
            d['arr'] = np.array(info_arr)
            output[rid] = d
    del df
    return output
