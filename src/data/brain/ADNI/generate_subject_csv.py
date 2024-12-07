import pandas as pd
import numpy as np
import argparse
import os


def generate_subject_info(df):
    subject_info = {rid: dict() for rid in df['RID'].unique()}
    for rid in subject_info:
        sub_df = df[df['RID'] == rid]
        subject_info[rid]['viscode'] = list(sub_df['VISCODE'])
        subject_info[rid]['DX_CHANGE'] = list(sub_df['DXCHANGE_NEW'])
        subject_info[rid]['APOE4'] = list(sub_df['APOE4'])
        subject_info[rid]['DX_bl'] = list(sub_df['DX_bl'])
        subject_info[rid]['DX'] = list(sub_df['DX'])
    return subject_info


def get_roi_indices(header, roi_labels):
    return {roi: [header.get_loc(col) for col in header if col.endswith(roi)]
            for roi in roi_labels}


def concat_time(subject_df, indices, filter_mri, filter_ab, filter_tau, include_pet_volume):
    """

    :param subject_df:
    :param indices:
    :param filter_mri:
    :param filter_ab:
    :param filter_tau:
    :param include_pet_volume:
    :return:
    """
    if filter_mri:
        T, V, D = subject_df.shape[0], len(indices), int(1 * filter_mri + filter_ab + filter_tau)
    else:
        if include_pet_volume:
            T, V, D = subject_df.shape[0], len(indices), int(2 * filter_ab + 2 * filter_tau)
        else:
            T, V, D = subject_df.shape[0], len(indices), int(filter_ab + filter_tau)
    info = np.empty((T, V, D), np.float_)

    for r in range(T):
        row = subject_df.iloc[r]
        tmp = [row.iloc[idx].to_numpy() for idx in indices]
        tmp_arr = np.empty((V, D), np.float_)

        col_idx = 0
        if filter_mri:
            # FreeSurfer MRI ROI: [Cortical Volume, Surface Area, Thickness avg, Thickness std]
            # default is take volume and thickness average
            if any([not isinstance(roi[0], str) for roi in tmp]):
                mri = np.nan_to_num(np.array([[roi[0]] for roi in tmp]))
            else:
                mri = np.array([
                    np.array(
                        [roi[0][1:-1].split()[2]],
                        dtype=np.float_
                    )
                    for roi in tmp
                ])
            tmp_arr[:, col_idx:col_idx + 1] = mri
            col_idx += 1
        if filter_ab:
            # Each PETSurfer ROI: [SUVR, FreeSurfer defined volume],
            if include_pet_volume and not filter_mri:
                if any([not isinstance(roi[1], str) for roi in tmp]):
                    ab = np.nan_to_num(np.array([[roi[1]] for roi in tmp]))
                else:
                    ab = np.array([np.array(roi[1][1:-1].split(), dtype=np.float_) for roi in tmp])
                tmp_arr[:, col_idx:col_idx + 2] = ab
                col_idx += 2
            else:
                # only take SUVR
                if any([not isinstance(roi[1], str) for roi in tmp]):
                    ab = np.nan_to_num(np.array([[roi[1]] for roi in tmp]))
                else:
                    ab = np.array([np.array(roi[1][1:-1].split()[:1], dtype=np.float_) for roi in tmp])
                tmp_arr[:, col_idx:col_idx + 1] = ab
                col_idx += 1
        if filter_tau:
            # Each PETSurfer ROI: [SUVR, FreeSurfer defined volume]
            if include_pet_volume and not filter_mri:
                if any([not isinstance(roi[2], str) for roi in tmp]):
                    tau = np.nan_to_num(np.array([[roi[2]] for roi in tmp]))
                else:
                    tau = np.array([np.array(roi[2][1:-1].split(), dtype=np.float_) for roi in tmp])
                tmp_arr[:, col_idx:col_idx + 2] = tau
            else:
                # only take SUVR
                if any([not isinstance(roi[2], str) for roi in tmp]):
                    tau = np.nan_to_num(np.array([[roi[2]] for roi in tmp]))
                else:
                    tau = np.array([np.array(roi[2][1:-1].split()[:1], dtype=np.float_) for roi in tmp])
                tmp_arr[:, col_idx:col_idx + 1] = tau
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
    each subject dictionary contains 'viscode, DX_CHANGE, APOE4, DX_bl, DX, arr'
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

    subject_info = generate_subject_info(df)
    output = dict()
    for rid, d in subject_info.items():
        if len(d['viscode']) >= num_visits:
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
