"""
from https://tadpole.grand-challenge.org/Data/
combine all preprocessed ADNI files

from ADNI Merge meta information file,
find the corresponding visit in FresSurfer MRI and UCBerkeley PET, and collate them into a big dataframe

match preprocessed csvs to MERGE by checking the roster ID (RID) first, then the visit code if available
for PET, since there's no visit code, use the exam date, and match with the closest exam date in MERGE
"""

import os
import glob
import numpy as np
import pandas as pd
import argparse
import time
from datetime import datetime


def find_nearest(array, value):
    array = [a if not pd.isnull(a) else np.inf for a in array]
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]


def get_args_parser():
    parser = argparse.ArgumentParser(
        description=
        'Aggregate from several ADNI spreadsheets, and produce a meta-info spreadsheet'
        'The script takes ADNIMERGE and adds extra MRI, PET biomarkers.'
        'Code can be modified easily to accomodate other biomarker files can be '
        r"""E.g. the following spreadsheets from ADNI study data can be used:
           * UCSFFSL_02_01_16.csv
           * UCSFFSL51_03_01_22.csv
           * UCSFFSX_11_02_15.csv
           * UCSFFSX51_11_08_19.csv
           * UCBERKELEYFDG_8mm.csv
           * UCBERKELEY_AMY_6MM.csv
           * UCBERKELEY_TAU_6MM.csv

           The simplest way to run it is with

               python3 process_meta.py

           If the spreadsheets are in a different directory (e.g. parent directory), run it as follows:

               python3 process_meta.py --csv_fp ..
        """, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--csv_fp', dest='csv_fp', default='./data',
                        help='folder of ADNI csv files')
    parser.add_argument('--output_fp', default='output.csv',
                        help='folder of output spreadsheet')
    parser.add_argument('--QC_control', action='store_false')
    return parser


def check_csv(dir_path, spreadsheet_list):
    """
    fucntion to help locate the csvs in a folder
    this function is made so that you don't have to manually delete the ADNI downloaded dates
    :param args: arg parse
    :param spreadsheet_list: list of the spreadsheet names
    :return:
    """

    found = []
    for s in spreadsheet_list:
        s = os.path.join(dir_path, s)
        search = glob.glob(f'{s}*.csv')
        if search:
            found += [search[0]]
        else:
            raise ValueError(f'File {s} not found')
    return found


def read_csvs(fps):
    return [pd.read_csv(fp) for fp in fps]


def process_dfs(df_list):
    # sort by quality, and drop failed rows
    df_list_new = list()

    for i, df in enumerate(df_list):
        df_header = df.columns.values
        if 'OVERALLQC' in df_header:
            visit_code = 'VISCODE2' if 'VISCODE2' in df_header else 'VISCODE'
            df['OVERALLQC_NR'] = df['OVERALLQC']
            df['TEMPQC_NR'] = df['TEMPQC']
            df['FRONTQC_NR'] = df['FRONTQC']
            df['PARQC_NR'] = df['PARQC']
            df['INSULAQC_NR'] = df['INSULAQC']
            df['OCCQC_NR'] = df['OCCQC']
            df['BGQC_NR'] = df['BGQC']
            df['CWMQC_NR'] = df['CWMQC']
            df['VENTQC_NR'] = df['VENTQC']
            mapping = {'Pass': 0, 'Partial': 1, 'Fail': 2}
            df.replace({'OVERALL_QC_NR': mapping, 'TEMPQC_NR': mapping, 'FRONTQC_NR': mapping,
                        'PARQC_NR': mapping, 'INSULAQC_NR': mapping, 'OCCQC_NR': mapping, 'CWMQC_NR': mapping,
                        'VENTQC_NR': mapping}, inplace=True)
            df['QCSUM_NR'] = df['TEMPQC_NR'] + df['FRONTQC_NR'] + df['PARQC_NR'] + df['INSULAQC_NR'] \
                             + df['OCCQC_NR'] + df['CWMQC_NR'] + df['VENTQC_NR']

            df.sort_values(by=['RID', 'EXAMDATE', 'OVERALLQC_NR', 'QCSUM_NR', 'RUNDATE', 'IMAGEUID'],
                           ascending=[True, True, True, True, False, False], inplace=True)
            drop_indices = np.logical_and(df['RID'] == 1066, df['EXAMDATE'] == '2011-12-19')
            drop_indices = np.logical_or(
                drop_indices,
                np.logical_and(df['RID'] == 1066, df[visit_code] == 'bl')
            )
            drop_indices = np.logical_or(drop_indices, df['OVERALLQC'] != 'Pass')
            keep_indices = np.logical_not(drop_indices)
            df = df[keep_indices]
            df = df.drop(
                [c for c in df.columns if 'QC' in c],
                axis=1,
            )
            df.reset_index(drop=True, inplace=True)
        elif 'qc_flag' in df.columns.values:
            df = df[df['qc_flag'] == 2]
            df = df.drop(
                [
                    'qc_flag'
                ],
                axis=1,
            )
        df_list_new += [df]
    return df_list_new


def parse_dx(dxChange, dxCurr, dxConv, dxConvType, dxRev):
    # get the dxchange index based on dx information
    if not np.isnan(dxChange):
        adni2_diag = dxChange
    else:
        if dxConv == 0:
            adni2_diag = int(dxCurr)
        elif dxConv == 1 and dxConvType == 1:
            adni2_diag = 4
        elif dxConv == 1 and dxConvType == 3:
            adni2_diag = 5
        elif dxConv == 1 and dxConvType == 2:
            adni2_diag = 6
        elif dxConv == 2:
            adni2_diag = int(dxRev) + 6
        elif np.isnan(dxConv):
            adni2_diag = -1
        else:
            print(dxChange, dxCurr, dxConv, dxConvType, dxRev)
            return ValueError('wrong values for diagnosis')
    return adni2_diag


def fix_dx(row):
    bl, dx = row['DX_bl'], row['DX']
    if pd.isnull(bl):
        return np.NaN

    if dx == 'CN':
        if bl == 'CN' or bl == 'SMC':
            return 1
        elif bl == 'EMCI':
            return 7
        elif bl == 'AD' or bl == 'LMCI':
            return 9
    elif dx == 'MCI':
        if bl == 'CN' or bl == 'SMC':
            return 4
        elif bl == 'EMCI' or bl == 'LMCI':
            return 2
        elif bl == 'AD':
            return 8
    elif dx == 'Dementia':
        if bl == 'CN' or bl == 'SMC':
            return 6
        elif bl == 'EMCI' or bl == 'LMCI':
            return 5
        elif bl == 'AD':
            return 3
    else:
        return np.NaN


def append_headers(old_header, dataframes):
    df_column_dict = dict()
    new_header = np.copy(old_header)
    avoid = ['VISCODE2', 'LONIUID', 'SITEID']

    # add the parse_dx output
    new_header = np.append(new_header, ['DXCHANGE_NEW'])
    df_headers = [df.columns.values for df in dataframes]
    df_column_dict[0] = [-1]
    # add new columns for each df
    for i, h in enumerate(df_headers):
        cols = list()
        for j, col in enumerate(h):
            if col not in old_header and col not in avoid:
                cols.append(j)
                col_ = str(i) + '_' + col
                new_header = np.append(new_header, col_)
        df_column_dict[i + 1] = cols
    return new_header, df_column_dict


def match_with_merge(row, df_new, patient_df_dict):
    header_new = df_new.columns
    if 'RID' in header_new:
        # match roster ID and store it in dict for faster look up
        roster_id = row['RID']
        if roster_id in patient_df_dict:
            patient_match = patient_df_dict[roster_id]
        else:
            patient_match = df_new[df_new['RID'] == roster_id]
            patient_df_dict[roster_id] = patient_match

        if patient_match.shape[0] == 0:
            return None
        else:
            if 'VISCODE2' in header_new or 'VISCODE' in header_new:
                viscode_version = 'VISCODE2' if 'VISCODE2' in header_new else 'VISCODE'
                viscodes = patient_match[viscode_version]
                r_viscode = row['VISCODE']
                if r_viscode == 'bl':  # ADNI merge only has bl, no sc VISCODE
                    if viscodes.str.contains('bl').any():
                        match = patient_match[patient_match[viscode_version] == r_viscode]
                    else:
                        # if there is no baseline, then consider sc as baseline
                        match = patient_match[(patient_match[viscode_version] == 'sc') |
                                              (patient_match[viscode_version] == 'scmri')]
                else:
                    match = patient_match[patient_match[viscode_version] == r_viscode]
            elif 'SCANDATE' in header_new:
                # scan date can be different from exam date in UCBerkeley files
                # find the closest date, and check ADNIMerge row has the PET tracer column
                scan_dates = [datetime.strptime(date, '%Y-%m-%d') if not pd.isnull(date) else pd.NaT
                              for date in patient_match['SCANDATE']]
                exam_date = datetime.strptime(row['EXAMDATE'], '%Y-%m-%d')
                idx, closest_date = find_nearest(scan_dates, exam_date)
                if abs(closest_date - exam_date).days > 30:
                    return None

                match = patient_match.iloc[[idx]]
                # if 'TRACER' in header_new:
                #     if match['TRACER'] == 'FBP':
                #         # av45, amyloid
                #         if np.isnan(row['AV45']) and :
                #             return None
                #     elif match['TRACER'] == 'FTP':
                #         # av1451, tau
                #         if np.isnan(row['TAU']):
                #             return None
            else:
                raise RuntimeError('dataframe does not contain visit ID nor exam date')
        if match.shape[0] == 1:
            # if 'EXAMDATE' in header_new:
            #     # try to match exam date
            #     exam_dates = [datetime.strptime(date, '%Y-%m-%d') if not pd.isnull(date) else pd.Nat
            #                   for date in patient_match['EXAMDATE']]
            #     exam_date = datetime.strptime(row['EXAMDATE'], '%Y-%m-%d')
            #     idx, closest_date = find_nearest(exam_dates, exam_date)
            #     date_match = patient_match.iloc[idx]
            #
            #     if (closest_date - exam_date).days > 0:
            #         bools = (match == date_match)
            #         bools[pd.isnull(match) & pd.isnull(date_match)] = True
            #         assert bools.all().all()
            return match
        elif match.shape[0] > 1:
            if 'ROINAME' in header_new:
                return match[match['ROINAME'] == 'MetaROI']

            # print(f'found more than 1 row for roster {roster_id}, visit code {r_viscode}')
            # print(match)
            if 'VERSION' in header_new:
                match = match.sort_values(by='VERSION', ascending=False)
            return match.iloc[[0]]
        else:
            return None
    else:
        raise RuntimeError('dataframe does not contain roster ID')


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    np.random.seed(1)

    csv_names = [
        'ADNIMERGE', 'DXSUM_PDXCONV_ADNIALL',
        'UCSFFSL_02_01_16', 'UCSFFSL51_03_01_22',
        'UCSFFSX_11_02_15', 'UCSFFSX51_11_08_19',
        'UCBERKELEYFDG_8mm',
        'UCBERKELEY_AMY_6MM', 'UCBERKELEY_TAU_6MM'
    ]
    csv_fps = check_csv(args.csv_fp, csv_names)
    dfs = read_csvs(csv_fps)
    if args.QC_control:
        dfs = process_dfs(dfs)

    df_merge, df_dx = dfs[0], dfs[1]
    ADNI1FSL_df, ADNI2FSL_df = dfs[2], dfs[3]  # Longitudinal FreeSurfer csvs
    ADNI1FSX_df, ADNI2FSX_df = dfs[4], dfs[5]  # Cross-sectional FreeSurfer csvs
    fdgPet_df = dfs[6]
    av45_df, av1451_df = dfs[7], dfs[8]  # amyloid and tau PET tracer data

    print('Finished reading csvs')
    start = time.time()
    header = df_merge.columns.values

    dx_change = df_dx.apply(
        lambda row: parse_dx(
            row['DXCHANGE'],
            row['DXCURREN'],
            row['DXCONV'],
            row['DXCONTYP'],
            row['DXREV']
        ), axis=1).to_numpy(dtype=int)
    df_dx['dx_change_new'] = dx_change

    header, column_dict = append_headers(header, dfs[2:])
    meta_arr = np.empty((df_merge.shape[0], header.shape[0]), np.object_)
    patient_dict = {i: dict() for i, df in enumerate(dfs[1:])}
    for r in range(df_merge.shape[0]):
        row = df_merge.iloc[r]
        row_arr = np.copy(row.to_numpy())
        for i, df in enumerate(dfs[1:]):
            matched_df = match_with_merge(row, df, patient_dict[i])

            if matched_df is not None:
                cols = matched_df.iloc[:, column_dict[i]].to_numpy()
            else:
                cols = np.empty((1, len(column_dict[i])))
                cols[:] = np.NaN
            if i == 0:
                change = fix_dx(row)
                dx_val = cols[0][0]
                if not np.isnan(change) and (np.isnan(dx_val) or dx_val == -1):
                    cols[0][0] = change
            row_arr = np.append(row_arr, cols)
        assert row_arr.shape[0] == header.shape[0]
        meta_arr[r, :row_arr.shape[0]] = row_arr

    content_arr = np.concatenate((header[np.newaxis, ...], meta_arr), axis=0)
    np.savetxt(args.output_fp, content_arr, delimiter=',', fmt='%s')
    t_delta = time.time() - start
    print(f'Finished in {t_delta}')
