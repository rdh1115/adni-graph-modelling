""" gather ROI MRI and PET values from meta csv file

cortical regions have arrays of len 5 [MRI volume, MRI surface area, MRI thickness avg, MRI thickness std, PET SUVR, PET volume]
subcortical regions have arrays of len 3 [MRI volume, PET SUVR, PET volume]

for each patient visit, if there's MRI duplicates:
use cross-sectional MRI if available, longitudinal otherwise
the type used is stored in the column FS_type

if args.filter_scans is true,
filter out patient visits that do not have both MRI and PET scans
"""

import re
import numpy as np
import pandas as pd
import argparse

ROI_map = {
    'BRAINSTEM': 'ST1',
    'CC_ANTERIOR': 'ST2',
    'CC_CENTRAL': 'ST3',
    'CC_MID_ANTERIOR': 'ST4',
    'CC_MID_POSTERIOR': 'ST5',
    'CC_POSTERIOR': 'ST6',
    'CSF': 'ST7',
    'VENTRICLE_3RD': 'ST127',
    'VENTRICLE_4TH': 'ST9',
    'VENTRICLE_5TH': 'ST8',
    # ventricle is important
    'NON_WM_HYPOINTENSITIES': 'ST68',
    'WM_HYPOINTENSITIES': 'ST128',
    'OPTIC_CHIASM': 'ST69',

    'CTX_LH_BANKSSTS': 'ST13',
    'CTX_LH_CAUDALANTERIORCINGULATE': 'ST14',
    'CTX_LH_CAUDALMIDDLEFRONTAL': 'ST15',
    'CTX_LH_CUNEUS': 'ST23',
    'CTX_LH_ENTORHINAL': 'ST24',
    'CTX_LH_FRONTALPOLE': 'ST25',
    'CTX_LH_FUSIFORM': 'ST26',
    'CTX_LH_INFERIORPARIETAL': 'ST31',
    'CTX_LH_INFERIORTEMPORAL': 'ST32',
    'CTX_LH_ISTHMUSCINGULATE': 'ST34',
    'CTX_LH_LATERALOCCIPITAL': 'ST35',
    'CTX_LH_LATERALORBITOFRONTAL': 'ST36',
    'CTX_LH_LINGUAL': 'ST38',
    'CTX_LH_MEDIALORBITOFRONTAL': 'ST39',
    'CTX_LH_MIDDLETEMPORAL': 'ST40',
    'CTX_LH_PARACENTRAL': 'ST43',
    'CTX_LH_PARAHIPPOCAMPAL': 'ST44',
    'CTX_LH_PARSOPERCULARIS': 'ST45',
    'CTX_LH_PARSORBITALIS': 'ST46',
    'CTX_LH_PARSTRIANGULARIS': 'ST47',
    'CTX_LH_PERICALCARINE': 'ST48',
    'CTX_LH_POSTCENTRAL': 'ST49',
    'CTX_LH_POSTERIORCINGULATE': 'ST50',
    'CTX_LH_PRECENTRAL': 'ST51',
    'CTX_LH_PRECUNEUS': 'ST52',
    'CTX_LH_ROSTRALANTERIORCINGULATE': 'ST54',
    'CTX_LH_ROSTRALMIDDLEFRONTAL': 'ST55',
    'CTX_LH_SUPERIORFRONTAL': 'ST56',
    'CTX_LH_SUPERIORPARIETAL': 'ST57',
    'CTX_LH_SUPERIORTEMPORAL': 'ST58',
    'CTX_LH_SUPRAMARGINAL': 'ST59',
    'CTX_LH_TEMPORALPOLE': 'ST60',
    'CTX_LH_TRANSVERSETEMPORAL': 'ST62',
    'CTX_LH_UNKNOWN': 'ST64',

    'CTX_RH_BANKSSTS': 'ST72',
    'CTX_RH_CAUDALANTERIORCINGULATE': 'ST73',
    'CTX_RH_CAUDALMIDDLEFRONTAL': 'ST74',
    'CTX_RH_CUNEUS': 'ST82',
    'CTX_RH_ENTORHINAL': 'ST83',
    'CTX_RH_FRONTALPOLE': 'ST84',
    'CTX_RH_FUSIFORM': 'ST85',
    'CTX_RH_INFERIORPARIETAL': 'ST90',
    'CTX_RH_INFERIORTEMPORAL': 'ST91',
    'CTX_RH_ISTHMUSCINGULATE': 'ST93',
    'CTX_RH_LATERALOCCIPITAL': 'ST94',
    'CTX_RH_LATERALORBITOFRONTAL': 'ST95',
    'CTX_RH_LINGUAL': 'ST97',
    'CTX_RH_MEDIALORBITOFRONTAL': 'ST98',
    'CTX_RH_MIDDLETEMPORAL': 'ST99',
    'CTX_RH_PARACENTRAL': 'ST102',
    'CTX_RH_PARAHIPPOCAMPAL': 'ST103',
    'CTX_RH_PARSOPERCULARIS': 'ST104',
    'CTX_RH_PARSORBITALIS': 'ST105',
    'CTX_RH_PARSTRIANGULARIS': 'ST106',
    'CTX_RH_PERICALCARINE': 'ST107',
    'CTX_RH_POSTCENTRAL': 'ST108',
    'CTX_RH_POSTERIORCINGULATE': 'ST109',
    'CTX_RH_PRECENTRAL': 'ST110',
    'CTX_RH_PRECUNEUS': 'ST111',
    'CTX_RH_ROSTRALANTERIORCINGULATE': 'ST113',
    'CTX_RH_ROSTRALMIDDLEFRONTAL': 'ST114',
    'CTX_RH_SUPERIORFRONTAL': 'ST115',
    'CTX_RH_SUPERIORPARIETAL': 'ST116',
    'CTX_RH_SUPERIORTEMPORAL': 'ST117',
    'CTX_RH_SUPRAMARGINAL': 'ST118',
    'CTX_RH_TEMPORALPOLE': 'ST119',
    'CTX_RH_TRANSVERSETEMPORAL': 'ST121',
    'CTX_RH_UNKNOWN': 'ST123',

    'LEFT_ACCUMBENS_AREA': 'ST11',
    'LEFT_AMYGDALA': 'ST12',
    'LEFT_CAUDATE': 'ST16',
    'LEFT_CEREBELLUM_CORTEX': 'ST17',
    'LEFT_CEREBELLUM_WHITE_MATTER': 'ST18',
    'LEFT_CEREBRAL_WHITE_MATTER': 'ST20',
    'LEFT_CHOROID_PLEXUS': 'ST21',
    'LEFT_HIPPOCAMPUS': 'ST29',
    'LEFT_INF_LAT_VENT': 'ST30',
    'LEFT_LATERAL_VENTRICLE': 'ST37',
    'LEFT_PALLIDUM': 'ST42',
    'LEFT_PUTAMEN': 'ST53',
    'LEFT_THALAMUS_PROPER': 'ST61',
    'LEFT_VENTRALDC': 'ST65',
    'LEFT_VESSEL': 'ST66',

    'RIGHT_ACCUMBENS_AREA': 'ST70',
    'RIGHT_AMYGDALA': 'ST71',
    'RIGHT_CAUDATE': 'ST75',
    'RIGHT_CEREBELLUM_CORTEX': 'ST76',
    'RIGHT_CEREBELLUM_WHITE_MATTER': 'ST77',
    'RIGHT_CEREBRAL_WHITE_MATTER': 'ST79',
    'RIGHT_CHOROID_PLEXUS': 'ST80',
    'RIGHT_HIPPOCAMPUS': 'ST88',
    'RIGHT_INF_LAT_VENT': 'ST89',
    'RIGHT_LATERAL_VENTRICLE': 'ST96',
    'RIGHT_PALLIDUM': 'ST101',
    'RIGHT_PUTAMEN': 'ST112',
    'RIGHT_THALAMUS_PROPER': 'ST120',
    'RIGHT_VENTRALDC': 'ST124',
    'RIGHT_VESSEL': 'ST125',
    'CTX_LH_INSULA': 'ST129',
    'CTX_RH_INSULA': 'ST130',
}


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

               python3 TADPOLE_D1_D2.py --csv_fp ..
        """, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--meta_csv_fp', default='./output.csv',
                        help='folder of meta csv file')
    parser.add_argument('--output_fp', default='adni_roi.csv',
                        help='folder of output spreadsheet')
    parser.add_argument('--no_filter', action='store_false', default=True)
    return parser


def get_csv_column_indices(header, mapping, num_csvs=7):
    csv_columns = dict()
    start_dict = dict()
    invert_ROI_map = {anat: st for st, anat in mapping.items()}
    sts = sorted(mapping.values())
    anats = sorted(mapping.keys())
    rois = sorted([int(v.strip('ST')) for v in mapping.values()])

    for i in range(num_csvs):
        if i == 4:
            continue
        csv_columns[i] = {st: dict() for st in sts}
    csv_start, cur_idx = None, None
    for j, h in enumerate(header):
        if re.findall(r'^\d+_', h):  # if header column starts with i_
            csv_idx = int(re.findall(r'\d+', h)[0])
            if csv_idx != cur_idx:
                start_dict[csv_idx] = j
                cur_idx = csv_idx

            h_split = h.split(f'{csv_idx}_')
            if len(h_split) > 1:
                roi = ''
                for st in sts:
                    if st in h:
                        roi_idx = int(re.findall(r'\d+', h_split[1])[0])
                        roi = 'ST' + str(roi_idx)
                        info = ''
                        if 'SA' in h_split[1]:
                            info = 'mri surface area'
                        elif 'TA' in h_split[1]:
                            info = 'mri cortical thickness average'
                        elif 'CV' in h_split[1]:
                            info = 'mri volume cortical'
                        elif 'SV' in h_split[1]:
                            info = 'mri volume subcortical'
                        elif 'HS' in h_split[1]:
                            info = 'mri hippo volume'
                        elif 'TS' in h_split[1]:
                            info = 'mri cortical thickness standard deviation'
                        else:
                            raise RuntimeError('unexpected info type')
                        break
                    elif invert_ROI_map[st] in h:
                        if len(h_split[1].split('_VOLUME')) > 1:
                            if invert_ROI_map[st] == h_split[1].split('_VOLUME')[0]:
                                roi = st
                                info = 'pet volume'
                                break
                        elif len(h_split[1].split('_SUVR')) > 1:
                            if invert_ROI_map[st] == h_split[1].split('_SUVR')[0]:
                                roi = st
                                info = 'pet suvr'
                                break
                        else:
                            raise RuntimeError('unexpected info type')

                if roi and roi in sts:
                    csv_columns[csv_idx][roi][j] = info
    return csv_columns, start_dict


def get_ctx_wm_rois(csv_roi_dict):
    ctx_rois, wm_rois = set(), set()
    for i, d in csv_roi_dict.items():
        cur_rois = set()
        for st, col in d.items():
            if col:
                if len(col) >= 2:
                    # cortical region MRI have volume, surface area, thickness information
                    cur_rois.add(st)
                else:
                    # subcortical regions have only volume MRI information
                    wm_rois.add(st)
                    assert 'subcortical' in list(col.values())[0]
        if i == 0:
            ctx_rois = ctx_rois | cur_rois
        else:
            ctx_rois = ctx_rois & cur_rois
    return ctx_rois, wm_rois


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    num_csvs = 7
    meta_df = pd.read_csv(args.meta_csv_fp)
    meta_df_header = meta_df.columns

    csv_roi_dict, start_dict = get_csv_column_indices(meta_df_header, ROI_map, num_csvs)

    ctx_rois, wm_rois = get_ctx_wm_rois(csv_roi_dict)
    count = 0

    new_col_idx = meta_df_header.get_loc('DXCHANGE_NEW')
    header = meta_df_header[0:new_col_idx + 1]
    header = np.append(
        header,
        ['FS_type', 'AMY_TRACER', 'TAU_TRACER'] +
        ['CTX_MRI_' + str(i) for i in ctx_rois] +
        ['SUB_MRI_' + str(i) for i in wm_rois] +
        ['CTX_AMY_' + str(i) for i in ctx_rois] +
        ['SUB_AMY_' + str(i) for i in wm_rois] +
        ['CTX_TAU_' + str(i) for i in ctx_rois] +
        ['SUB_TAU_' + str(i) for i in wm_rois]

    )
    # print(ROI_map)

    rows_list = list()
    for r in range(meta_df.shape[0]):
        row = meta_df.iloc[r]

        fs_type = None

        mri_arr, amyloid_arr, tau_arr = dict(), dict(), dict()
        for i in reversed(range(num_csvs)):
            if i == 4:
                continue
            col_dict = csv_roi_dict[i]
            if i < 4:
                if pd.notnull(row.iloc[start_dict[i]]):
                    if not mri_arr:
                        for roi in ctx_rois:
                            mri_arr[roi] = row.iloc[list(col_dict[roi].keys())].to_numpy()
                            assert len(mri_arr[roi]) == 0 or len(mri_arr[roi]) == 4

                        for roi in wm_rois:
                            mri_arr[roi] = row.iloc[list(col_dict[roi].keys())].to_numpy()
                            assert len(mri_arr[roi]) == 0 or len(mri_arr[roi]) == 1

                        if i >= 2:
                            fs_type = 'X'
                        else:
                            fs_type = 'L'
            elif i > 4:
                if pd.notnull(row[f'{i}_TRACER']):
                    if i == 5:
                        if not amyloid_arr:
                            for roi in ctx_rois:
                                amyloid_arr[roi] = row.iloc[list(col_dict[roi].keys())].to_numpy()
                                assert len(amyloid_arr[roi]) == 0 or len(amyloid_arr[roi]) == 2
                            for roi in wm_rois:
                                amyloid_arr[roi] = row.iloc[list(col_dict[roi].keys())].to_numpy()
                                assert len(amyloid_arr[roi]) == 0 or len(amyloid_arr[roi]) == 2

                    elif i == 6:
                        if not tau_arr:
                            for roi in ctx_rois:
                                tau_arr[roi] = row.iloc[list(col_dict[roi].keys())].to_numpy()
                                assert len(tau_arr[roi]) == 0 or len(tau_arr[roi]) == 2

                            for roi in wm_rois:
                                tau_arr[roi] = row.iloc[list(col_dict[roi].keys())].to_numpy()
                                assert len(tau_arr[roi]) == 0 or len(tau_arr[roi]) == 2

        existing_info = row.iloc[:new_col_idx + 1].to_dict()
        existing_info['FS_type'] = fs_type
        existing_info['AMY_TRACER'] = row['5_TRACER']
        existing_info['TAU_TRACER'] = row['6_TRACER']
        new_info = dict()
        if not args.no_filter:
            if mri_arr and amyloid_arr and tau_arr:
                count += 1
                continue
        else:
            if mri_arr:
                for roi in ctx_rois:
                    new_info[f'CTX_MRI_{roi}'] = mri_arr[roi]
                for roi in wm_rois:
                    new_info[f'SUB_MRI_{roi}'] = mri_arr[roi]
            else:
                for roi in ctx_rois:
                    new_info[f'CTX_MRI_{roi}'] = pd.NaT
                for roi in wm_rois:
                    new_info[f'SUB_MRI_{roi}'] = pd.NaT
            if amyloid_arr:
                for roi in ctx_rois:
                    new_info[f'CTX_AMY_{roi}'] = amyloid_arr[roi]
                for roi in wm_rois:
                    new_info[f'SUB_AMY_{roi}'] = amyloid_arr[roi]
            else:
                for roi in ctx_rois:
                    new_info[f'CTX_AMY_{roi}'] = pd.NaT
                for roi in wm_rois:
                    new_info[f'SUB_AMY_{roi}'] = pd.NaT
            if tau_arr:
                for roi in ctx_rois:
                    new_info[f'CTX_TAU_{roi}'] = tau_arr[roi]
                for roi in wm_rois:
                    new_info[f'SUB_TAU_{roi}'] = tau_arr[roi]
            else:
                for roi in ctx_rois:
                    new_info[f'CTX_TAU_{roi}'] = pd.NaT
                for roi in wm_rois:
                    new_info[f'SUB_TAU_{roi}'] = pd.NaT
            existing_info.update(new_info)

        rows_list.append(existing_info)
        if r % 1000 == 0:
            print('row: ', r)
    df = pd.DataFrame(rows_list, columns=header)
    df.to_csv('merge_process.csv')
    print('CTX regions: ', [roi for roi in ctx_rois])
    print('SUB regions: ', [roi for roi in wm_rois])
    print(count)
