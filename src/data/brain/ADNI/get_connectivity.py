# based on https://enigma-toolbox.readthedocs.io/en/latest/index.html
import os

import numpy as np
from nilearn import plotting
from src.data.brain.ADNI import process_merge as process_merge

ROI_map = process_merge.ROI_map

connec_map = {
    'accumb': 'ACCUMBENS_AREA',
    'amyg': 'AMYGDALA',
    'caud': 'CAUDATE',
    'hippo': 'HIPPOCAMPUS',
    'pal': 'PALLIDUM',
    'put': 'PUTAMEN',
    'thal': 'THALAMUS_PROPER',
}


def load_sc(data_dir=None):
    """Load structural connectivity data

        structural connectivity matrices were generated from preprocessed diffusion MRI data using MRtrix3.
        mapped onto the 68 cortical and 14 subcortical (including hippocampus) regions

        group-average normative structural connectome was defined using a distance-dependent thresholding procedure,
        which preserved the edge length distribution in individual patients,
        and was log transformed to reduce connectivity strength variance

        Parameters
        ----------
        data_dir : where the connectivity matrix information is located

        Returns
        -------
        strucMatrix_ctx : 2D ndarray
            structural connectivity, shape = (n+14, n+14)
        strucLabels_ctx : 1D ndarray
            region labels, shape = (n+14)
    """
    if not data_dir:
        root_pth = os.path.dirname(__file__)
        data_dir = os.path.join(root_pth, 'data')

    ctx = 'strucMatrix_with_sctx.csv'
    ctx_ipth = os.path.join(data_dir, 'Connectivity', ctx)

    ctxL = 'strucLabels_with_sctx.csv'
    ctxL_ipth = os.path.join(data_dir, 'Connectivity', ctxL)

    return np.loadtxt(ctx_ipth, dtype=np.float64, delimiter=','), \
        np.loadtxt(ctxL_ipth, dtype='str', delimiter=',')


def save_connectivity():
    # Load cortical-cortical structural connectivity data
    sc_ctx, sc_ctx_labels = load_sc()
    sc_plot = plotting.plot_matrix(sc_ctx, figure=(9, 9), labels=sc_ctx_labels, vmax=10, vmin=0, cmap='Blues')
    # plotting.show()
    new_labels = list()
    for label in sc_ctx_labels[:68]:
        label = label.upper()
        l_split = label.split('_')
        r = 'CTX_LH_' if l_split[0] == 'L' else 'CTX_RH_'
        label = r + l_split[1]
        roi = ROI_map[label]
        new_labels.append(roi)
    for label in sc_ctx_labels[68:]:
        if len(label.split('L')) > 1:
            l_split = label.split('L')[1]
            label = 'LEFT_' + connec_map[l_split]
        else:
            l_split = label.split('R')[1]
            label = 'RIGHT_' + connec_map[l_split]
        roi = ROI_map[label]
        new_labels.append(roi)
    print(new_labels)
    sc_plot = plotting.plot_matrix(sc_ctx, figure=(9, 9), labels=new_labels, vmax=10, vmin=0, cmap='Blues')
    # plotting.show()
    np.savetxt('ROI_labels.csv', new_labels, delimiter=',', fmt='%s')
    np.savetxt('brain_connectivity.csv', sc_ctx, delimiter=',', fmt='%s')
    print(sc_ctx.size, sc_ctx_labels.size)
