# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import cython
import numpy as np
from cython.parallel cimport prange, parallel
cimport numpy
import numpy

def floyd_warshall(adjacency_matrix):
    # O(n^3) complexity
    (nrows, ncols) = adjacency_matrix.shape
    assert nrows == ncols
    cdef unsigned int n = nrows

    # cast and copy the adj matrix
    adj_mat_copy = adjacency_matrix.astype(numpy.double, order='C', casting='safe', copy=True)
    assert adj_mat_copy.flags['C_CONTIGUOUS']
    cdef numpy.ndarray[numpy.double_t, ndim=2, mode='c'] M = adj_mat_copy

    cdef numpy.ndarray[long, ndim=2, mode='c'] path = numpy.zeros([n, n], dtype=numpy.int64)

    cdef unsigned int i, j, k
    cdef double M_ij, M_ik, cost_ikkj
    cdef double * M_ptr = &M[0, 0]
    cdef double * M_i_ptr
    cdef double * M_k_ptr

    # set unreachable nodes distance to inf
    for i in range(n):
        for j in range(n):
            if i == j:
                M[i][j] = 0
            elif M[i][j] == 0:
                M[i][j] = np.inf
    assert (numpy.diagonal(M) == 0.0).all()

    # floyed algo
    for k in range(n):
        M_k_ptr = M_ptr + n * k
        for i in range(n):
            M_i_ptr = M_ptr + n * i
            M_ik = M_i_ptr[k]
            for j in range(n):
                cost_ikkj = M_ik + M_k_ptr[j]
                M_ij = M_i_ptr[j]
                if M_ij > cost_ikkj:
                    M_i_ptr[j] = cost_ikkj
                    # save the intermediate variable k that is used in the SDP
                    path[i][j] = k

    # set unreachable path to 510
    for i in range(n):
        for j in range(n):
            if M[i][j] >= 510:
                path[i][j] = 510
                M[i][j] = 510

    return M, path

def get_all_edges(path, i, j):
    cdef unsigned int k = path[i][j]
    if k == 0:
        return []
    else:
        return get_all_edges(path, i, k) + [k] + get_all_edges(path, k, j)

def gen_edge_input(max_dist, path, edge_feat):
    (nrows, ncols) = path.shape
    assert nrows == ncols
    cdef unsigned int n = nrows
    cdef unsigned int max_dist_copy = max_dist

    path_copy = path.astype(long, order='C', casting='safe', copy=True)
    edge_feat_copy = edge_feat.astype(long, order='C', casting='safe', copy=True)
    assert path_copy.flags['C_CONTIGUOUS']
    assert edge_feat_copy.flags['C_CONTIGUOUS']

    cdef numpy.ndarray[long, ndim=4, mode='c'] edge_fea_all = -1 * numpy.ones(
        [n, n, max_dist_copy, edge_feat.shape[-1]], dtype=numpy.int64)
    cdef unsigned int i, j, k, path_len, cur

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if path_copy[i][j] == 510:
                continue

            # reconstructs the shortest path based on the intermediate k
            # path is [i, ..., k, ..., j]
            path = [i] + get_all_edges(path_copy, i, j) + [j]
            path_len = len(path) - 1
            for k in range(path_len):
                edge_fea_all[i, j, k, :] = edge_feat_copy[path[k], path[k + 1], :]

    return edge_fea_all
