import warnings

import numpy as np
from scipy.spatial import cKDTree


def g_cross_function(points_i, points_j, radii, workers=16):
    """
    Compute multitype G_cross function G_ij(r).

    Args:
        points_i: N x 2 numpy array for type i points
        points_j: M x 2 numpy array for type j points
        radii: array of distances (r values)
        workers: number of workers for querying tree

    Returns:
        g_ij_r: array of G_ij(r) values
    """
    tree_j = cKDTree(points_j)
    # For each point of type i, find distance to nearest type j
    distances, _ = tree_j.query(points_i, k=1, workers=workers)

    g_ij_r = [(distances < r).mean() for r in radii]
    return np.array(g_ij_r)


def area_under_g_cross_curve(points_i, points_j, radii, slice=None, workers=16):
    if slice is None:
        warnings.warn("No slice provided, calculate area under the whole curve.")
        slice = (0, np.max(radii))
    if slice[1] > np.max(radii):
        raise ValueError("Slice cannot be larger than radii")
    g_ij = g_cross_function(points_i, points_j, radii, workers=workers)
    mask = (radii >= slice[0]) & (radii <= slice[1])
    r_selected = radii[mask]
    g_selected = g_ij[mask]

    auc = np.trapezoid(g_selected, r_selected)

    return auc
