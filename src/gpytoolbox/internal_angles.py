import numpy as np
from .halfedge_lengths_squared import halfedge_lengths_squared
from .internal_angles_intrinsic import internal_angles_intrinsic

def internal_angles(V,F):
    """Compute internal angles per face (in radians) for a triangle mesh. This is a clone of gptoolbox/mesh/internalangles.m

    Parameters
    ----------
    l : ndarray
        M x 3 matrix of edge lengths.

    Returns
    -------
    A : ndarray
        M x 3 list of triples of triangle angles. Each row contains the
        three angles of the corresponding triangle. The first angle is opposite the first edge, etc.
    """

    #l_sq = halfedge_lengths_squared(V,F)
    i1 = F[:, 0]
    i2 = F[:, 1]
    i3 = F[:, 2]

    s12 = np.linalg.norm(V[i2] - V[i1], axis=1)
    s13 = np.linalg.norm(V[i3] - V[i1], axis=1)
    s23 = np.linalg.norm(V[i3] - V[i2], axis=1)

    l = np.column_stack((s23, s13, s12))
    return internal_angles_intrinsic(l)
