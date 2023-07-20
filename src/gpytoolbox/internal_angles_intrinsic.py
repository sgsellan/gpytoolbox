import numpy as np

import numpy as np

def internal_angles_intrinsic(l):
    """
    Compute internal angles per face (in radians) using edge lengths. This is a clone of gptoolbox/mesh/internalangles_intrinsic.m

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

    # Compute sides
    s23 = l[:, 0]
    s31 = l[:, 1]
    s12 = l[:, 2]

    # Compute angles
    a23 = np.arccos((s12**2 + s31**2 - s23**2) / (2 * s12 * s31))
    a31 = np.arccos((s23**2 + s12**2 - s31**2) / (2 * s23 * s12))
    a12 = np.arccos((s31**2 + s23**2 - s12**2) / (2 * s31 * s23))

    # Combine angles into an array
    A = np.column_stack([a23, a31, a12])

    return A
