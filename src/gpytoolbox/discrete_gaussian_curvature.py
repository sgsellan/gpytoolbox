import numpy as np
from .internal_angles import internal_angles
from .boundary_vertices import boundary_vertices

def discrete_gaussian_curvature(V, F):
    """
    Compute per-vertex angle defect Gaussian curvature.

    Parameters
    ----------
    V : ndarray
        N x 3 array of vertex positions.
    F : ndarray
        M x 3 array of face indices.

    Returns
    -------
    k : ndarray
        N x 1 array of discrete Gaussian curvature values.

    """

    # Compute the discrete Gaussian curvature
    N = V.shape[0]
    vec = lambda X: X.flatten()
    IA = internal_angles(V, F)
    k = 2 * np.pi - np.bincount(vec(F), weights=vec(IA), minlength=N)
    
    # Adjust curvature for boundary vertices
    b = boundary_vertices(F)
    k[b] -= np.pi
    
    return k
