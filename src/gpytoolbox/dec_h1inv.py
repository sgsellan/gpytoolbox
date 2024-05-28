import numpy as np
import scipy as sp
from .halfedge_lengths_squared import halfedge_lengths_squared
from .dec_h1inv_intrinsic import dec_h1inv_intrinsic

def dec_h1inv(V,F,E_to_he=None):
    """Builds the inverse DEC 1-Hodge-star operator as described, for example,
    in Crane et al. 2013. "Digital Geometry Processing with Discrete Exterior
    Calculus".

    The edge labeling in E_to_he follows the convention from Gpytoolbox's
    `halfedge_edge_map`.

    The input mesh _must_ be a manifold mesh.

    Parameters
    ----------
    V : (n,d) numpy array
        vertex list of a triangle mesh
    F : (m,3) numpy int array
        face index list of a triangle mesh
    E_to_he : (e,2,2) numpy int array, optional (default None)
        index map from e to corresponding row and col in the list of
        all halfedges `he` as computed by `halfedge_edge_map` for two
        halfedges (or -1 if only one halfedge exists)
        If absent, will be computed using `halfedge_edge_map`

    Returns
    -------
    h1inv : (e,e) scipy csr_matrix
        inverse of DEC operator h1

    Examples
    --------
    ```python
    # Mesh in V,F
    h1inv = gpy.dec_h1inv(V,F)
    ```
    
    """

    l_sq = halfedge_lengths_squared(V,F)
    return dec_h1inv_intrinsic(l_sq,F,E_to_he=E_to_he)
