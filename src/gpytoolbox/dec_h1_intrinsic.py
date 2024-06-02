import numpy as np
import scipy as sp
from .halfedge_edge_map import halfedge_edge_map
from .cotangent_weights_intrinsic import cotangent_weights_intrinsic

def dec_h1_intrinsic(l_sq,F,E_to_he=None):
    """Builds the DEC 1-Hodge-star operator as described, for example, in Crane
    et al. 2013. "Digital Geometry Processing with Discrete Exterior Calculus".

    The edge labeling in E_to_he follows the convention from Gpytoolbox's
    `halfedge_edge_map`.

    The input mesh _must_ be a manifold mesh.

    Parameters
    ----------
    l_sq : (m,3) numpy array
        squared halfedge lengths as computed by halfedge_lengths_squared
    F : (m,3) numpy int array
        face index list of a triangle mesh
    E_to_he : (e,2,2) numpy int array, optional (default None)
        index map from e to corresponding row and col in the list of
        all halfedges `he` as computed by `halfedge_edge_map` for two
        halfedges (or -1 if only one halfedge exists)
        If absent, will be computed using `halfedge_edge_map`

    Returns
    -------
    h1 : (e,e) scipy csr_matrix
        DEC operator h1

    Examples
    --------
    ```python
    # Mesh in V,F
    l_sq = gpy.halfedge_lengths_squared(V,F)
    h1 = gpy.dec_h1_intrinsic(l_sq,F)
    ```
    
    """

    assert F.shape[1] == 3

    if E_to_he is None:
        _,_,_,E_to_he = halfedge_edge_map(F, assume_manifold=True)

    # A second halfedge exists for these
    se = E_to_he[:,1,0] >= 0

    C = cotangent_weights_intrinsic(l_sq,F)
    diag = C[E_to_he[:,0,0],E_to_he[:,0,1]]
    diag[se] += C[E_to_he[se,1,0],E_to_he[se,1,1]]
    h1 = sp.sparse.diags(diag, format='csr',
        shape=(E_to_he.shape[0],E_to_he.shape[0]))

    return h1

