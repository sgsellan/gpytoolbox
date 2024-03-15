import numpy as np
import scipy as sp
from .halfedge_edge_map import halfedge_edge_map

def dec_d1(F,E_to_he=None):
    """Builds the DEC d1 operator as described, for example, in Crane et al.
    2013. "Digital Geometry Processing with Discrete Exterior Calculus".

    The edge labeling in E_to_he follows the convention from Gpytoolbox's
    `halfedge_edge_map`.

    The input mesh _must_ be a manifold mesh.

    Parameters
    ----------
    F : (m,3) numpy int array
        face index list of a triangle mesh
    E_to_he : (e,2,2) numpy int array, optional (default None)
        index map from e to corresponding row and col in the list of
        all halfedges `he` as computed by `halfedge_edge_map` for two
        halfedges (or -1 if only one halfedge exists)
        If absent, will be computed using `halfedge_edge_map`

    Returns
    -------
    d1 : (m,e) scipy csr_matrix
        DEC operator d1

    Examples
    --------
    ```python
    # Mesh in V,F
    d1 = gpy.dec_d1(F)
    ```
    
    """

    assert F.shape[1] == 3

    if E_to_he is None:
        _,_,_,E_to_he = halfedge_edge_map(F, assume_manifold=True)

    # A second halfedge exists for these
    se = E_to_he[:,1,0] >= 0

    i = np.concatenate((E_to_he[:,0,0], E_to_he[se,1,0]), axis=0)
    j = np.concatenate((np.arange(E_to_he.shape[0]),
        np.arange(E_to_he.shape[0])[se]), axis=0)
    k = np.concatenate((np.ones(E_to_he.shape[0], dtype=float),
        -np.ones(np.sum(se), dtype=float)), axis=0)
    d1 = sp.sparse.csr_matrix((k, (i,j)), shape=(F.shape[0],E_to_he.shape[0]))

    return d1
