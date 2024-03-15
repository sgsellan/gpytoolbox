import numpy as np
import scipy as sp
from .halfedge_edge_map import halfedge_edge_map

def dec_d0(F,E=None,n=None):
    """Builds the DEC d0 operator as described, for example, in Crane et al.
    2013. "Digital Geometry Processing with Discrete Exterior Calculus".

    The edge labeling in E follows the convention from Gpytoolbox's
    `halfedge_edge_map`.

    The input mesh _must_ be a manifold mesh.

    Parameters
    ----------
    F : (m,3) numpy int array
        face index list of a triangle mesh
    E : (e,2) numpy int array, optional (default None)
        edge index list of a triangle mesh.
        If absent, will be computed using `halfedge_edge_map`
    n : int, optional (default None)
        number of vertices in the mesh.
        If absent, will try to infer from F.

    Returns
    -------
    d0 : (e,n) scipy csr_matrix
        DEC operator d0

    Examples
    --------
    ```python
    # Mesh in V,F
    d0 = gpy.dec_d0(F)
    ```
    
    """

    assert F.shape[1] == 3

    if n is None:
        n = np.max(F)+1

    if E is None:
        _,E,_,_ = halfedge_edge_map(F, assume_manifold=True)

    i = np.concatenate((np.arange(E.shape[0]), np.arange(E.shape[0])), axis=0)
    j = np.concatenate((E[:,0], E[:,1]), axis=0)
    k = np.concatenate((np.ones(E.shape[0], dtype=float),
        -np.ones(E.shape[0], dtype=float)))
    d0 = sp.sparse.csr_matrix((k, (i,j)), shape=(E.shape[0],n))

    return d0
