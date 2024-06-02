import numpy as np
import scipy as sp
from .doublearea_intrinsic import doublearea_intrinsic

def dec_h0inv_intrinsic(l_sq,F,n=None):
    """Builds the inverse DEC 0-Hodge-star operator as described, for example,
    in Crane et al. 2013. "Digital Geometry Processing with Discrete Exterior
    Calculus".

    Parameters
    ----------
    l_sq : (m,3) numpy array
        squared halfedge lengths as computed by halfedge_lengths_squared
    F : (m,3) numpy int array
        face index list of a triangle mesh
    n : int, optional (default None)
        number of vertices in the mesh.
        If absent, will try to infer from F.

    Returns
    -------
    h0inv : (n,n) scipy csr_matrix
        inverse of DEC operator h0

    Examples
    --------
    ```python
    # Mesh in V,F
    l_sq = gpy.halfedge_lengths_squared(V,F)
    h0inv = gpy.dec_h0inv_intrinsic(l_sq,F)
    ```
    
    """

    assert F.shape[1] == 3

    if n is None:
        n = np.max(F)+1

    A3 = 0.5/3. * doublearea_intrinsic(l_sq,F)
    i = np.concatenate((F[:,0],F[:,1],F[:,2]), axis=0)
    j = np.concatenate((F[:,0],F[:,1],F[:,2]), axis=0)
    k = np.nan_to_num(1. / np.concatenate((A3,A3,A3), axis=0))
    h0inv = sp.sparse.csr_matrix((k, (i,j)), shape=(n,n))

    return h0inv
