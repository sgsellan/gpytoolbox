import numpy as np
import scipy as sp
from .doublearea_intrinsic import doublearea_intrinsic

def dec_h2inv_intrinsic(l_sq,F):
    """Builds the inverse DEC 2-Hodge-star operator as described, for example,
    in Crane et al. 2013. "Digital Geometry Processing with Discrete Exterior
    Calculus".

    Parameters
    ----------
    l_sq : (m,3) numpy array
        squared halfedge lengths as computed by halfedge_lengths_squared
    F : (m,3) numpy int array
        face index list of a triangle mesh

    Returns
    -------
    h2inv : (n,n) scipy csr_matrix
        inverse of DEC operator h2

    Examples
    --------
    ```python
    # Mesh in V,F
    l_sq = gpy.halfedge_lengths_squared(V,F)
    h2inv = gpy.dec_h2inv_intrinsic(l_sq,F)
    ```
    
    """

    assert F.shape[1] == 3

    A = 0.5 * doublearea_intrinsic(l_sq,F)
    h2inv = sp.sparse.diags(np.nan_to_num(1./A),
        shape=(F.shape[0],F.shape[0]), format='csr')

    return h2inv
