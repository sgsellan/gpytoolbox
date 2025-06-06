import numpy as np
import scipy as sp
from .doublearea_intrinsic import doublearea_intrinsic

def dec_h2_intrinsic(l_sq,F):
    """Builds the DEC 2-Hodge-star operator as described, for example, in Crane
    et al. 2013. "Digital Geometry Processing with Discrete Exterior Calculus".

    Parameters
    ----------
    l_sq : (m,3) numpy array
        squared halfedge lengths as computed by halfedge_lengths_squared
    F : (m,3) numpy int array
        face index list of a triangle mesh

    Returns
    -------
    h2 : (n,n) scipy csr_matrix
        DEC operator h2

    Examples
    --------
    ```python
    # Mesh in V,F
    l_sq = gpy.halfedge_lengths_squared(V,F)
    h2 = gpy.dec_h2_intrinsic(l_sq,F)
    ```
    
    """

    assert F.shape[1] == 3

    A = 0.5 * doublearea_intrinsic(l_sq,F)
    h2 = sp.sparse.diags(A, shape=(F.shape[0],F.shape[0]), format='csr')

    return h2
