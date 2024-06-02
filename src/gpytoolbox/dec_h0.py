import numpy as np
import scipy as sp
from .halfedge_lengths_squared import halfedge_lengths_squared
from .dec_h0_intrinsic import dec_h0_intrinsic

def dec_h0(V,F):
    """Builds the DEC 0-Hodge-star operator as described, for example, in Crane
    et al. 2013. "Digital Geometry Processing with Discrete Exterior Calculus".

    Parameters
    ----------
    V : (n,d) numpy array
        vertex list of a triangle mesh
    F : (m,3) numpy int array
        face index list of a triangle mesh

    Returns
    -------
    h0 : (n,n) scipy csr_matrix
        DEC operator h0

    Examples
    --------
    ```python
    # Mesh in V,F
    h0 = gpy.dec_h0(V,F)
    ```
    
    """

    l_sq = halfedge_lengths_squared(V,F)
    return dec_h0_intrinsic(l_sq,F,n=V.shape[0])

