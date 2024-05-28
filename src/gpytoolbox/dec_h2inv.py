import numpy as np
import scipy as sp
from .halfedge_lengths_squared import halfedge_lengths_squared
from .dec_h2inv_intrinsic import dec_h2inv_intrinsic

def dec_h2inv(V,F):
    """Builds the inverse DEC 2-Hodge-star operator as described, for example,
    in Crane et al. 2013. "Digital Geometry Processing with Discrete Exterior
    Calculus".

    Parameters
    ----------
    V : (n,d) numpy array
        vertex list of a triangle mesh
    F : (m,3) numpy int array
        face index list of a triangle mesh

    Returns
    -------
    h2inv : (m,m) scipy csr_matrix
        inverse of DEC operator h2

    Examples
    --------
    ```python
    # Mesh in V,F
    h2inv = gpy.dec_h2inv(V,F)
    ```
    
    """

    l_sq = halfedge_lengths_squared(V,F)
    return dec_h2inv_intrinsic(l_sq,F)
