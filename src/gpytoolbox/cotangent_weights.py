import numpy as np
from gpytoolbox.halfedge_lengths_squared import halfedge_lengths_squared
from gpytoolbox.cotangent_weights_intrinsic import cotangent_weights_intrinsic

def cotangent_weights(V,F):
    """Builds the cotangent weights (cotangent/2) for each halfedge.

    The ordering convention for halfedges is the following:
    [halfedge opposite vertex 0,
     halfedge opposite vertex 1,
     halfedge opposite vertex 2]

    Parameters
    ----------
    V : (n,d) numpy array
        vertex list of a triangle mesh
    F : (m,3) numpy int array
        face index list of a triangle mesh

    Returns
    -------
    C : (m,3) numpy array
        cotangent weights for each halfedge

    Examples
    --------
    TODO
    
    """

    l_sq = halfedge_lengths_squared(V,F)
    return cotangent_weights_intrinsic(l_sq,F)
