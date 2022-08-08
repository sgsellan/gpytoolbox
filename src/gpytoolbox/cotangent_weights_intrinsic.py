import numpy as np
from gpytoolbox.doublearea_intrinsic import doublearea_intrinsic

def cotangent_weights_intrinsic(l_sq,F):
    """Builds the cotangent weights (cotangent/2) for each halfedge using only
    intrinsic information (squared halfedge edge lengths).

    The ordering convention for halfedges is the following:
    [halfedge opposite vertex 0,
     halfedge opposite vertex 1,
     halfedge opposite vertex 2]

    Parameters
    ----------
    l_sq : (m,3) numpy array
        squared halfedge lengths as computed by halfedge_lengths_squared
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

    assert F.shape[1] == 3
    assert l_sq.shape == F.shape
    assert np.all(l_sq>=0)

    a,b,c = l_sq[:,0], l_sq[:,1], l_sq[:,2]
    A = doublearea_intrinsic(l_sq, F)

    # See https://github.com/libigl/libigl/blob/main/include/igl/cotmatrix_entries.cpp
    C = 0.25 * np.stack((b+c-a, c+a-b, a+b-c), axis=1) / A[:,None]

    return C
    