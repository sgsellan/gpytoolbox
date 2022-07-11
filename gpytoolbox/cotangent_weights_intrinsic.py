import numpy as np
from gpytoolbox.doublearea_intrinsic import doublearea_intrinsic

def cotangent_weights_intrinsic(l_sq,F):
    # Builds the cotangent weights (cotangent/2) for each halfedge using only
    # intrinsic information (squared halfedge edge lengths).
    #
    # The ordering convention for halfedges is the following:
    # [halfedge opposite vertex 0,
    #  halfedge opposite vertex 1,
    #  halfedge opposite vertex 2]
    #
    # Input:
    #       l_sq  #F by 3 numpy array of squared halfedge lengths as computed
    #             by halfedge_lengths_squared
    #       F  #F by 3 int numpy array of face/edge vertex indeces into V
    #
    # Output:
    #       C  #F by 3 numpy array of cotangent weights for each halfedge

    assert F.shape[1] == 3
    assert l_sq.shape == F.shape

    a,b,c = l_sq[:,0], l_sq[:,1], l_sq[:,2]
    A = doublearea_intrinsic(l_sq, F)

    # See https://github.com/libigl/libigl/blob/main/include/igl/cotmatrix_entries.cpp
    C = 0.25 * np.stack((b+c-a, c+a-b, a+b-c), axis=1) / A[:,None]

    return C
    