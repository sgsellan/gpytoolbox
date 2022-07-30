import numpy as np
from gpytoolbox.halfedge_lengths_squared import halfedge_lengths_squared
from gpytoolbox.cotangent_weights_intrinsic import cotangent_weights_intrinsic

def cotangent_weights(V,F):
    # Builds the cotangent weights (cotangent/2) for each halfedge.
    #
    # The ordering convention for halfedges is the following:
    # [halfedge opposite vertex 0,
    #  halfedge opposite vertex 1,
    #  halfedge opposite vertex 2]
    #
    # Input:
    #       V  #V by 3 numpy array of mesh vertex positions
    #       F  #F by 3 int numpy array of face/edge vertex indeces into V
    #
    # Output:
    #       C  #F by 3 numpy array of cotangent weights for each halfedge

    l_sq = halfedge_lengths_squared(V,F)
    return cotangent_weights_intrinsic(l_sq,F)
