import numpy as np
from gpytoolbox.halfedge_lengths_squared import halfedge_lengths_squared
from gpytoolbox.cotangent_laplacian_intrinsic import cotangent_laplacian_intrinsic

def cotangent_laplacian(V,F):
    # Builds the (pos. def.) cotangent Laplacian for a triangle mesh.
    #
    # Input:
    #       V  #V by 3 numpy array of mesh vertex positions
    #       F  #F by 3 int numpy array of face/edge vertex indeces into V
    #
    # Output:
    #       L  #V by #V scipy csr_matrix cotangent Laplacian

    l_sq = halfedge_lengths_squared(V,F)
    return cotangent_laplacian_intrinsic(l_sq,F,n=V.shape[0])
