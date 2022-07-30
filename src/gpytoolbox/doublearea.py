import numpy as np
from gpytoolbox.halfedge_lengths_squared import halfedge_lengths_squared
from gpytoolbox.edge_indeces import edge_indeces
from gpytoolbox.doublearea_intrinsic import doublearea_intrinsic

def doublearea(V,F=None):
    # Construct the doublearea of each element of a line or triangle mesh
    #
    # Input:
    #       V  #V by 3 numpy array of mesh vertex positions
    #       F  #F by 3 int numpy array of face/edge vertex indices into V
    #
    # Output:
    #       A  #F vector of twice the (unsigned) area/length 

    # if you didn't pass an F then this is a ordered polyline
    if (F is None):
        F = edge_indeces(V.shape[0])

    dim = V.shape[1]
    simplex_size = F.shape[1]
    # Option 1: simplex size is two
    if simplex_size==2:
        # Then this is just finite difference with varying edge lengths
        A = np.linalg.norm(V[F[:,1],:] - V[F[:,0],:],axis=1)
        dblA = 2.0*A
        
    if simplex_size==3:
        l_sq = halfedge_lengths_squared(V,F)
        dblA = doublearea_intrinsic(l_sq,F)

    return dblA
