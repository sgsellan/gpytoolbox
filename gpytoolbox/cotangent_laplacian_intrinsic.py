import numpy as np
import scipy as sp
from gpytoolbox.cotangent_weights_intrinsic import cotangent_weights_intrinsic

def cotangent_laplacian_intrinsic(l_sq,F,n=None):
    # Builds the (pos. def.) cotangent Laplacian for a triangle mesh using only
    # intrinsic information (squared halfedge edge lengths).
    #
    # Input:
    #       l_sq  #F by 3 numpy array of squared halfedge lengths as computed
    #             by halfedge_lengths_squared
    #       F  #F by 3 int numpy array of face/edge vertex indeces into V
    #       Optional:
    #                n  an integer denoting the number of vertices in the mesh
    #
    # Output:
    #       L  #V by #V scipy csr_matrix cotangent Laplacian

    assert F.shape[1] == 3
    assert l_sq.shape == F.shape

    if n==None:
        n = np.max(F)+1

    C = cotangent_weights_intrinsic(l_sq,F)

    rows = np.concatenate((F[:,0], F[:,1], F[:,1], F[:,2], F[:,2], F[:,0],
        F[:,0], F[:,1], F[:,2]))
    cols = np.concatenate((F[:,1], F[:,0], F[:,2], F[:,1], F[:,0], F[:,2],
        F[:,0], F[:,1], F[:,2]))
    data = np.concatenate((-C[:,2], -C[:,2], -C[:,0], -C[:,0], -C[:,1], -C[:,1],
        C[:,1]+C[:,2], C[:,2]+C[:,0], C[:,0]+C[:,1]))
    L = sp.sparse.csr_matrix((data, (rows,cols)), shape=(n,n))

    return L
