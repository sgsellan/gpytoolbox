import numpy as np
from scipy.sparse import csr_matrix

from gpytoolbox.edge_indices import edge_indices
from gpytoolbox.halfedge_lengths_squared import halfedge_lengths_squared
from gpytoolbox.massmatrix_intrinsic import massmatrix_intrinsic


def massmatrix(V,F=None,type='voronoi'):
    # Builds the finite elements mass matrix using a piecewise linear hat
    # function basis.
    #
    # Input:
    #       V  #V by 3 numpy array of mesh vertex positions, or polyline vertices
    #       F  #F by 3 int numpy array of face/edge vertex indices into V
    #          (or None if polyline)
    #       Optional:
    #           type either of 'voronoi' {default}, 'full', or 'barycentric'
    #
    # Output:
    #       M  #V by #V mass matrix

    # if you didn't pass an F then this is a ordered polyline
    if (F is None):
        F = edge_indices(V.shape[0])

    simplex_size = F.shape[1]
    # Option 1: simplex size is two
    if simplex_size==2:
        # Then this is just finite difference with varying edge lengths
        edge_lengths = np.linalg.norm(V[F[:,1],:] - V[F[:,0],:],axis=1)
        vals = np.concatenate((edge_lengths,edge_lengths))/2.
        I = np.concatenate((F[:,0],F[:,1]))
        M = csr_matrix((vals,(I,I)),shape=(V.shape[0],V.shape[0]))

    # Option 2: simplex size is three - use intrinsic function
    if simplex_size==3:
        l_sq = halfedge_lengths_squared(V,F)
        M = massmatrix_intrinsic(l_sq,F,n=V.shape[0],type=type)

    return M
