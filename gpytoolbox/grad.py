import numpy as np
from scipy.sparse import csr_matrix

from gpytoolbox.edge_indeces import edge_indeces

# From https://github.com/alecjacobson/gptoolbox/blob/master/mesh/grad.m

def grad(V,F=None):
    # Builds the finite elements gradient matrix using a piecewise linear hat functions basis.
    #
    # Input:
    #       V #V by 3 numpy array of mesh vertex positions
    #       F #F by 3 int numpy array of face/edge vertex indeces into V
    #
    # Output:
    #       G 

    # if you didn't pass an F then this is a ordered polyline
    if (F is None):
        F = edge_indeces(V.shape[0])

    dim = V.shape[1]
    simplex_size = F.shape[1]
    # Option 1: simplex size is two
    if simplex_size==2:
        # Then this is just finite difference with varying edge lengths
        edge_lengths = np.linalg.norm(V[F[:,1],:] - V[F[:,0],:],axis=1)
        I = np.linspace(0,F.shape[0]-1,F.shape[0],dtype=int)
        I = np.concatenate((I,I))
        J = np.concatenate((F[:,0],F[:,1]))
        vals = np.ones(F.shape[0])/edge_lengths
        vals = np.concatenate((-vals,vals))
        G = csr_matrix((vals,(I,J)),shape=(F.shape[0],V.shape[0]))



    return G
