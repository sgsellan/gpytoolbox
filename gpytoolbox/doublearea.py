import numpy as np
from gpytoolbox.edge_indeces import edge_indeces


def doublearea(V,F=None):
    # Builds the finite elements gradient matrix using a piecewise linear hat functions basis.
    #
    # Input:
    #       V #V by 3 numpy array of mesh vertex positions
    #       F #F by 3 int numpy array of face/edge vertex indeces into V
    #
    # Output:
    #       A #F vector of twice the (unsigned) area/length 

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
        # There are two options: dimension two or three. If it's two, we'll add a third zero dimension for convenience
        if dim==2:
            V = np.hstack((V,np.zeros((V.shape[0],1))))

        i0 = F[:,0]
        i1 = F[:,1]
        i2 = F[:,2]

        v21 = V[i2,:] - V[i1,:]
        v02 = V[i0,:] - V[i2,:]

        n = np.cross(v21,v02,axis=1)
        dblA = np.linalg.norm(n,axis=1)

    return dblA
