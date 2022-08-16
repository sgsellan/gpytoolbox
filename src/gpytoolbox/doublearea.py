import numpy as np
from gpytoolbox.halfedge_lengths_squared import halfedge_lengths_squared
from gpytoolbox.edge_indices import edge_indices
from gpytoolbox.doublearea_intrinsic import doublearea_intrinsic

def doublearea(V,F=None,signed=False):
    """Construct the doublearea of each element of a line or triangle mesh.

    Parameters
    ----------
    V : (n,d) numpy array
        vertex list of a polyline or triangle mesh
    F : numpy int array, optional (default: None)
        if None or (m,2), interpret input as ordered polyline;
        if (m,3) numpy int array, interpred as face index list of a triangle
        mesh
    signed : bool, optional (default False)
        Whether to sign output using right-hand rule in 2D 

    Returns
    -------
    dblA : (m,) numpy double array
        vector of twice the (unsigned) area/length 

    See Also
    --------
    doublearea_intrinsic.

    Examples
    --------
    TO-DO
    """

    # if you didn't pass an F then this is a ordered polyline
    if (F is None):
        F = edge_indices(V.shape[0])

    dim = V.shape[1]
    simplex_size = F.shape[1]
    # Option 1: simplex size is two
    if simplex_size==2:
        # Then this is just finite difference with varying edge lengths
        A = np.linalg.norm(V[F[:,1],:] - V[F[:,0],:],axis=1)
        dblA = 2.0*A
        
    if simplex_size==3:
        if (signed and dim==2):
            # Then use determinant
            r = V[F[:,0],:] - V[F[:,2],:]
            s = V[F[:,1],:] - V[F[:,2],:]
            dblA = r[:,0]*s[:,1] - r[:,1]*s[:,0]
        else:
            l_sq = halfedge_lengths_squared(V,F)
            dblA = doublearea_intrinsic(l_sq,F)

    return dblA
