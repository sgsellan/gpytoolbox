import numpy as np
import warnings
from .edge_indices import edge_indices
from .random_points_on_mesh import random_points_on_mesh

def random_points_on_polyline(V, n, EC=None):
    """Generate points on the edges of a polyline.

    Use a uniform distribution to randomly distribute random points in a given polyline.

    Parameters
    ----------
    V : numpy double array
        Matrix of polyline vertices 
    n : int
        Number of desired points
    EC : numpy int array, optional (default None)
        Matrix of polyline indices into V. If None, the polyline is assumed to be open and ordered.

    Returns
    -------
    P : numpy double array
        Matrix of randomly sampled points that lay on the polyline
    N : numpy double array
        Matrix of outward facing polyline normals at P

    See Also
    --------
    sample_mesh
    edge_indices

    Examples
    --------
    TODO
    """

    raise Exception("random_points_on_polyline was deprecated in gpytoolbox-0.0.3 in favour of the more general random_points_on_mesh. This error message will disappear in gpytoolbox-0.0.4")

    # warnings.warn("random_points_on_polyline will be deprecated in gpytoolbox-0.0.3 in favour of the more general random_points_on_mesh",DeprecationWarning)

    # if (EC is None):
    #     EC = edge_indices(V.shape[0],closed=False)

    # x,I,_ = random_points_on_mesh(V, EC, n, return_indices=True)

    # vecs = V[EC[:,1],:] - V[EC[:,0],:]
    # vecs /= np.linalg.norm(vecs, axis=1)[:,None]
    # J = np.array([[0., -1.], [1., 0.]])
    # N = vecs @ J.T

    # sampled_N = N[I,:]
    
    return x, sampled_N
