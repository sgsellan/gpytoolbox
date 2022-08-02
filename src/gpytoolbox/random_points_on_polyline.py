import numpy as np
from .edge_indices import edge_indices

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
        Matrix of polyline indeces into V. If None, the polyline is assumed to be open and ordered.

    Returns
    -------
    P : numpy double array
        Matrix of randomly sampled points that lay on the polyline
    N : numpy double array
        Matrix of outward facing polyline normals at P

    See Also
    --------
    edge_indices.

    Examples
    --------
    TODO
    """

    if (EC is None):
        EC = edge_indices(V.shape[0],closed=False)
    
    edge_lengths = np.linalg.norm(V[EC[:,1],:] - V[EC[:,0],:],axis=1)
    normalized_edge_lengths = np.cumsum(edge_lengths)/np.sum(edge_lengths)

    # These random numbers will choose the segment
    random_numbers = np.random.rand(n)
    # These random numbers will choose where in the chosen segment
    random_numbers_in_edge = np.random.rand(n)

    P = np.zeros((n,2))
    N = np.zeros((n,2))
    for i in range(n):
        # Pick the edge
        edge_index = np.argmax((random_numbers[i]<=normalized_edge_lengths))
        # Pick the point in the edge
        P[i,:] = random_numbers_in_edge[i]*V[EC[edge_index,0],:] + (1-random_numbers_in_edge[i])*V[EC[edge_index,1],:]
        #Compute normal
        n = np.array([-(V[EC[edge_index,1],1] - V[EC[edge_index,0],1]),V[EC[edge_index,1],0] - V[EC[edge_index,0],0]])
        N[i,:] =  n/np.linalg.norm(n)
    
    return P, N