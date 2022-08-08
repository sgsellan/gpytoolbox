import numpy as np
from gpytoolbox.halfedges import halfedges

def halfedge_lengths_squared(V,F):
    """Given a triangle mesh V,F, returns the lengths of all halfedges, squared.
    The reason to work with squared lengths instead of just lengths is that
    lengths are computed as squared lengths, and then often used as squared
    lengths, and thus keeping track of the squared lengths circumvents a
    lossy square root followed by a square.

    The ordering convention for halfedges is the following:
    [halfedge opposite vertex 0,
     halfedge opposite vertex 1,
     halfedge opposite vertex 2]

    Parameters
    ----------
    V : (n,d) numpy array
        vertex list of a triangle mesh
    F : (m,3) numpy int array
        face index list of a triangle mesh

    Returns
    -------
    l_sq : (m,3) numpy array
        squared lengths of halfedges

    Examples
    --------
    TODO
    
    """

    assert F.shape[0] > 0
    assert F.shape[1] == 3
    assert V.shape[0] > 0
    assert V.shape[1] > 0

    he = halfedges(F)

    edge_vectors = V[he[:,:,1],:] - V[he[:,:,0],:]
    return np.sum(edge_vectors**2, axis=-1)
