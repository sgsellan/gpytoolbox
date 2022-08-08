import numpy as np
from gpytoolbox.halfedge_lengths_squared import halfedge_lengths_squared

def halfedge_lengths(V,F):
    """Given a triangle mesh V,F, returns the lengths of all halfedges.

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
    l : (m,3) numpy array
        lengths of halfedges

    Examples
    --------
    TODO
    
    """
    
    return np.sqrt(halfedge_lengths_squared(V,F))
