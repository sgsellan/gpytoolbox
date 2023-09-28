import numpy as np
import scipy as sp

from .edges import edges

def adjacency_matrix(F):
    """Computes the sparse adjacency matrix from a face list or a list of edges.

    Parameters
    ----------
    F : (m,k) triangle or edge list.
        If k=2, this is assumed to be an edge list;
        if k=3, this is assumed to be a triangle list of a triangle mesh.

    Returns
    -------
    A : (n,n) scipy csr sparse matrix.
        sparse adjacenct matrix

    Examples
    --------
    ```python
    V,F = gpy.read_mesh("mesh.obj")
    A = gpy.adjacency_matrix(F)
    ```
    """

    if F.size==0:
        return sp.sparse.csr_matrix()
    if F.shape[1]==2:
        E = F
    elif F.shape[1]==3:
        E = edges(F)
    else:
        assert False, "Unsupported dimension of F."

    n = np.max(E)+1
    A = sp.sparse.csr_matrix(
        (np.ones(2*E.shape[0]),
            (np.concatenate((E[:,0], E[:,1])), np.concatenate((E[:,1],E[:,0])))),
        shape=(n,n),
        dtype=int)

    return A