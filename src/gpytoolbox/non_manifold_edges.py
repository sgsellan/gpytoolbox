import numpy as np
from gpytoolbox.halfedges import halfedges

def non_manifold_edges(F):
    """Given a triangle mesh with face indices F, returns (unoriented) edges that are non-manifold; i.e., edges that are incident to more than two faces.

    Parameters
    ----------
    F : (m,3) numpy int array
        face index list of a triangle mesh

    Returns
    -------
    ne : (n,2) numpy int array
        list of unique non-manifold edges. Columns are sorted in ascending index order, and rows are sorted in lexicographic order.

    Notes
    -----
    It would be nice to also have a non_manifold_vertices function that wraps 2D and 3D functionality.

    Examples
    --------
    ```python
    from gpy import non_manifold_edges
    # bad mesh with one non-manifold edge in [0,2]
    f = np.array([[0,1,2],[0,2,3],[2,0,4]],dtype=int)
    ne = gpy.non_manifold_edges(f)
    # ne is now np.array([[0,2]])
    ```
    
    """

    assert F.shape[1] == 3

    he = halfedges(F).reshape(-1,2)
    he = np.sort(he, axis=1)
    # print(he)
    he_u = np.unique(he, axis=0, return_counts=True)
    # print(he)
    ne = he_u[0][he_u[1]>2]

    return ne

