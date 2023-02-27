import numpy as np
from gpytoolbox.boundary_edges import boundary_edges

def boundary_vertices(F):
    """Given a triangle mesh with face indices F, returns the indices of all
    boundary vertices.
    Works only on manifold meshes.

    Parameters
    ----------
    F : (m,3) numpy int array
        face index list of a triangle mesh

    Returns
    -------
    bV : (bv,) numpy int array
        list of indices into V of boundary vertices

    Examples
    --------
    ```python
    from gpytoolbox import read_mesh, boundary_vertices
    v,f = read_mesh("test/unit_tests_data/bunny_oded.obj")
    bv = boundary_vertices(f)
    ```
    
    """

    return np.unique(boundary_edges(F))

