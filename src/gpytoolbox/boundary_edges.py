from .edges import edges

def boundary_edges(F):
    """Given a triangle mesh with face indices F, returns all unique oriented
    boundary edges as indices into the vertex array.
    Works only on manifold meshes.

    Parameters
    ----------
    F : (m,3) numpy int array.
        face index list of a triangle mesh

    Returns
    -------
    bE : (be,2) numpy int array.
        indices of boundary edges into the vertex array

    Examples
    --------
    ```python
    from gpytoolbox import read_mesh, boundary_edges
    v,f = read_mesh("test/unit_tests_data/bunny_oded.obj")
    be = boundary_edges(f)
    ```
    """

    E,b = edges(F, return_boundary_indices=True)
    return E[b,:]
