from .faces import faces

def boundary_faces(T):
    """Given a tet mesh with tet indices T, returns all unique oriented
    boundary faces as indices into the vertex array.
    Works only on manifold meshes.

    Parameters
    ----------
    T : (m,4) numpy int array.
        face index list of a triangle mesh

    Returns
    -------
    bF : (bf,3) numpy int array.
        indices of boundary faces into the vertex array

    Examples
    --------
    ```python
    import gpytoolbox as gpy
    v,t = gpy.regular_cube_mesh(4)
    bf = gpy.boundary_faces(t)
    ```
    """

    F,b = faces(T, return_boundary_indices=True)
    return F[b,:]
