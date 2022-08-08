from gpytoolbox.edges import edges

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
    TODO
    
    """

    E,b = edges(F, return_boundary_indices=True)
    return E[b,:]
