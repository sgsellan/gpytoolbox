import numpy as np
from gpytoolbox.halfedges import halfedges

def edges(F,
    return_boundary_indices=False,
    return_interior_indices=False,
    return_nonmanifold_indices=False):
    """Given a triangle mesh with face indices F, returns all unique unoriented
    edges as indices into the vertex array.
    There is no particular ordering convention for edges.
    Boundary edges are guaranteed to be oriented as in F.

    Parameters
    ----------
    F : (m,3) numpy int array
        face index list of a triangle mesh
    return_boundary_indices : bool, optional (default False)
        whether to return a list of indices into E denoting the boundary edges
    return_interior_indices : bool, optional (default False)
        whether to return a list of indices into E denoting the interior edges
    return_nonmanifold_indices : bool, optional (default False)
        whether to return a list of indices into E denoting the nonmanifold edges

    Returns
    -------
    E : (e,2) numpy int array
        indices of edges into the vertex array
    boundary_indices : if requested, (b,) numpy int array
        list of indices into E of boundary edges
    interior_indices : if requested, (i,) numpy int array
        list of indices into E of interior edges
    nonmanifold_indices : if requested, (nm,) numpy int array
        list of indices into E of nonmanifold edges

    Examples
    --------
    TODO
    
    """

    assert F.shape[0] > 0
    assert F.shape[1] == 3

    #Sort halfedges. Remove duplicates.
    he = halfedges(F)
    flat_he = np.concatenate([he[:,0,:],he[:,1,:],he[:,2,:]], axis=0)
    sorted_he = np.sort(flat_he, axis=1)
    unique_he, unique_indices, unique_count = np.unique(sorted_he, axis=0,
        return_index=True, return_counts=True)

    #Construct edge arrays by preserving the unique indices of boundary and
    # picking sorted orientation for interior edges.
    #Boundary edges have only one halfedge.
    bdry_edges = flat_he[unique_indices[unique_count==1],:]
    #Interior edges have two ore more halfedges.
    interior_edges = unique_he[unique_count>1,:]
    E = np.concatenate([bdry_edges,interior_edges], axis=0)
    assert E.shape == unique_he.shape

    if return_boundary_indices or return_interior_indices or return_nonmanifold_indices:
        retval = [E]
        if return_boundary_indices:
            retval.append(np.arange(0, bdry_edges.shape[0]))
        if return_interior_indices:
            retval.append(np.where(unique_count>1)[0])
        if return_nonmanifold_indices:
            #Nonmanifold edges have three or more halfedges.
            retval.append(np.where(unique_count>2)[0])
        return retval
    else:
        return E

