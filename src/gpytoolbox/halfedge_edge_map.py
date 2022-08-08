import numpy as np
from gpytoolbox.halfedges import halfedges
from gpytoolbox.array_correspondence import array_correspondence

def halfedge_edge_map(F, assume_manifold=True):
    """Computes unique edge indices, and a unique map from halfedges to edges,
    as well as its inverse.
    There is no particular ordering convention for edges.
    Boundary edges are guaranteed to be oriented as in F.

    The ordering convention for halfedges is the following:
    [halfedge opposite vertex 0,
     halfedge opposite vertex 1,
     halfedge opposite vertex 2]

    Parameters
    ----------
    F : (m,3) numpy int array
        face index list of a triangle mesh
    assume_manifold : bool, optional (default True)
        if this is true, will assume that F is manifold, and thus every edge can 
        be incident to at most two halfedges.
        The algorithm is very slow if this is False.

    Returns
    -------
    he : (m,3,2) numpy int array
        halfedge list as per above conventions
    E : (e,2) numpy int array
        indices of edges into the vertex array
    he_to_E : (m,3) numpy int array
        index map from he to corresponding row in E
    E_to_he : (e,2,2) numpy int array of list of m (k,2) numpy int arrays
        if assume_manifold, (e,2,2) index map from e to corresponding row and
        col in he for two halfedges (or -1 if only one halfedge exists);
        if not assume_manifold, python list with m entries of (k,2) arrays,
        where k is however many halfedges are adjacent to each edge. 

    Examples
    --------
    TODO
    
    """
    
    m = F.shape[0]
    assert m > 0
    assert F.shape[1] == 3

    he = halfedges(F)
    flat_he = np.concatenate([he[:,0,:],he[:,1,:],he[:,2,:]], axis=0)
    sorted_he = np.sort(flat_he, axis=1)
    unique_he, unique_indices, unique_count = np.unique(sorted_he, axis=0,
        return_index=True, return_counts=True)

    # Boundary edges come first, in order to preserve their orientation.
    bdry_E_to_flat_he = unique_indices[unique_count==1]
    n_b = bdry_E_to_flat_he.shape[0]
    bdry_E = flat_he[bdry_E_to_flat_he,:]

    # Interior edges come after.
    interior_E_to_first_flat_he = unique_indices[unique_count>1]
    n_i = interior_E_to_first_flat_he.shape[0]
    interior_E = flat_he[interior_E_to_first_flat_he,:]

    # Construct edges array
    E = np.concatenate([bdry_E,interior_E], axis=0)
    assert E.shape == unique_he.shape

    # Construct the E_to_he map
    def unflat_he(flat_he):
        return np.stack((flat_he%m, flat_he//m), axis=-1)
    if assume_manifold:
        E_to_he = np.full((E.shape[0],2,2), -1, dtype=F.dtype)
        if n_b>0:
            E_to_he[0:n_b,0,:] = unflat_he(bdry_E_to_flat_he)
        if n_i>0:
            E_to_he[n_b:(n_b+n_i),0,:] = \
            unflat_he(array_correspondence(interior_E,flat_he,axis=1))
            E_to_he[n_b:(n_b+n_i),1,:] = \
            unflat_he(array_correspondence(interior_E,np.flip(flat_he, axis=-1),axis=1))
    else:
        E_to_he = []
        bdry_E_to_he = unflat_he(bdry_E_to_flat_he)
        for e in range(n_b):
            E_to_he.append(bdry_E_to_he[e,:][None,:])
        for e in range(n_b, (n_b+n_i)):
            flat_he_inds = np.nonzero((E[e,:][None,:]==flat_he).all(axis=-1) |
                (np.flip(E[e,:],axis=-1)[None,:]==flat_he).all(axis=-1))[0]
            E_to_he.append(unflat_he(flat_he_inds))

    # Construct the he_to_E map
    he_to_E = np.full(F.shape, -1, dtype=F.dtype)
    if assume_manifold:
        he_r = E_to_he[:,:,0].flatten()
        he_c = E_to_he[:,:,1].flatten()
        es_to_put = np.stack((np.arange(E_to_he.shape[0]),np.arange(E_to_he.shape[0])), axis=-1).flatten()
        valid = he_r >= 0
        he_to_E[he_r[valid], he_c[valid]] = es_to_put[valid]
    else:
        for e in range(len(E_to_he)):
            he_to_E[E_to_he[e][:,0], E_to_he[e][:,1]] = e
    assert not np.any(he_to_E == -1)

    return he, E, he_to_E, E_to_he


