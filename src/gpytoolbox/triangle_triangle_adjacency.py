import numpy as np
from gpytoolbox.halfedges import halfedges
from gpytoolbox.array_correspondence import array_correspondence

def triangle_triangle_adjacency(F):
    """Given a manifold triangle mesh with face indices F, computes adjacency
    info between triangles
    
    The ordering convention for halfedges is the following:
    [halfedge opposite vertex 0,
     halfedge opposite vertex 1,
     halfedge opposite vertex 2]

    Parameters
    ----------
    F : (m,3) numpy int array
        face index list of a triangle mesh

    Returns
    -------
    TT : (m,3) numpy int array
        Index list specifying which face j is adjacent to which face i across
        the respective halfedge in position (i,j).
        If there is no adjacent face (boundary halfedge), the entry is -1.
    TTi : (m,3) numpy int array
        Index list specifying which halfedge of face j (0,1,2) is adjacent to i
        in position (i,j).

    Examples
    --------
    TODO
    
    """
    
    m = F.shape[0]
    assert m > 0
    assert F.shape[1] == 3

    he = halfedges(F)
    he_flat = np.concatenate((he[:,0,:], he[:,1,:], he[:,2,:]), axis=0)
    he_flip_flat = np.flip(he_flat, axis=-1)

    map_to_flip = array_correspondence(he_flat,he_flip_flat,axis=1)
    TT = np.reshape(np.where(map_to_flip<0, -1, map_to_flip % m), F.shape, order='F')
    TTi = np.reshape(np.where(map_to_flip<0, -1, map_to_flip // m), F.shape, order='F')

    return TT, TTi


