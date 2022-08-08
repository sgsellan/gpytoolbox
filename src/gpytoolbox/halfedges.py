import numpy as np

def halfedges(F):
    """Given a triangle mesh with face indices F, returns all oriented halfedges
    as indices into the vertex array.

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
    he : (m,3,2) numpy int array
        halfedge list as per above conventions

    Examples
    --------
    TODO
    
    """
    
    assert F.shape[0] > 0
    assert F.shape[1] == 3

    he = np.block([[[F[:,1][:,None], F[:,2][:,None]]],
        [[F[:,2][:,None], F[:,0][:,None]]],
        [[F[:,0][:,None], F[:,1][:,None]]]]).transpose((1,0,2))
    return he

