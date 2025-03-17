import numpy as np
import scipy as sp

from .halfedges import halfedges
from .array_correspondence import array_correspondence
from .remove_unreferenced import remove_unreferenced

def cut_edges(F,E):
    """Cut a triangle mesh along a specified set of edges.

    Given a triangle mesh and a set of edges, this returns a new mesh that has been "cut" along those edges; meaning, such that the two faces incident on that edge are no longer combinatorially connected. This is done by duplicating vertices along the cut edges (see note), and creating a new geometrically identical edge between the duplicated vertices.

    Parameters
    ----------
    F : (m,3) numpy int array
        face index list of a triangle mesh
    E : (k,2) numpy int array
        index list of edges to cut, indexing the same array as F.
        If E contains edges that are not present in F, they will be ignored.

    Returns
    -------
    G : (m,3) numpy int array
        face index list of cut triangle mesh
    I : (l,) numpy int array
        list of indices into V of vertices in new mesh such that V[I,:] are the
        vertices in the new mesh.
        This takes care of correctly duplicating vertices.

    Notes
    -----
    Since only vertices that no longer share an edge in common are duplicated, you cannot cut a single interior edge. This function mirrors gptoolbox's cut_edges (https://github.com/alecjacobson/gptoolbox/blob/master/mesh/cut_edges.m)

    Examples
    --------
    ```python
    _,F = gpy.read_mesh("mesh.obj")
    E = np.array([[0,1], [1,2]])
    G,I = gpy.cut_edges(F,E)
    W = V[I,:]
    # My new mesh is now W,G
    ```
    """

    assert F.shape[0]>0, "F must be nonempty."
    assert F.shape[1]==3, "Only works for triangle meshes."
    n = np.max(F)+1
    if E.size==0:
        return np.arange(F.size[0]), np.arange(n)
    assert E.shape[1]==2, "E is a (k,2) matrix."

    # This code is a translation of https://github.com/alecjacobson/gptoolbox/blob/master/mesh/cut_edges.m by Alec Jacobson
    he = halfedges(F)
    flat_he = np.concatenate([he[:,0,:],he[:,1,:],he[:,2,:]], axis=0)
    sorted_he = np.sort(flat_he, axis=1)
    unique_he, unique_inverse = np.unique(sorted_he, axis=0,
        return_inverse=True)
    unique_he_to_F = sp.sparse.csr_matrix(
        (np.ones(unique_inverse.shape[0]),
            (unique_inverse,
                np.tile(np.arange(F.shape[0]),3))),
        shape=(unique_he.shape[0], F.shape[0])
        )

    FF = np.arange(3*F.shape[0]).reshape((-1,3), order='F')
    sorted_unique_he = np.sort(unique_he, axis=1)
    sorted_E = np.sort(E, axis=1)
    isin_unique_he_but_not_E = np.nonzero(
        array_correspondence(sorted_unique_he, sorted_E, axis=0) < 0)[0]
    noncut = sp.sparse.csr_matrix(
        (np.ones(isin_unique_he_but_not_E.shape[0]),
            (isin_unique_he_but_not_E, isin_unique_he_but_not_E)),
        shape=(unique_he.shape[0], unique_he.shape[0])
        )
    unique_he_to_EE = sp.sparse.csr_matrix(
        (np.ones(unique_inverse.shape[0]),
            (unique_inverse, np.arange(3*F.shape[0]))),
        shape=(unique_he.shape[0], 3*F.shape[0])
        )
    I = np.arange(3*F.shape[0]).reshape(F.shape[0], 3, order='F')
    VV_to_EE = sp.sparse.csr_matrix(
        (np.ones(6*F.shape[0]),
            (np.concatenate((FF[:,0],FF[:,1],FF[:,2],FF[:,0],FF[:,1],FF[:,2])),
                np.concatenate((I[:,1],I[:,2],I[:,0],I[:,2],I[:,0],I[:,1])))),
        shape=(3*F.shape[0], 3*F.shape[0])
        )
    VV_to_V = sp.sparse.csr_matrix(
        (np.ones(I.size), (I.ravel(), F.ravel())),
        shape=(3*F.shape[0], n)
        )
    A = (VV_to_EE * (unique_he_to_EE.T * noncut * unique_he_to_EE)
        * VV_to_EE.T).multiply(VV_to_V * VV_to_V.T)

    I = F.flatten(order='F')
    VV = np.zeros(np.max(FF)+1) #dummy vertex data for remove unreferenced
    _,labels = sp.sparse.csgraph.connected_components(A, return_labels=True)
    VV[labels] = labels
    I[labels] = I
    FF = labels[FF]
    W,_,IM,_ = remove_unreferenced(VV[:,None], FF, return_maps=True)
    I = I[(IM.ravel() < W.shape[0]) & (IM.ravel() >= 0)]
    G = IM.ravel(order='F')[FF]

    return G,I

