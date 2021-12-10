import igl
import numpy as np
from scipy.sparse import csr_matrix, diags
from cotan_weights_tets import cotan_weights_tets

def cotmatrix_tets(V, T):
    """
    Returns the cotangent Laplacian matrix for tet-meshes, implemented as described in
    "Algorithms and Interfaces for Real-Time Deformation of 2D and 3D Shapes" [Jacobson, 2013]
    :param V:   |V|xdim Vertices of your tet mesh
    :param T:   |T|x4 Indices into V for each tetrahedron
    :return:    |V|x|V| sparse csr_matrix representing the cotangent laplacian matrix for a tetrahedral mesh
    """
    #get cotan weights for tet mesh
    cotan_weights = cotan_weights_tets(V, T)

    #fill indices
    i = (T[:, [1, 2, 0, 3, 3, 3]]).flatten()
    j = (T[:, [2, 0, 1, 0, 1, 2]]).flatten()
    v =(cotan_weights).flatten()
    L = csr_matrix((v, (i, j)), shape=(V.shape[0], V.shape[0]))

    L += L.T

    diag_entries = -np.array(L.sum(1)).flatten()
    L += diags(diag_entries)
    return L





