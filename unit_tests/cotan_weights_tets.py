import igl
import numpy as np
from scipy.sparse import csr_matrix, diags


def cotan_weights_tets(V, T):
    """
    Returns the cotan weights for a tet-mesh, implemented as described in
    "Algorithms and Interfaces for Real-Time Deformation of 2D and 3D Shapes" [Jacobson, 2013]
    :param V:   |V|xdim Vertices of your tet mesh
    :param T:   |T|x4 Indices into V for each tetrahedron
    :return:    |T|x6 sparse matrix of cotan weights, one for each dihedral in the tet and organized as follows
                        [12 20 01 30 31 32], where the indices ij in each entries represents the two triangles sandwiching the dihedral angle
    """

    #keep track of length of edge joining each pair of face for each tet [12 20 01 30 31 32]:
    l12 = np.linalg.norm(V[T[:, 3], :] - V[T[:, 0], :], 2, axis=1, keepdims=True)
    l20 = np.linalg.norm(V[T[:, 3], :] - V[T[:, 1], :], 2, axis=1, keepdims=True)
    l01 = np.linalg.norm(V[T[:, 3], :] - V[T[:, 2], :], 2, axis=1, keepdims=True)
    l30 = np.linalg.norm(V[T[:, 1], :] - V[T[:, 2], :], 2, axis=1, keepdims=True)
    l31 = np.linalg.norm(V[T[:, 2], :] - V[T[:, 0], :], 2, axis=1, keepdims=True)
    l32 = np.linalg.norm(V[T[:, 0], :] - V[T[:, 1], :], 2, axis=1, keepdims=True)
    l = np.concatenate([l12, l20, l01, l30, l31, l32], axis=1)

    #get the areas opposite verices of each quad [0 1 2 3]
    a0 = igl.doublearea(V, T[:, [1, 2, 3]])[:, np.newaxis]
    a1 = igl.doublearea(V, T[:, [2, 3, 0]])[:, np.newaxis]
    a2 = igl.doublearea(V, T[:, [3, 0, 1]])[:, np.newaxis]
    a3 = igl.doublearea(V, T[:, [0, 1, 2]])[:, np.newaxis]
    a =  np.concatenate([a0, a1, a2, a3], axis=1)

    # figure out sin_theta using law of sines: http://mathworld.wolfram.com/Tetrahedron.html
    volume = igl.volume(V, T)[:, np.newaxis]
    rhs = (2.0 / (3.0 * l)) * a[:, [1, 2, 0, 3, 3, 3] ] * a[:, [2, 0, 1, 0, 1, 2]]

    sin_alpha = volume / rhs

    #chuck out the theta, may cause sin issues (as done in libigl/gptoolbox)
    #this is the dihedral angle made for each face pair of each tet: [12 20 01 30 31 32]
    _, cos_alpha = igl.dihedral_angles(V, T)

    cotan_weights = (1./6.) * l * (cos_alpha / sin_alpha)

    return cotan_weights





