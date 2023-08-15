import numpy as np
from .adjacency_dihedral_angle_matrix import adjacency_dihedral_angle_matrix

def edge_lengths(V, F):
    e1 = np.linalg.norm(V[F[:, 1], :] - V[F[:, 0], :], axis=1)
    e2 = np.linalg.norm(V[F[:, 2], :] - V[F[:, 1], :], axis=1)
    e3 = np.linalg.norm(V[F[:, 0], :] - V[F[:, 2], :], axis=1)
    return e1, e2, e3

def discrete_mean_curvature(V, F):
    """
    Computes the discrete mean curvature of a mesh at each vertex.

    Parameters
    ----------
    V : ndarray
        N x 3 array of vertex positions.
    F : ndarray
        M x 3 array of face indices.

    Returns
    -------
    H : ndarray
        N x 1 array of integrated mean-curvature values for each vertex.
    """
    # Calculate dihedral angles
    A, _ = adjacency_dihedral_angle_matrix(V, F)
    A = A.toarray()
    
    # Calculate edge lengths for each face
    L1, L2, L3 = edge_lengths(V, F)
    
    # Integrated mean curvature for each vertex
    H = np.zeros(V.shape[0])
    for i, face in enumerate(F):
        for j, v in enumerate(face):
            if j == 0:
                L = L1[i]
                neighbors = [face[1], face[2]]
            elif j == 1:
                L = L2[i]
                neighbors = [face[0], face[2]]
            else:
                L = L3[i]
                neighbors = [face[0], face[1]]

            # Keenan's formula from DDG Slide 21 Lecture16 DiscreteCurvature II
            for neighbor in neighbors:
                H[v] += 0.5 * L * A[i, neighbor]
    
    return H