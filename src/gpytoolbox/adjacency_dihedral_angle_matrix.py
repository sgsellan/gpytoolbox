import numpy as np
from scipy import sparse

def normalizerow(N):
    """Normalize the rows of matrix N."""
    return (N.T / np.linalg.norm(N, axis=1)).T

def normals(V, F):
    """Compute normals for faces defined by V and F."""
    fn = np.cross(V[F[:, 1], :] - V[F[:, 0], :], V[F[:, 2], :] - V[F[:, 0], :])
    return normalizerow(fn)

def adjacency_dihedral_angle_matrix(V, F):
    """
    Compute a matrix A such that A(i,j) = θij
    the dihedral angle between faces F[i,:] and F[j,:] if they're neighbors (share an edge) or 0 if they're not.

    Parameters
    ----------
    V : ndarray
        N x 3 array of vertex positions.
    F : ndarray
        M x 3 array of face indices.

    Returns
    -------
    A : scipy sparse.coo_matrix
        M x M sparse matrix of signed dihedral angles. All entries are in [0,2*pi]:
        A-->0 as edge is more convex, A-->pi as edge is more flat, A-->2*pi as edge is more concave

    C: scipy sparse.csr_matrix
        M x M sparse matrix. C(i,j) = c that face j is incident on face i across its corner c.
    """

    # Create edges and associated faces
    edge_to_face = {}
    edge_to_corner = {}

    # Create all edges and store face and corner information
    for i, face in enumerate(F):
        for j, e in enumerate([(face[0], face[1]), (face[1], face[2]), (face[0], face[2])]):
            edge = tuple(sorted(e))
            if edge not in edge_to_face:
                edge_to_face[edge] = []
                edge_to_corner[edge] = []
            edge_to_face[edge].append(i)
            edge_to_corner[edge].append(j)

    # Compute normals for each face
    N = normals(V, F)

    # Compute adjacency dihedral angle matrix
    rows_a, cols_a, data_a = [], [], []
    rows_c, cols_c, data_c = [], [], []

    for edge, faces in edge_to_face.items():
        if len(faces) > 1:
            f1, f2 = faces
            n1, n2 = N[f1], N[f2]

            # Calculate the dihedral angle
            v = V[edge[1]] - V[edge[0]]
            v = v / np.linalg.norm(v)
            angle = np.pi - np.arctan2(np.dot(np.cross(n1, n2), v), np.dot(n1, n2))

            rows_a.extend([f1, f2])
            cols_a.extend([f2, f1])
            data_a.extend([angle, angle])

            # Store corner data for C matrix
            c1, c2 = edge_to_corner[edge]
            rows_c.extend([f1, f2])
            cols_c.extend([f2, f1])
            data_c.extend([c2, c1])

    A = sparse.coo_matrix((data_a, (rows_a, cols_a)), shape=(len(F), len(F)))
    C = sparse.coo_matrix((data_c, (rows_c, cols_c)), shape=(len(F), len(F))).toarray()

    return A, C