import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse import find
# Mostly written with chatGPT TBH

def on_boundary(F):
    """
    Determines if each face of a manifold mesh is on the boundary
    (contains at least one boundary edge).

    Parameters:
    F : ndarray
        #F by simplex-size list of element indices

    Returns:
    I : ndarray
        #F list of bools, whether on boundary
    C : ndarray
        #F by simplex size matrix of bools, whether opposite edge on boundary
    """
    simplex_size = F.shape[1]

    if simplex_size == 3:
        E = np.vstack([
            np.column_stack((F[:, 1], F[:, 2])),
            np.column_stack((F[:, 2], F[:, 0])),
            np.column_stack((F[:, 0], F[:, 1]))
        ])
        
        sortedE = np.sort(E, axis=1)

        if np.min(sortedE) > 0:
            _, _, n = np.unique(sortedE[:, 0] + np.max(sortedE[:, 0]) * sortedE[:, 1], return_inverse=True)
        else:
            _, _, n = np.unique(sortedE, axis=0, return_inverse=True)

        counts = np.bincount(n)
        C = counts[n]
        C = C.reshape(F.shape)
        C = (C == 1)
        I = np.any(C, axis=1)

    elif simplex_size == 4:
        T = F
        allF = np.vstack([
            np.column_stack((T[:, 1], T[:, 3], T[:, 2])),
            np.column_stack((T[:, 0], T[:, 2], T[:, 3])),
            np.column_stack((T[:, 0], T[:, 3], T[:, 1])),
            np.column_stack((T[:, 0], T[:, 1], T[:, 2]))
        ])

        sortedF = np.sort(allF, axis=1)
        u, _, n = np.unique(sortedF, axis=0, return_inverse=True)
        counts = np.bincount(n)
        C = counts[n]
        C = C.reshape(T.shape)
        C = (C == 1)
        I = np.any(C, axis=1)

    else:
        raise ValueError(f"Unsupported simplex size {simplex_size}")

    return I, C


def triangle_triangle_adjacency(F):
    """
    Build a face adjacency data structure for a **manifold** triangle mesh.
    From each face, we can find where on its neighboring faces it's incident.

    Parameters:
    F : ndarray
        #F by 3 list of face indices.

    Returns:
    Fp : ndarray
        #F by 3, where Fp[i, j] tells the index of the neighboring triangle
        to the j-th edge of the i-th triangle in F. -1 if the j-th edge is a border edge.
    Fi : ndarray
        #F by 3, where Fi[i, j] tells the position on the neighboring triangle 
        corresponding to the j-th edge of the i-th triangle in F. -1 if it's a border edge.
    """

    num_faces = F.shape[0]

    # Construct edge list (each row is an edge: vertex1, vertex2, face_index, edge_index)
    edges = np.vstack([
        np.column_stack((F[:, 1], F[:, 2], np.arange(num_faces), np.full(num_faces, 0))),  # Edge opposite vertex 0
        np.column_stack((F[:, 2], F[:, 0], np.arange(num_faces), np.full(num_faces, 1))),  # Edge opposite vertex 1
        np.column_stack((F[:, 0], F[:, 1], np.arange(num_faces), np.full(num_faces, 2)))   # Edge opposite vertex 2
    ])

    # Ensure edges are sorted such that (min, max) representation is consistent
    sorted_edges = np.sort(edges[:, :2], axis=1)
    edges[:, :2] = sorted_edges  # Replace with sorted edges

    # Sort edges lexicographically
    sort_indices = np.lexsort((edges[:, 1], edges[:, 0]))
    edges = edges[sort_indices]

    # Find unique edges and their counts
    unique_edges, first_occurrence, edge_map = np.unique(edges[:, :2], axis=0, return_index=True, return_inverse=True)
    edge_counts = np.bincount(edge_map)

    # Identify internal edges (edges that appear exactly twice)
    internal_edge_mask = edge_counts == 2

    # Find indices of matching edges (adjacent faces)
    internal_edges_indices = np.where(internal_edge_mask[edge_map])[0].reshape(-1, 2)
    
    # Extract corresponding faces and edge indices
    f1, v1 = edges[internal_edges_indices[:, 0], 2], edges[internal_edges_indices[:, 0], 3]
    f2, v2 = edges[internal_edges_indices[:, 1], 2], edges[internal_edges_indices[:, 1], 3]

    # Initialize adjacency matrices with -1 (default: boundary edges)
    Fp = -np.ones((num_faces, 3), dtype=int)
    Fi = -np.ones((num_faces, 3), dtype=int)

    # Assign adjacency relationships for internal edges
    Fp[f1, v1] = f2
    Fi[f1, v1] = v2
    Fp[f2, v2] = f1
    Fi[f2, v2] = v1

    return Fp, Fi




def find_ears(F):
    """
    Find all ears (faces with two boundary edges) in a given mesh.

    Parameters:
    F : ndarray
        #F by 3 list of triangle mesh indices.

    Returns:
    ears : ndarray
        Indices into F of ears.
    ear_opp : ndarray
        Indices indicating which edge is non-boundary (connecting to flops).
    flops : ndarray
        Indices into F of faces incident on ears.
    flop_opp : ndarray
        Indices indicating which edge is non-boundary (connecting to ears).
    """

    if False:  # Replace with "if nargout <= 2" behavior, but Python functions return all outputs.
        # Works on non-manifold meshes...
        _, B = on_boundary(F)
        ears = np.where(np.sum(B, axis=1) == 2)[0]
        ear_opp = np.argmin(B[ears, :], axis=1)

        return ears, ear_opp

    else:
        # Must be manifold (actually only ears need to be)
        Fp, Fi = triangle_triangle_adjacency(F)

        # Definition: an ear has 2 boundary **edges**
        ears = np.where(np.sum(Fp == -1, axis=1) == 2)[0]
        flops = np.max(Fp[ears, :], axis=1)

        ear_opp = np.argmax(Fp[ears, :], axis=1)  # Find the edge index
        flop_opp = Fi[ears, ear_opp]
        return ears, ear_opp, flops, flop_opp


def flip_ears(V, F, PlanarOnly=False, PlanarEpsilon=1e-8, FlipAndClip=False):
    """
    Flip ears (triangles with two boundary edges).

    Parameters:
    V : ndarray
        #V by 3 list of mesh vertices.
    F : ndarray
        #F by 3 list of triangle mesh indices.
    PlanarOnly : bool, optional
        Whether to flip only if ears and their flops form a planar quad. Default is False.
    PlanarEpsilon : float, optional
        Epsilon used to determine planarity. Default is 1e-8.
    FlipAndClip : bool, optional
        Whether to flip ears then clip any new ears, then repeat until convergence. Default is False.

    Returns:
    FF : ndarray
        #F by 3 list of new triangle mesh indices.
    """

    FF = F.copy()

    while True:
        F = FF.copy()
        ears, ear_opp, flops, flop_opp = find_ears(F)

        if PlanarOnly:
            # Ensure V is 3D
            if V.shape[1] < 3:
                V = np.hstack((V, np.zeros((V.shape[0], 3 - V.shape[1]))))

            def normalizerow(M):
                return M / np.linalg.norm(M, axis=1, keepdims=True)

            def face_normals(V, F):
                """Compute face normals of triangular mesh."""
                v0 = V[F[:, 0], :]
                v1 = V[F[:, 1], :]
                v2 = V[F[:, 2], :]
                normals = np.cross(v1 - v0, v2 - v0)
                return normalizerow(normals)

            Near = face_normals(V, F[ears, :])
            Nflop = face_normals(V, F[flops, :])

            D = (1 - np.sum(Near * Nflop, axis=1)) < PlanarEpsilon
            # Keep only planar pairs
            ears = ears[D]
            flops = flops[D]
            ear_opp = ear_opp[D]
            flop_opp = flop_opp[D]

        edgeToVS = np.array([[1, 2], [2, 0], [0, 1]]) # vertices in the edge.
        # edge addreses the vertex not on the edge
        # edgeTo

        # Perform edge flipping
        FF = F.copy()
        # replace the first vertex of the bad edge with the vertex not on the other edge
        FF[ears, edgeToVS[ear_opp][:, 0]] = F[flops, flop_opp]
        # on the other edge, take a bad vertex and overwitte with the vertex not on the edge.
        FF[flops, edgeToVS[flop_opp][:, 1]] = F[ears, ear_opp]

        if FlipAndClip:
            ears, _, _, _ = find_ears(FF)
            if len(ears) == 0:
                break
            else:
                FF = np.delete(FF, ears, axis=0)
        else:
            break

    return FF