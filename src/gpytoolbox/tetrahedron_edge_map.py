import numpy as np


def tetrahedron_edge_map(T):
    """Computes unique edges of a tetrahedral mesh and a map from each tetrahedron local edges to the unique edge list

    Edge ordering convention is: [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]

    Input
    ----------
    T : (m,4) numpy int array
        m = number of tetrahedra
        4 vertices

    Returns
    -------
    E : (e,2) numpy int array
        Unique unoriented edges of the tetrahedral mesh
    
    T_to_E : (m,6) numpy int array
        For each tetrahedron, index into E corresponding to its six local edges
    """

    assert T.shape[0] > 0
    assert T.shape[1] == 4

    local_edges = np.array([
        [0, 1],
        [0, 2],
        [0, 3],
        [1, 2],
        [1, 3],
        [2, 3],
    ])

    all_edges = T[:, local_edges].reshape(-1, 2)
    all_edges = np.sort(all_edges, axis=1)

    E, inverse = np.unique(
        all_edges,
        axis=0,
        return_inverse=True
    )

    T_to_E = inverse.reshape(T.shape[0], 6)

    return E, T_to_E