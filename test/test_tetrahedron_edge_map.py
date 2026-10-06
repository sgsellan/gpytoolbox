import numpy as np
from gpytoolbox import tetrahedron_edge_map


def test_single_tet():
    T = np.array([[0, 1, 2, 3]])

    E, T_to_E = tetrahedron_edge_map(T)

    assert E.shape == (6, 2)
    assert T_to_E.shape == (1, 6)


def test_two_tet_shared_face():
    T = np.array([
        [0, 1, 2, 3],
        [1, 2, 3, 4],
    ])

    E, T_to_E = tetrahedron_edge_map(T)

    assert E.shape == (9, 2)
    assert T_to_E.shape == (2, 6)

    # Shared edges: (1,2), (1,3), (2,3)
    assert T_to_E[0, 3] == T_to_E[1, 0]
    assert T_to_E[0, 4] == T_to_E[1, 1]
    assert T_to_E[0, 5] == T_to_E[1, 3]