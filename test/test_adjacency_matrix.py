from .context import gpytoolbox as gpy
from scipy.sparse import csc_matrix
from .context import numpy as np
from .context import unittest

class TestAdjacencyMatrix(unittest.TestCase):
    def test_basic(self):
        F = np.array([
            [0, 1, 2],
            [2, 3, 4],
            [1, 2, 5]
        ])

        expected = csc_matrix([
            [0, 1, 1, 0, 0, 0],
            [1, 0, 1, 0, 0, 1],
            [1, 1, 0, 1, 1, 1],
            [0, 0, 1, 0, 1, 0],
            [0, 0, 1, 1, 0, 0],
            [0, 1, 1, 0, 0, 0]
        ])

        result = gpy.adjacency_matrix(F)

        self.assertTrue((result.toarray() == expected.toarray()).all())


if __name__ == '__main__':
    unittest.main()
