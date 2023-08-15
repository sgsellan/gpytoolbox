from .context import gpytoolbox as gpy
import scipy.io as sio
from .context import numpy as np
from .context import unittest

class TestAdjacencyDihedralAngleMatrix(unittest.TestCase):
    def test_mock(self):
        V,F = gpy.read_mesh("test/unit_tests_data/bunny_oded.obj")

        result_A, result_C = gpy.adjacency_dihedral_angle_matrix(V,F)
        self.assertTrue(False)


if __name__ == '__main__':
    unittest.main()