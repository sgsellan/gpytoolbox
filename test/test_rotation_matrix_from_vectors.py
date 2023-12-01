from .context import gpytoolbox as gpy
from .context import numpy as np
from .context import unittest

class TestRotationMatrixFromVectors(unittest.TestCase):
    def test_randomly_generated(self):
        np.random.seed(0)
        for i in range(10000):
            u = np.random.randn(3)
            # normalize
            v = np.random.randn(3)
            R = gpy.rotation_matrix_from_vectors(u, v)
            # check that they are aligned:
            are_aligned = np.cross(v, R.dot(u))
            self.assertTrue(np.allclose(are_aligned, np.zeros(3), atol=1e-12))
            # normalize
            v = v/np.linalg.norm(v)
            u = u/np.linalg.norm(u)
            R = gpy.rotation_matrix_from_vectors(u, v)
            self.assertTrue(np.allclose(R.dot(u), v, atol=1e-12))


if __name__ == '__main__':
    unittest.main()