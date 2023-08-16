from .context import gpytoolbox as gpy
import scipy.io as sio
from .context import numpy as np
from .context import unittest

class TestDiscreteGaussianCurvature(unittest.TestCase):
    def test_bunny(self):
        V,F = gpy.read_mesh("test/unit_tests_data/bunny_oded.obj")
        
        expected_k = sio.loadmat("test/unit_tests_data/bunny_oded_gaussian_curv.mat")["k"].flatten()

        result_k = gpy.discrete_gaussian_curvature(V,F)
        self.assertTrue(np.allclose(result_k, expected_k), msg=f"{result_k} {expected_k}")


if __name__ == '__main__':
    unittest.main()