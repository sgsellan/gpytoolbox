from .context import gpytoolbox as gpy
from .context import numpy as np
from .context import unittest

class TestDiscreteMeanCurvature(unittest.TestCase):
    def test_bunny_mock(self):
        V,F = gpy.read_mesh("test/unit_tests_data/bunny_oded.obj")
        
        result_h = gpy.discrete_mean_curvature(V,F)
        self.assertTrue(False)


if __name__ == '__main__':
    unittest.main()