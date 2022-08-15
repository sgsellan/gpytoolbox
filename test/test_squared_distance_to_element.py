from .context import gpytoolbox
from .context import unittest
from .context import numpy as np

class TestSquaredDistanceToElement(unittest.TestCase):
    def test_random_point_pairs_2d(self):
        np.random.seed(0)
        for i in range(200):
            p = np.random.rand(1,2)
            V = np.random.rand(1,2)
            sqrD,_ = gpytoolbox.squared_distance_to_element(p,V,np.array([0]))
            distance_gt = np.linalg.norm(p-V)
            self.assertTrue(np.isclose(np.sqrt(sqrD) - distance_gt,0.0,atol=1e-4))
    def test_random_point_pairs_3d(self):
        np.random.seed(0)
        for i in range(200):
            p = np.random.rand(1,3)
            V = np.random.rand(1,3)
            sqrD,_ = gpytoolbox.squared_distance_to_element(p,V,np.array([0]))
            distance_gt = np.linalg.norm(p-V)
            self.assertTrue(np.isclose(np.sqrt(sqrD) - distance_gt,0.0,atol=1e-4))

        




if __name__ == '__main__':
    unittest.main()
