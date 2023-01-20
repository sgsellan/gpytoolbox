import numpy as np
from .context import gpytoolbox
from .context import unittest

class TestHausdorffDistance(unittest.TestCase):
    def test_cube(self):
        V,F = gpytoolbox.read_mesh("test/unit_tests_data/cube.obj")
        # Normalize mesh
        V = gpytoolbox.normalize_points(V,center=np.array([0.5,0.5,0.5]))
        U,G = gpytoolbox.read_mesh("test/unit_tests_data/cube.obj")
        # Normalize mesh
        U = gpytoolbox.normalize_points(U,center=np.array([0.5,0.5,0.5]))
        random_displacements = 6*np.random.rand(20)
        for i in range(20):
            U2 = U.copy()
            U2[:,0] += random_displacements[i]
            dist = gpytoolbox.hausdorff_distance(V,F,U2,G)
            # self.assertTrue(np.isclose(dist,0.0,atol=1e-2))
            dist_gt = np.clip(random_displacements[i]-1,0,np.Inf)
            # print(dist_gt,dist)
            self.assertTrue(np.isclose(dist,dist_gt,atol=1e-4))
    # It would be nice to have more principled tests here...
    

if __name__ == '__main__':
    unittest.main()
