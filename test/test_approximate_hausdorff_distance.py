import numpy as np
from .context import gpytoolbox
from .context import unittest

class TestApproximateHausdorffDistance(unittest.TestCase):
    def test_bunny(self):
        V,F = gpytoolbox.read_mesh("test/unit_tests_data/bunny_oded.obj")
        # Normalize mesh
        V = gpytoolbox.normalize_points(V,center=np.array([0.5,0.5,0.5]))
        n = gpytoolbox.per_vertex_normals(V,F)
        np.random.seed(0)
        r = np.random.rand()
        u = V + r*n
        g = F.copy()
        dist1 = gpytoolbox.approximate_hausdorff_distance(V,F,u,g,use_cpp=False)
        dist2 = gpytoolbox.approximate_hausdorff_distance(V,F,u,g,use_cpp=True)
        # print(r)
        # print(dist1)
        # print(dist2)
        self.assertTrue(np.isclose(dist1,dist2))
        self.assertTrue(np.isclose(dist1,r))
    def test_bunny_smaller(self):
        # our bunny is too big for the python implementation, so we'll use a smaller one
        V,F = gpytoolbox.read_mesh("test/unit_tests_data/bunny_oded.obj")
        V,F,_,_ = gpytoolbox.decimate(V,F,face_ratio=0.1)
        # Normalize mesh
        V = gpytoolbox.normalize_points(V,center=np.array([0.5,0.5,0.5]))
        n = gpytoolbox.per_vertex_normals(V,F)
        np.random.seed(0)
        for i in range(10):
            r = np.random.rand()
            u = V + r*n
            g = F.copy()
            dist1 = gpytoolbox.approximate_hausdorff_distance(V,F,u,g,use_cpp=False)
            dist2 = gpytoolbox.approximate_hausdorff_distance(V,F,u,g,use_cpp=True)
            # print(r)
            # print(dist1)
            # print(dist2)
            self.assertTrue(np.isclose(dist1,dist2))
            self.assertTrue(np.isclose(dist1,r))

    # It would be nice to have more principled tests here...
    

if __name__ == '__main__':
    unittest.main()
