from .context import gpytoolbox
from .context import unittest
from .context import numpy as np

class TestSquaredDistance(unittest.TestCase):
    def test_find_closest_point_2d_pointcloud(self):
        np.random.seed(0)
        ss = 20
        for ss in range(10,2000,100):       
            P = np.random.rand(ss,2)
            ptest = P[9,:] + 1e-5
            sqrD,ind = gpytoolbox.squared_distance(ptest,P)
            # print(np.sqrt(sqrD))
            self.assertTrue(ind==9)
            self.assertTrue(np.isclose(np.sqrt(sqrD) - np.sqrt(2)*1e-5,0,atol=1e-5))
            sqrD,ind = gpytoolbox.squared_distance(ptest,P,use_aabb=True)
            # print(np.sqrt(sqrD))
            self.assertTrue(ind==9)
            self.assertTrue(np.isclose(np.sqrt(sqrD) - np.sqrt(2)*1e-5,0,atol=1e-5))
    def test_find_closest_point_3d_pointcloud(self):
        np.random.seed(0)
        ss = 20
        for ss in range(10,2000,100):       
            P = np.random.rand(ss,3)
            ptest = P[9,:] + 1e-5
            sqrD,ind = gpytoolbox.squared_distance(ptest,P)
            # print(np.sqrt(sqrD))
            self.assertTrue(ind==9)
            self.assertTrue(np.isclose(np.sqrt(sqrD) - np.sqrt(3)*1e-5,0,atol=1e-5))
            sqrD,ind = gpytoolbox.squared_distance(ptest,P,use_aabb=True)
            # print(np.sqrt(sqrD))
            self.assertTrue(ind==9)
            self.assertTrue(np.isclose(np.sqrt(sqrD) - np.sqrt(3)*1e-5,0,atol=1e-5))
    def test_polygon_synthetic(self):
        # Build a polyline; for example, a square
        V = np.array([ [-1.0, -1.0], [-1.0, 1.0], [1.0, 1.0], [1.0, -1.0] ])
        sample_points = np.array([  [0.0,0.0],
                                    [0.3,0.0],
                                    [-1.5,0.5],
                                    [1.2,0.0]])
        groundtruth_vals = np.array([1.0,0.7,0.5,0.2])**2.0
        E = gpytoolbox.edge_indices(V.shape[0])
        for i in range(sample_points.shape[0]):
            sqrD,ind = gpytoolbox.squared_distance(sample_points[i,:],V,F=E)
            # print(groundtruth_vals[i])
            self.assertTrue(np.isclose(sqrD-groundtruth_vals[i],0).all())
            sqrD,ind = gpytoolbox.squared_distance(sample_points[i,:],V,F=E,use_aabb=True)
            # print(sqrD)
            # print(groundtruth_vals[i])
            self.assertTrue(np.isclose(sqrD-groundtruth_vals[i],0).all())

    def test_polygon_from_image(self):
        filename = "test/unit_tests_data/poly.png"
        poly = gpytoolbox.png2poly(filename)
        V = gpytoolbox.normalize_points(poly[0])
        V = V[0:V.shape[0]:100,:]
        # print(V.shape[0])
        E = gpytoolbox.edge_indices(V.shape[0])
        P = 2*np.random.rand(100,2)-4
        for i in range(P.shape[0]):
            sqrD_gt,ind = gpytoolbox.squared_distance(P[i,:],V,F=E)
            # print(groundtruth_vals[i])
            sqrD_aabb,ind = gpytoolbox.squared_distance(P[i,:],V,F=E,use_aabb=True)
            self.assertTrue(np.isclose(sqrD_aabb-sqrD_gt,0).all())




if __name__ == '__main__':
    unittest.main()
