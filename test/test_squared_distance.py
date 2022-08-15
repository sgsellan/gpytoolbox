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

        
        




if __name__ == '__main__':
    unittest.main()
