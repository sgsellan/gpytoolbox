from .context import gpytoolbox
from .context import numpy as np
from .context import unittest

class TestMassmatrix(unittest.TestCase):
    def test_polyline_integral(self):
        # This is a cube, centered at the origin, with side length 1
        # v,f = igl.read_triangle_mesh("test/unit_tests_data/cube.obj")
        #
        # Let's make up a simple polyline
        v = np.array([[0],[0.2],[0.5],[0.98],[1.0]])
        fun_integral_five = 0*v + 5
        fun_integral_other = v + 5
        # integral is 1^3/2 + 5*1
        M = gpytoolbox.massmatrix(v)
        self.assertTrue(np.isclose(np.sum(M @ fun_integral_five),5.))
        self.assertTrue(np.isclose(np.sum(M @ fun_integral_other),5.5))
        

    def test_single_triangle_2d(self):
        v = np.array([[0.0,0.0],[1.0,0.0],[0.0,1.0]])
        f = np.array([[0,1,2]],dtype=int)
        M = gpytoolbox.massmatrix(v,f)
        M_gt = np.array([[0.5,0.0,0.0],[0.0,0.5,0.0],[0.0,0.0,0.5]])/3.
        self.assertTrue(np.isclose(M.toarray() - M_gt,0.0).all())
    
    def test_single_triangle_3d(self):
        v = np.array([[0.0,0.0,0.0],[1.0,0.0,0.0],[0.0,0.0,1.0]])
        f = np.array([[0,1,2]],dtype=int)
        M = gpytoolbox.massmatrix(v,f)
        M_gt = np.array([[0.5,0.0,0.0],[0.0,0.5,0.0],[0.0,0.0,0.5]])/3.
        self.assertTrue(np.isclose(M.toarray() - M_gt,0.0).all())

if __name__ == '__main__':
    unittest.main()