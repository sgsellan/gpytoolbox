from gpytoolbox.edge_indeces import edge_indeces
from .context import gpytoolbox
from .context import numpy as np
from .context import unittest
import igl

class TestGrad(unittest.TestCase):
    def test_polyline_grad(self):
        # This is a cube, centered at the origin, with side length 1
        # v,f = igl.read_triangle_mesh("test/unit_tests_data/cube.obj")
        #
        # Let's make up a simple polyline
        v = np.array([[0],[0.2],[0.5],[0.98],[1.0]])
        edge_centers = (v[0:4,:] + v[1:5,:])/2.0
        fun_zero_grad = 0*v + 5
        fun_constant_grad = 2*v
        fun_other_grad = v**2.0
        G = gpytoolbox.grad(v)
        # Finite elements should get exact gradients if they are analytically piecewise linear
        self.assertTrue(np.isclose((G @ fun_zero_grad),0.0).all())
        self.assertTrue(np.isclose((G @ fun_constant_grad) - 2.0,0.0).all())
        self.assertTrue(np.isclose((G @ fun_other_grad) - 3.0*edge_centers,0.0).all())





if __name__ == '__main__':
    unittest.main()