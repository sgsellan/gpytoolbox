from .context import gpytoolbox as gpy
from .context import numpy as np
from .context import unittest

class TestHalfedgeLengths(unittest.TestCase):

    def test_single_triangle(self):
        v = np.array([[0.0,0.0],[1.0,0.0],[0.0,1.0]])
        f = np.array([[0,1,2]],dtype=int)
        l = gpy.halfedge_lengths(v,f)
        self.assertTrue(np.isclose(l, np.array([[np.sqrt(2.), 1., 1.]])).all())
    
    def test_bunny(self):
        v,f = gpy.read_mesh("test/unit_tests_data/bunny_oded.obj")

        l = gpy.halfedge_lengths(v,f)

        # The bunny's summed squared edge lengths are approximately
        # 0.5*884.4055104366355
        self.assertTrue(np.isclose(np.sum(l), 884.4055104366355))

    def test_2d_regular(self):
        n = 40
        v,f = gpy.regular_square_mesh(n)

        l = gpy.halfedge_lengths(v,f)
        self.assertTrue(np.isclose(np.sum(l), 532.6173157302491))


if __name__ == '__main__':
    unittest.main()