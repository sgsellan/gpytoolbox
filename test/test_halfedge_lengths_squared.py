from .context import gpytoolbox as gpy
from .context import numpy as np
from .context import unittest

class TestHalfedgeLengthsSquared(unittest.TestCase):

    def test_single_triangle(self):
        v = np.array([[0.0,0.0],[1.0,0.0],[0.0,1.0]])
        f = np.array([[0,1,2]],dtype=int)
        l_sq = gpy.halfedge_lengths_squared(v,f)
        self.assertTrue(np.isclose(l_sq, np.array([[2., 1., 1.]])).all())
    
    def test_bunny(self):
        v,f = gpy.read_mesh("test/unit_tests_data/bunny_oded.obj")

        l_sq = gpy.halfedge_lengths_squared(v,f)

        # The bunny's summed squared edge lengths are approximately
        # 0.5*39.74617451913487
        self.assertTrue(np.isclose(np.sum(l_sq), 39.74617451913487))

    def test_2d_regular(self):
        n = 40
        v,f = gpy.regular_square_mesh(n)

        l_sq = gpy.halfedge_lengths_squared(v,f)
        self.assertTrue(np.isclose(np.sum(l_sq), 32.))


if __name__ == '__main__':
    unittest.main()