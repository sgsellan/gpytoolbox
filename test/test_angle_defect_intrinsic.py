from .context import gpytoolbox as gpy
from .context import numpy as np
from .context import unittest

class TestAngleDefectIntrinsic(unittest.TestCase):

    def test_uniform_triangle(self):
        l_sq = 3.4 * np.array([[1., 1., 1.]])
        f = np.array([[0,1,2]],dtype=int)

        k = gpy.angle_defect_intrinsic(l_sq,f)

        self.assertTrue(np.isclose(k,
            np.array([0., 0., 0.])).all())


    def test_uniform_triangles(self):
        l_sq = 6.7 * np.array([[1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.]])
        f = np.array([[0,1,2],[1,3,2],[3,0,1]],dtype=int)

        k = gpy.angle_defect_intrinsic(l_sq,f)

        self.assertTrue(np.isclose(k,
            np.array([0., np.pi, 0., 0.])).all())

if __name__ == '__main__':
    unittest.main()