from .context import gpytoolbox as gpy
from .context import numpy as np
from .context import unittest

class TestDoubleareaIntrinsic(unittest.TestCase):
    def test_equilateral_triangle(self):
        c = np.random.default_rng().random() + 0.1

        l_sq = c * np.array([[1.,1.,1.]])
        f = np.array([[0,1,2]])



        A = gpy.doublearea_intrinsic(l_sq,f)
        self.assertTrue(np.isclose(A[0], 2.*np.sqrt(3.)/4. * c))

    def test_many_equilateral_triangles(self):
        c = np.random.default_rng().random() + 0.1

        _,f = gpy.regular_square_mesh(40)
        l_sq = c * np.ones(f.shape)

        A = gpy.doublearea_intrinsic(l_sq,f)
        self.assertTrue(np.isclose(A, 2.*np.full(f.shape[0],np.sqrt(3.)/4. * c)).all())

        


if __name__ == '__main__':
    unittest.main()