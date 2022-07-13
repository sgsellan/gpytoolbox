from .context import gpytoolbox as gpy
from .context import numpy as np
from .context import unittest

class TestTipAnglesIntrinsic(unittest.TestCase):

    def test_uniform_triangle(self):
        c = np.random.default_rng().random() + 0.1

        l_sq = c * np.array([[1., 1., 1.]])
        f = np.array([[0,1,2]],dtype=int)

        alpha = gpy.tip_angles_intrinsic(l_sq,f)
        alpha_gt = np.array([[np.pi/3., np.pi/3., np.pi/3.]])

        self.assertTrue(np.isclose(alpha, alpha_gt).all())

    def test_uniform_grid(self):
        c = np.random.default_rng().random() + 0.1

        _,f = gpy.regular_square_mesh(400)
        l_sq = c * np.ones(f.shape)

        alpha = gpy.tip_angles_intrinsic(l_sq,f)
        alpha_gt = np.full(f.shape, np.pi/3.)

        self.assertTrue(np.isclose(alpha, alpha_gt).all())

if __name__ == '__main__':
    unittest.main()