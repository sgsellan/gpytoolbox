from .context import gpytoolbox as gpy
from .context import numpy as np
from .context import unittest

class TestCotangentWeightsIntrinsic(unittest.TestCase):

    def test_uniform_triangle(self):
        c = np.random.default_rng().random() + 0.1

        l_sq = c * np.array([[1., 1., 1.]])
        f = np.array([[0,1,2]],dtype=int)

        C = gpy.cotangent_weights_intrinsic(l_sq,f)
        C_gt = np.array([[0.5/np.sqrt(3.), 0.5/np.sqrt(3.), 0.5/np.sqrt(3.)]])

        self.assertTrue(np.isclose(C, C_gt).all())

    def test_uniform_grid(self):
        c = np.random.default_rng().random() + 0.1

        _,f = gpy.regular_square_mesh(400)
        l_sq = c * np.ones(f.shape)

        C = gpy.cotangent_weights_intrinsic(l_sq,f)
        C_gt = np.full(f.shape, 0.5/np.sqrt(3.))

        self.assertTrue(np.isclose(C, C_gt).all())

if __name__ == '__main__':
    unittest.main()