from .context import gpytoolbox as gpy
from .context import numpy as np
from .context import unittest

class TestCotangentLaplacianIntrinsic(unittest.TestCase):

    def test_uniform_triangle(self):
        c = np.random.default_rng().random() + 0.1

        l_sq = c * np.array([[1., 1., 1.]])
        f = np.array([[0,1,2]],dtype=int)

        L = gpy.cotangent_laplacian_intrinsic(l_sq,f)
        a = 0.5/np.sqrt(3.)
        L_gt_arr = np.array([[2.*a, -a, -a], [-a, 2.*a, -a], [-a, -a, 2.*a]])
        self.assertTrue(np.isclose(L.toarray(), L_gt_arr).all())

if __name__ == '__main__':
    unittest.main()