from .context import gpytoolbox as gpy
from .context import numpy as np
from .context import unittest

class TestBiharmonicEnergyIntrinsic(unittest.TestCase):

    def test_uniform_triangle(self):
        c = np.random.default_rng().random() + 0.1

        l_sq = c * np.array([[1., 1., 1.]])
        f = np.array([[0,1,2]],dtype=int)

        Q = gpy.biharmonic_energy_intrinsic(l_sq, f, bc='mixedfem_zero_neumann')
        Q_gt = np.sqrt(3)/c * np.array([[2., -1., -1.],
            [-1., 2., -1.],
            [-1., -1., 2.]])
        self.assertTrue(np.isclose(Q.toarray(), Q_gt).all())

        Q = gpy.biharmonic_energy_intrinsic(l_sq, f, bc='curved_hessian')
        Q_gt = np.zeros((3,3))
        self.assertTrue(np.isclose(Q.toarray(), Q_gt).all())

if __name__ == '__main__':
    unittest.main()

