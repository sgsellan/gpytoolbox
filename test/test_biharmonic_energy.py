from .context import gpytoolbox as gpy
from .context import numpy as np
from .context import unittest
import scipy as sp

class TestCotangentLaplacian(unittest.TestCase):

    def test_single_triangle_2d(self):
        v = np.array([[0.0,0.0],[1.0,0.0],[0.0,1.0]])
        f = np.array([[0,1,2]],dtype=int)

        Q = gpy.biharmonic_energy(v, f, bc='mixedfem_zero_neumann')
        Q_gt = np.array([[8., -4., -4.],
            [-4., 3., 1.],
            [-4., 1., 3.]])
        self.assertTrue(np.isclose(Q.toarray(), Q_gt).all())

        Q = gpy.biharmonic_energy(v, f, bc='hessian')
        Q_gt = np.zeros((3,3))
        self.assertTrue(np.isclose(Q.toarray(), Q_gt).all())

        Q = gpy.biharmonic_energy(v, f, bc='curved_hessian')
        Q_gt = np.zeros((3,3))
        self.assertTrue(np.isclose(Q.toarray(), Q_gt).all())
    

if __name__ == '__main__':
    unittest.main()