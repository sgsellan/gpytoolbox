from .context import gpytoolbox
from .context import numpy as np
from .context import scipy as sp
from .context import unittest
from .context import gpytoolbox as gpy


class TestGradIntrinsic(unittest.TestCase):

    def test_single_triangle_2d(self):
        c = np.random.default_rng().random() + 0.1

        l_sq = c * np.array([[1., 1., 1.]])
        f = np.array([[0,1,2]],dtype=int)

        G = gpy.grad_intrinsic(l_sq, f)

        G_gt = np.array([[0., -1./np.sqrt(c), 1./np.sqrt(c)],
            [2./np.sqrt(3*c), -1./np.sqrt(3*c), -1./np.sqrt(3*c)]])

        self.assertTrue(np.isclose(G.toarray(), G_gt).all())


    def test_compare_with_cotangent_laplacian(self):
        meshes = ['cube.obj', 'mountain.obj', 'armadillo.obj']
        for mesh in meshes:
            V,F = gpy.read_mesh("test/unit_tests_data/" + mesh)
            m = F.shape[0]
            l_sq = gpy.halfedge_lengths_squared(V,F)
            G = gpy.grad_intrinsic(l_sq, F)
            a = gpy.doublearea_intrinsic(l_sq, F) / 2.
            A = sp.sparse.spdiags([np.tile(a, 2)], 0, m=2*m, n=2*m, format='csr')
            L = G.transpose() * A * G

            L_gt = gpy.cotangent_laplacian(V,F)

            self.assertTrue(np.isclose(L[L_gt.nonzero()], L_gt[L_gt.nonzero()]).all())


    def test_example_compared_with_known_gt(self):
        l_sq = np.array([[2.,            1.,            1.],
            [1.,            1.,            2.],
            [2.,            1.,            1.],
            [1.,            1.,            2.],
            [2.,            1.,            1.],
            [1.,            1.,            2.],
            [2.,            1.,            1.],
            [1.,            1.,            2.],
            [2.,            1.,            1.],
            [1.,            1.,            2.],
            [2.,            1.,            1.],
            [1.,            1.,            2.]])
        F = np.array([[1,     2,     3],
            [3,     2,     4],
            [3,     4,     5],
            [5,     4,     6],
            [5,     6,     7],
            [7,     6,     8],
            [7,     8,     1],
            [1,     8,     2],
            [2,     8,     4],
            [4,     8,     6],
            [7,     1,     5],
            [5,     1,     3]]) - 1
        G_gt = np.array([[0,     -0.70711,      0.70711,            0,            0,            0,            0,            0],
            [0,           -1,            0,            1,            0,            0,            0,            0],
            [0,            0,            0,     -0.70711,      0.70711,            0,            0,            0],
            [0,            0,            0,           -1,            0,            1,            0,            0],
            [0,            0,            0,            0,            0,     -0.70711,      0.70711,            0],
            [0,            0,            0,            0,            0,           -1,            0,            1],
      [0.70711,            0,            0,            0,            0,            0,            0,     -0.70711],
            [0,            1,            0,            0,            0,            0,            0,           -1],
            [0,            0,            0,      0.70711,            0,            0,            0,     -0.70711],
            [0,            0,            0,            0,            0,            1,            0,           -1],
     [-0.70711,            0,            0,            0,      0.70711,            0,            0,            0],
           [-1,            0,            1,            0,            0,            0,            0,            0],
       [1.4142,     -0.70711,     -0.70711,            0,            0,            0,            0,            0],
            [0,   2.2204e-16,            1,           -1,            0,            0,            0,            0],
            [0,            0,       1.4142,     -0.70711,     -0.70711,            0,            0,            0],
            [0,            0,            0,   2.2204e-16,            1,           -1,            0,            0],
            [0,            0,            0,            0,       1.4142,     -0.70711,     -0.70711,            0],
            [0,            0,            0,            0,            0,   2.2204e-16,            1,           -1],
     [-0.70711,            0,            0,            0,            0,            0,       1.4142,     -0.70711],
            [1,           -1,            0,            0,            0,            0,            0,   2.2204e-16],
            [0,       1.4142,            0,     -0.70711,            0,            0,            0,     -0.70711],
            [0,            0,            0,            1,            0,           -1,            0,   2.2204e-16],
     [-0.70711,            0,            0,            0,     -0.70711,            0,       1.4142,            0],
   [2.2204e-16,            0,           -1,            0,            1,            0,            0,            0]])

        G = gpy.grad_intrinsic(l_sq, F)

        self.assertTrue(np.isclose(G.toarray(), G_gt).all())
    

if __name__ == '__main__':
    unittest.main()