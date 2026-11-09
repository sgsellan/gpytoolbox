from .context import gpytoolbox as gpy
from .context import numpy as np
from .context import unittest
import itertools

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

    def test_very_tiny_triangle(self):
        c = np.random.default_rng().random() + 0.1

        l_sq = c * np.array([[0.0000001,0.0000001,1.]])
        f = np.array([[0,1,2]])


        # this returns NaN as of v0.2.0
        A = gpy.doublearea_intrinsic(l_sq,f)
        # it shouldn't anymore
        self.assertTrue(np.isnan(A[0]) == False)

    def test_thin_right_triangle(self):
        # Legs 1 and eps, so twice the area is exactly eps.
        f = np.array([[0,1,2]])
        for eps in [1e-2, 1e-4, 1e-6, 1e-8]:
            for l_sq in itertools.permutations([1., eps**2, 1.+eps**2]):
                A = gpy.doublearea_intrinsic(np.array([l_sq]),f)
                self.assertTrue(np.isclose(A[0], eps, rtol=1e-14, atol=0.))

    def test_thin_isosceles_triangle(self):
        # Legs 1 and base eps, so the height is sqrt(1 - eps^2/4).
        f = np.array([[0,1,2]])
        for eps in [1e-2, 1e-4, 1e-6, 1e-8]:
            dblA = eps * np.sqrt(1. - eps**2/4.)
            for l_sq in itertools.permutations([1., 1., eps**2]):
                A = gpy.doublearea_intrinsic(np.array([l_sq]),f)
                self.assertTrue(np.isclose(A[0], dblA, rtol=1e-14, atol=0.))

    def test_independent_of_edge_order(self):
        rng = np.random.default_rng(2)
        V = rng.random((300,2))
        V[:,1] *= 1e-5
        F = np.arange(V.shape[0]).reshape(-1,3)
        l_sq = gpy.halfedge_lengths_squared(V,F)
        A = gpy.doublearea_intrinsic(l_sq,F)
        for p in itertools.permutations(range(3)):
            self.assertTrue(np.all(
                gpy.doublearea_intrinsic(l_sq[:,p],F) == A))


if __name__ == '__main__':
    unittest.main()