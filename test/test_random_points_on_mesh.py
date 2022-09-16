from .context import gpytoolbox as gpy
from .context import unittest
from .context import numpy as np


class TestRandomPointsOnMesh(unittest.TestCase):
    def test_is_uniform(self):

        # Do not error on zero input
        x = gpy.random_points_on_mesh(np.array([]), np.array([], dtype=int), n=0)
        self.assertTrue(len(x)==0)

        # Test 1: Straight line
        V = np.array([[0.0,0.0],[1.0,1.0],[2.0,2.0]])
        E = gpy.edge_indices(V.shape[0],closed=False)
        n = 100000
        rng = np.random.default_rng(5)
        x = gpy.random_points_on_mesh(V, E, n, rng=rng)
        rng = np.random.default_rng(5)
        y,I,u = gpy.random_points_on_mesh(V, E, n, rng=rng, return_indices=True)

        self.check_consistency(V, E, [x,y], I, u)
        for d in range(2):
            hist, bin_edges = np.histogram(x[:,d],bins=20, density=True)
            self.assertTrue(np.std(hist)<0.01)
            self.assertTrue(np.isclose(np.mean(hist),0.5,atol=1e-3))
            self.assertTrue(np.isclose(np.mean(x[:,d]),1.,atol=1e-2))

        # Sample grid.
        V,F = gpy.regular_square_mesh(30)
        n = 100000
        rng = np.random.default_rng(526)
        x = gpy.random_points_on_mesh(V, F, n, rng=rng)
        rng = np.random.default_rng(526)
        y,I,u = gpy.random_points_on_mesh(V, F, n, rng=rng, return_indices=True)

        self.check_consistency(V, F, [x,y], I, u)
        for d in range(2):
            hist, bin_edges = np.histogram(x[:,d],bins=20, density=True)
            self.assertTrue(np.std(hist)<0.01)
            self.assertTrue(np.isclose(np.mean(hist),0.5,atol=1e-3))
            self.assertTrue(np.isclose(np.mean(x[:,d]),0.,atol=1e-2))

        # Sample cube. This time, no histogram test, just mean of output
        V,F = gpy.read_mesh("test/unit_tests_data/cube.obj")
        n = 100000
        rng = np.random.default_rng(80)
        x = gpy.random_points_on_mesh(V, F, n, rng=rng)
        rng = np.random.default_rng(80)
        y,I,u = gpy.random_points_on_mesh(V, F, n, rng=rng, return_indices=True)

        self.check_consistency(V, F, [x,y], I, u)
        for d in range(3):
            self.assertTrue(np.isclose(np.mean(x[:,d]),0.,atol=1e-2))


    def check_consistency(self, V, F, xs, I=None, u=None):
        if len(xs)<1:
            return None
        for x in xs:
            self.assertTrue(np.isclose(x,xs[0]).all())
        if I is not None and u is not None:
            y = 0.
            for d in range(F.shape[1]):
                y = y + u[:,d][:,None]*V[F[I,d],:]
            self.assertTrue(np.isclose(y,xs[0]).all())


if __name__ == '__main__':
    unittest.main()
