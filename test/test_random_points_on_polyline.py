from .context import gpytoolbox
from .context import unittest
from .context import numpy as np


class TestRandomPointsOnPolyline(unittest.TestCase):
    def test_is_uniform(self):
        np.random.seed(0)
        # Test 1: Histogram of uniform distribution
        V = np.array([[0.0,0.0],[1.0,1.2],[2.0,2.4]])
        P, N = gpytoolbox.random_points_on_polyline(V,10)

        self.assertTrue(np.isclose(N,np.tile(np.array([[-0.76822128,  0.6401844]]),(10,1))).all())

        P, N = gpytoolbox.random_points_on_polyline(V,200000)
        hist, bin_edges = np.histogram(P[:,0],bins=20, density=True)
        self.assertTrue(np.std(hist)<0.01)
        self.assertTrue(np.isclose(np.mean(hist),0.5,atol=1e-3))


        # Test 2: Very different edge lengths, should still be uniform
        V = np.array([[0.0,0.0],[0.05,0.05],[0.95,0.95],[1.0,1.0]])
        P, N = gpytoolbox.random_points_on_polyline(V,200000)
        hist, bin_edges = np.histogram(P[:,0],bins=20, density=True)
        self.assertTrue(np.std(hist)<0.02)
        self.assertTrue(np.isclose(np.mean(hist),1.0,atol=1e-3))

    def test_are_normals(self):
        # Test 3: Use circle to check that the normals work
        th = np.reshape(np.linspace(0.0,2*np.pi,15000),(-1,1))
        V = np.concatenate((-np.cos(th) + 0.1,np.sin(th) + 0.2),axis=1)
        P, N = gpytoolbox.random_points_on_polyline(V,40)
        self.assertTrue(np.isclose(P - np.tile(np.array([[0.1,0.2]]),(40,1)),N,atol=1e-3).all())

if __name__ == '__main__':
    unittest.main()
