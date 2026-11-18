from .context import gpytoolbox
from .context import numpy as np
from .context import unittest
import matplotlib.pyplot as plt
import scipy.sparse

class TestGridLaplacianEigenfunctions(unittest.TestCase):
    def test_2d_functions(self):
        # Let's make a fine grid to make sure we get the right eigenfunctions
        gs = np.array([100,100])
        num_modes = 10
        # corner = np.array([-1,-1])
        h = np.array([0.01,0.01])
        G = gpytoolbox.fd_grad(gs,h)
        grid_length = gs*h
        # Get the eigenfunctions
        eigenvecs = gpytoolbox.grid_laplacian_eigenfunctions(num_modes,gs,grid_length)
        # print("grid length: ", grid_length)

        # Mass matrix on the grid... or at least *one* mass matrix on the grid, definitely not the smartest one to use since we're mixing grid and mesh stuff but it's enough for testing
        v,f = gpytoolbox.regular_square_mesh(gs[0])
        v = 0.5*v # grid length should be one
        M = gpytoolbox.massmatrix(v,f,type='voronoi')
        L = gpytoolbox.cotangent_laplacian(v,f)
        # print(eigenvecs[:,0])
        # print(eigenvecs[:,4])
        # Make sure the eigenfunctions form an orthogonal basis
        self.assertTrue(np.allclose(eigenvecs.T @ M @ eigenvecs - scipy.sparse.diags(np.diag(eigenvecs.T @ M @ eigenvecs)), 0.0))
        # Make sure the eigenfunctions are actually eigenfunctions of L
        self.assertTrue(np.allclose(eigenvecs.T @ L @ eigenvecs - scipy.sparse.diags(np.diag(eigenvecs.T @ L @ eigenvecs)), 0.0))
    def test_3d_functions(self):
        gs = np.array([10, 9, 8])
        eigenvecs = gpytoolbox.grid_laplacian_eigenfunctions(10, gs, np.ones(3))
        weights = [np.ones(n) for n in gs]
        for w in weights:
            w[[0, -1]] = 0.5
        wx, wy, wz = np.meshgrid(*weights, indexing='ij')
        mass = (wx * wy * wz).ravel(order='F')
        gram = eigenvecs.T @ (mass[:, None] * eigenvecs)
        np.testing.assert_allclose(gram - np.diag(np.diag(gram)), 0, atol=1e-12)
        self.assertTrue(np.all(np.diag(gram) > 0))

    def test_3d_lowest_modes(self):
        gs = np.array([12, 9, 8])
        lengths = np.array([20., 2., 1.])
        gx, gy, gz = np.meshgrid(
            *[np.linspace(0, length, size) for length, size in zip(lengths, gs)],
            indexing='ij')
        x = gx.ravel(order='F')
        expected = np.cos(np.pi * x[:, None] * np.arange(8) / lengths[0])
        actual = gpytoolbox.grid_laplacian_eigenfunctions(8, gs, lengths)
        np.testing.assert_allclose(actual, expected, atol=1e-14)

    def test_3d_two_modes(self):
        gs = np.array([5, 6, 7])
        lengths = np.array([1., 1., 2.])
        eigenvecs = gpytoolbox.grid_laplacian_eigenfunctions(2, gs, lengths)
        z = np.repeat(np.linspace(0, lengths[2], gs[2]), gs[0] * gs[1])
        np.testing.assert_allclose(eigenvecs[:, 0], 1)
        np.testing.assert_allclose(eigenvecs[:, 1], np.cos(np.pi * z / lengths[2]))
