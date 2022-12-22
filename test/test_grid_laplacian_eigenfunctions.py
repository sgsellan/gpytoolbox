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
    # To-do: test 3D functions