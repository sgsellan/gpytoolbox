from .context import gpytoolbox
from .context import unittest
from .context import numpy as np

# This test is basically the same as fd_partial_derivative

class TestFdHessian(unittest.TestCase):
    def test_analytic_2d(self):
        # Choose grid size
        gs = np.array([19, 28])
        h = 1.0 / (gs - 1)

        interior_indices = np.array([
            i * gs[0] + j
            for i in range(1, gs[1] - 1)
            for j in range(1, gs[0] - 1)
        ])

        # Build a grid
        x, y = np.meshgrid(np.linspace(0, 1, gs[0]), np.linspace(0, 1, gs[1]))
        V = np.concatenate((np.reshape(x, (-1, 1)), np.reshape(y, (-1, 1))), axis=1)

        # Compute the Hessian
        H = gpytoolbox.fd_hessian(gs=gs, h=h)

        # Test function
        f = V[:, 0]**2 + V[:, 1]**2
        computed_hessian = H @ f

        # Extract Dxx results
        computed_derivative_xx = computed_hessian[:gs[0] * gs[1]]
        computed_derivative_xx = computed_derivative_xx[interior_indices]
        computed_derivative_xy = computed_hessian[gs[0] * gs[1]:2 * gs[0] * gs[1]]
        computed_derivative_xy = computed_derivative_xy[interior_indices]
        computed_derivative_yx = computed_hessian[2 * gs[0] * gs[1]:3 * gs[0] * gs[1]]
        computed_derivative_yx = computed_derivative_yx[interior_indices]
        computed_derivative_yy = computed_hessian[3 * gs[0] * gs[1]:]
        computed_derivative_yy = computed_derivative_yy[interior_indices]

        # Should be 2.0 everywhere
        self.assertTrue(np.allclose(computed_derivative_xx,2.0))
        self.assertTrue(np.allclose(computed_derivative_yy,2.0))
        self.assertTrue(np.allclose(computed_derivative_xy,0.0))



if __name__ == '__main__':
    unittest.main()