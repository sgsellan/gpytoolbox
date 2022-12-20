from .context import gpytoolbox
from .context import numpy as np
from .context import unittest
import matplotlib.pyplot as plt


class TestPoissonSurfaceReconstruction(unittest.TestCase):
    def test_indicator(self):
        np.random.seed(0)
        # First test: "uniform" sampling density
        # Sample points on a circle
        th = 2*np.pi*np.random.rand(80,1)
        P = np.concatenate((np.cos(th) + 0.1,np.sin(th) + 0.2),axis=1)
        # Normals are the same as positions on a circle
        N = np.concatenate((np.cos(th),np.sin(th)),axis=1)

        # corner = np.array([-1.5,-1.5])
        gs = np.array([80,80])
        # h = np.array([0.05,0.05])

        scalar_mean, scalar_var = gpytoolbox.poisson_surface_reconstruction(P,N,gs=gs,solve_subspace_dim=1000)

        # Plot mean and variance side by side with colormap
        fig, ax = plt.subplots(1,2)
        ax[0].imshow(scalar_mean.reshape(gs,order='F'))
        ax[0].set_title('Mean')
        # Add colorbar
        fig.colorbar(ax[0].imshow(scalar_mean.reshape(gs,order='F')), ax=ax[0])
        ax[1].imshow(scalar_var.reshape(gs,order='F'))
        ax[1].set_title('Variance')
        # Add colorbar
        fig.colorbar(ax[1].imshow(scalar_var.reshape(gs,order='F')), ax=ax[1])
        plt.show()
        

if __name__ == '__main__':
    unittest.main()