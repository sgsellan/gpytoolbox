from .context import gpytoolbox
from .context import numpy as np
from .context import unittest


class TestLinearElasticity(unittest.TestCase):
    def test_no_external_forces(self):
        # Build very small mesh
        V, F = gpytoolbox.regular_square_mesh(3)
        V = (V + 1.)/2.
        # Initial conditions
        fext = 0*V
        Ud0 = 0*V
        U0 = V.copy()
        U0[:,1] = 0.0
        U0[:,0] = U0[:,0] - 0.5
        # print(np.reshape(U0,(-1,1),order='F'))
        dt = 0.2
        U, sigma_v = gpytoolbox.linear_elasticity(V,F,U0,fext=fext,dt=dt,Ud0=Ud0)
        Ud0 = (np.reshape(U,(-1,2),order='F') - U0)/dt
        U0 = np.reshape(U,(-1,2),order='F')
        # Groundtruth obtained using Matlab's linear elasticity gptoolbox function
        U_groundtruth = np.array([  [-0.3660,    0.1304],
                                    [0.0000,     0.1304],
                                    [0.3660,     0.1304],
                                    [-0.3660,   -0.0000],
                                    [0.0000,    -0.0000],
                                    [0.3660,     0.0000],
                                    [-0.3660,   -0.1304],
                                    [-0.0000,   -0.1304],
                                    [0.3660,   -0.1304]])

        # Check python output matches Matlab groundtruth
        self.assertTrue(np.isclose(np.reshape(U,(-1,2),order='F'),U_groundtruth,atol=1e-4).all())

        # Check another iteration, with non-zero velocity
        U, sigma_v = gpytoolbox.linear_elasticity(V,F,U0,fext=fext,dt=dt,Ud0=Ud0)
        Ud0 = (np.reshape(U,(-1,2),order='F') - U0)/dt
        U0 = np.reshape(U,(-1,2),order='F')
        # Groundtruth computed with gptoolbox's function
        U_groundtruth = np.array([  [-0.2378,    0.2513],
                                    [0.0000,    0.2513],
                                    [0.2378,    0.2513],
                                    [-0.2378,   -0.0000],
                                    [0.0000,   -0.0000],
                                    [0.2378,    0.0000],
                                    [-0.2378,   -0.2513],
                                    [-0.0000,   -0.2513],
                                    [0.2378 ,  -0.2513]])
        # Check python output matches Matlab groundtruth
        self.assertTrue(np.isclose(np.reshape(U,(-1,2),order='F'),U_groundtruth,atol=1e-4).all())

if __name__ == '__main__':
    unittest.main()