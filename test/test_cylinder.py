from .context import gpytoolbox as gpy
from .context import numpy as np
from .context import unittest

class TestCylinder(unittest.TestCase):

    def test_cylinder(self):
        for nx in range(3,50,5):
            for nz in range(2,50,5):
                V,F = gpy.cylinder(nx,nz)
                # Correct number of elements
                self.assertTrue(V.shape[0] == nx*nz)
                self.assertTrue(F.shape[0] == 2*nx*(nz-1))
                # Check: vertices are between zero and one
                self.assertTrue(np.max(V[:,2]) <= 1.0)
                self.assertTrue(np.min(V[:,2]) >= 0.)
                # Check: vertices all on circle
                self.assertTrue(np.isclose(np.linalg.norm(V[:,:2], axis=-1), 1.).all())
                # Check: all faces properly oriented
                N = np.cross( V[F[:,1],:] - V[F[:,0],:], V[F[:,2],:] - V[F[:,0],:] , axis=1)
                B = (V[F[:,0],:]+V[F[:,1],:]+V[F[:,2],:]) / 3.
                self.assertTrue((np.sum(N*B, axis=-1)>0).all())


if __name__ == '__main__':
    unittest.main()