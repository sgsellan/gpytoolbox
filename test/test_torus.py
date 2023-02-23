from .context import gpytoolbox as gpy
from .context import numpy as np
from .context import unittest

class TestTorus(unittest.TestCase):

    def test_torus(self):
        rng = np.random.default_rng()

        for nR in range(3,50,5):
            for nr in range(3,50,5):
                R = rng.random()
                r = rng.random() * R
                V,F = gpy.torus(nR, nr, R=R, r=r)

                # Correct number of elements
                self.assertTrue(V.shape[0] == nR*nr)
                self.assertTrue(F.shape[0] == 2*nR*nr)

                # Elements have correct plane position and z-coords
                ε = np.finfo(V.dtype).eps
                self.assertTrue((np.linalg.norm(V[:,:2], axis=-1) <= R+r+ε).all())
                self.assertTrue((np.linalg.norm(V[:,:2], axis=-1) >= R-r-ε).all())
                self.assertTrue((V[:,2] <= r+ε).all())
                self.assertTrue((V[:,2] >= -r-ε).all())

if __name__ == '__main__':
    unittest.main()