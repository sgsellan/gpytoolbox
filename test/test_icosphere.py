from .context import gpytoolbox as gpy
from .context import numpy as np
from .context import unittest

class TestIcosphere(unittest.TestCase):

    def test_icosphere(self):
        for i in range(5):
            V,F = gpy.icosphere(i)
            self.assertTrue(np.all(np.isclose(np.mean(V,axis=0), 0.)))
            self.assertTrue(np.all(np.isclose(np.linalg.norm(V,axis=-1), 1.)))


if __name__ == '__main__':
    unittest.main()