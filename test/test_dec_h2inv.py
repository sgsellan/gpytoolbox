from .context import gpytoolbox as gpy
from .context import numpy as np
from .context import unittest

class TestDecH2Inv(unittest.TestCase):

    def test_uniform_triangle(self):
        v = np.array([[0.0,0.0],[1.0,0.0],[0.5,0.5*np.sqrt(3)]])
        f = np.array([[0,1,2]],dtype=int)

        h2inv = gpy.dec_h2inv(v,f)

        a = np.sqrt(3.)/4
        gt_arr = 1. / np.array([[a]])
        self.assertTrue(np.isclose(h2inv.toarray(), gt_arr).all())

if __name__ == '__main__':
    unittest.main()