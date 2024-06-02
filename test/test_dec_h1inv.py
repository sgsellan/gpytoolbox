from .context import gpytoolbox as gpy
from .context import numpy as np
from .context import unittest

class TestDecH1Inv(unittest.TestCase):

    def test_uniform_triangle(self):
        v = np.array([[0.0,0.0],[1.0,0.0],[0.5,0.5*np.sqrt(3)]])
        f = np.array([[0,1,2]],dtype=int)

        h1inv = gpy.dec_h1inv(v,f)

        a = 1. / (0.5/np.sqrt(3.))
        gt_arr = np.array([[a,0.,0.],[0.,a,0.],[0.,0.,a]])
        self.assertTrue(np.isclose(h1inv.toarray(), gt_arr).all())

if __name__ == '__main__':
    unittest.main()