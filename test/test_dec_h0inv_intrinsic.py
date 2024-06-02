from .context import gpytoolbox as gpy
from .context import numpy as np
from .context import unittest

class TestDecH0InvIntrinsic(unittest.TestCase):

    def test_uniform_triangle(self):
        l_sq = np.array([[1., 1., 1.]])
        f = np.array([[0,1,2]],dtype=int)

        h0inv = gpy.dec_h0inv_intrinsic(l_sq,f)

        a = 1. / (np.sqrt(3)/12.)
        gt_arr = np.array([[a,0.,0.],[0.,a,0.],[0.,0.,a]])
        self.assertTrue(np.isclose(h0inv.toarray(), gt_arr).all())

if __name__ == '__main__':
    unittest.main()