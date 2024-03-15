from .context import gpytoolbox as gpy
from .context import numpy as np
from .context import unittest

class TestDecH0Intrinsic(unittest.TestCase):

    def test_uniform_triangle(self):
        l_sq = np.array([[1., 1., 1.]])
        f = np.array([[0,1,2]],dtype=int)

        h0 = gpy.dec_h0_intrinsic(l_sq,f)

        a = np.sqrt(3)/12.
        gt_arr = np.array([[a,0.,0.],[0.,a,0.],[0.,0.,a]])
        self.assertTrue(np.isclose(h0.toarray(), gt_arr).all())

if __name__ == '__main__':
    unittest.main()