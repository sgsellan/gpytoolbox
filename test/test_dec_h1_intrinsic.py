from .context import gpytoolbox as gpy
from .context import numpy as np
from .context import unittest

class TestDecH1Intrinsic(unittest.TestCase):

    def test_uniform_triangle(self):
        l_sq = np.array([[1., 1., 1.]])
        f = np.array([[0,1,2]],dtype=int)

        h1 = gpy.dec_h1_intrinsic(l_sq,f)

        a = 0.5/np.sqrt(3.)
        gt_arr = np.array([[a,0.,0.],[0.,a,0.],[0.,0.,a]])
        self.assertTrue(np.isclose(h1.toarray(), gt_arr).all())

if __name__ == '__main__':
    unittest.main()