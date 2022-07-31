from .context import gpytoolbox as gpy
from .context import numpy as np
from .context import unittest
import numpy as np
import scipy as sp


class TestMassmatrixIntrinsic(unittest.TestCase):

    def test_single_triangle(self):
        l_sq = np.array([[1.,1.,1.]])
        f = np.array([[0,1,2]],dtype=int)

        M_b = gpy.massmatrix_intrinsic(l_sq,f, type='barycentric')
        M_b_gt = np.array([[0.25,0.0,0.0],[0.0,0.25,0.0],[0.0,0.0,0.25]])/np.sqrt(3.)
        self.assertTrue(np.isclose(M_b.toarray(), M_b_gt).all())

        M_v = gpy.massmatrix_intrinsic(l_sq,f, type='voronoi')
        M_v_gt = np.array([[0.25,0.0,0.0],[0.0,0.25,0.0],[0.0,0.0,0.25]])/np.sqrt(3.)
        self.assertTrue(np.isclose(M_v.toarray(), M_v_gt).all())

        M_d = gpy.massmatrix_intrinsic(l_sq,f)
        self.assertTrue(np.isclose(M_d.toarray(), M_v_gt).all())

        M_f = gpy.massmatrix_intrinsic(l_sq,f, type='full')
        M_f_gt = np.array([[2.,1.,1.],[1.,2.,1.],[1.,1.,2.]])/(16.*np.sqrt(3.))
        self.assertTrue(np.isclose(M_f.toarray(), M_f_gt).all())

if __name__ == '__main__':
    unittest.main()