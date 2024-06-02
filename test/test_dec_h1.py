from .context import gpytoolbox as gpy
from .context import numpy as np
from .context import scipy as sp
from .context import unittest

class TestDecH1(unittest.TestCase):

    def test_uniform_triangle(self):
        v = np.array([[0.0,0.0],[1.0,0.0],[0.5,0.5*np.sqrt(3)]])
        f = np.array([[0,1,2]],dtype=int)

        h1 = gpy.dec_h1(v,f)

        a = 0.5/np.sqrt(3.)
        gt_arr = np.array([[a,0.,0.],[0.,a,0.],[0.,0.,a]])
        self.assertTrue(np.isclose(h1.toarray(), gt_arr).all())

    def test_equivalence_to_cotangent_laplacian(self):
        meshes = ["armadillo.obj", "bunny.obj", "mountain.obj"]
        for mesh in meshes:
            v,f = gpy.read_mesh("test/unit_tests_data/" + mesh)

            d0 = gpy.dec_d0(f)
            h1 = gpy.dec_h1(v,f)
            L_dec = d0.transpose() * h1 * d0

            L_cotangent = gpy.cotangent_laplacian(v,f)

            self.assertTrue(np.isclose(sp.sparse.linalg.norm(L_dec-L_cotangent) / L_cotangent.nnz, 0))

if __name__ == '__main__':
    unittest.main()