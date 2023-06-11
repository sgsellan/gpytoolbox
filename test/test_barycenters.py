from .context import gpytoolbox
from .context import numpy as np
from .context import unittest

class TestBarycenters(unittest.TestCase):
    def test_2d(self):
        filename = "test/unit_tests_data/poly.png"
        poly = gpytoolbox.png2poly(filename)
        V = poly[0]
        F = gpytoolbox.edge_indices(V.shape[0])
        B = gpytoolbox.barycenters(V,F)
        B_gt = V[F[:,0],:] + V[F[:,1],:]
        B_gt /= 2
        self.assertTrue(np.allclose(B,B_gt))
        # print(B)
    def test_3d(self):
        v, f = gpytoolbox.read_mesh("test/unit_tests_data/bunny_oded.obj")
        b = gpytoolbox.barycenters(v,f)
        b_gt = v[f[:,0],:] + v[f[:,1],:] + v[f[:,2],:]
        b_gt /= 3
        self.assertTrue(np.allclose(b,b_gt))

if __name__ == '__main__':
    unittest.main()