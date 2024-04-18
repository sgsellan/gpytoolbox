from .context import gpytoolbox as gpy
from .context import numpy as np
from .context import unittest
from scipy.stats import norm
import os

class TestTetrahedralize(unittest.TestCase):
    def test_sphere(self):
        V,F = gpy.icosphere(3)

        W0,T,TF = gpy.copyleft.tetrahedralize(V)
        self.assertTrue((np.linalg.norm(W0, axis=-1)<1.+1e-6).all())
        self.assertTrue(W0.shape[0]>=V.shape[0])

        W1,T,TF = gpy.copyleft.tetrahedralize(V,F)
        self.assertTrue((np.linalg.norm(W1, axis=-1)<1.+1e-6).all())
        self.assertTrue(W1.shape[0]>=V.shape[0])

        W2,T,TF = gpy.copyleft.tetrahedralize(V,F, max_volume=0.001)
        self.assertTrue((np.linalg.norm(W2, axis=-1)<1.+1e-6).all())
        self.assertTrue(W2.shape[0]>=V.shape[0])
        self.assertTrue(W2.shape[0]>W1.shape[0])

        W3,T,TF = gpy.copyleft.tetrahedralize(V,F, min_rad_edge_ratio=0.2)
        self.assertTrue((np.linalg.norm(W3, axis=-1)<1.+1e-6).all())
        self.assertTrue(W3.shape[0]>=V.shape[0])
        self.assertTrue(W3.shape[0]>W1.shape[0])

        W4,T,TF = gpy.copyleft.tetrahedralize(V,F, min_rad_edge_ratio=0.2, max_volume=0.001)
        self.assertTrue((np.linalg.norm(W4, axis=-1)<1.+1e-6).all())
        self.assertTrue(W4.shape[0]>=V.shape[0])
        self.assertTrue(W4.shape[0]>W1.shape[0])


    def test_meshes(self):
        meshes = ["bunny_oded.obj", "spot.obj"]
        for mesh in meshes:
            V,F = gpy.read_mesh("test/unit_tests_data/" + mesh)
            V = gpy.normalize_points(V)

            W0,T,TF = gpy.copyleft.tetrahedralize(V)
            self.assertTrue((np.abs(W0)<0.5+1e-6).all())
            self.assertTrue(W0.shape[0]>=V.shape[0])

            W1,T,TF = gpy.copyleft.tetrahedralize(V,F)
            self.assertTrue((np.abs(W1)<0.5+1e-6).all())
            self.assertTrue(W1.shape[0]>=V.shape[0])

            W2,T,TF = gpy.copyleft.tetrahedralize(V,F, max_volume=0.001)
            self.assertTrue((np.abs(W2)<0.5+1e-6).all())
            self.assertTrue(W2.shape[0]>=V.shape[0])
            self.assertTrue(W2.shape[0]>W1.shape[0])

            W3,T,TF = gpy.copyleft.tetrahedralize(V,F, min_rad_edge_ratio=1.)
            self.assertTrue((np.abs(W3)<0.5+1e-6).all())
            self.assertTrue(W3.shape[0]>=V.shape[0])
            self.assertTrue(W3.shape[0]>W1.shape[0])

            W4,T,TF = gpy.copyleft.tetrahedralize(V,F, min_rad_edge_ratio=1., max_volume=0.001)
            self.assertTrue((np.abs(W4)<0.5+1e-6).all())
            self.assertTrue(W4.shape[0]>=V.shape[0])
            self.assertTrue(W4.shape[0]>W1.shape[0])
        

if __name__ == '__main__':
    unittest.main()