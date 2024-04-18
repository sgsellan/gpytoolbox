from .context import gpytoolbox as gpy
from .context import numpy as np
from .context import unittest
from scipy.stats import norm
import os

class TestTriangulate(unittest.TestCase):
    def test_circle(self):
        r = np.linspace(0, 2.*np.pi, num=100, endpoint=False)
        V = np.stack((np.cos(r), np.sin(r)), axis=-1)
        E = gpy.edge_indices(V.shape[0], closed=True)

        def check(V,F,area=np.inf,angle=0):
            self.assertTrue(len(gpy.non_manifold_edges(F))==0)
            self.assertTrue((np.linalg.norm(V,axis=-1)<1.+1e-6).all())
            self.assertTrue((gpy.doublearea(V,F)<2.*area+1e-6).all())
            self.assertTrue((gpy.tip_angles(V,F)>angle-1e-6).all())

        # Convex hull
        W,F = gpy.triangulate(V)
        check(W,F)

        # Including edges
        W,F = gpy.triangulate(V,E)
        check(W,F)


    def test_annulus(self):
        r0 = np.linspace(0, 2.*np.pi, num=50, endpoint=False)
        V0 = 0.5*np.stack((np.cos(r0), np.sin(r0)), axis=-1)
        E0 = gpy.edge_indices(V0.shape[0], closed=True)
        E0[:,[1,0]] = E0[:,[0,1]]
        r1 = np.linspace(0, 2.*np.pi, num=100, endpoint=False)
        V1 = np.stack((np.cos(r1), np.sin(r1)), axis=-1)
        E1 = gpy.edge_indices(V1.shape[0], closed=True)
        V = np.concatenate((V0,V1), axis=0)
        E = np.concatenate((E0,E1+V0.shape[0]), axis=0)

        # Convex hull
        W,F = gpy.triangulate(V)
        self.assertTrue((np.linalg.norm(W,axis=-1)<1.+1e-6).all())

        def check(V,F,area=np.inf,angle=0):
            self.assertTrue(len(gpy.non_manifold_edges(F))==0)
            self.assertTrue((np.linalg.norm(V,axis=-1)<1.+1e-6).all()
                and (np.linalg.norm(V,axis=-1)>0.5-1e-6).all())
            self.assertTrue((gpy.doublearea(V,F)<2.*area+1e-6).all())
            self.assertTrue((gpy.tip_angles(V,F)>angle-1e-6).all())

        # Including edges
        W,F = gpy.triangulate(V,E)
        check(W,F)


    def test_image(self):
        filename = "test/unit_tests_data/illustrator.png"
        V0 = gpy.png2poly(filename)[0][:-1]
        E0 = gpy.edge_indices(V0.shape[0], closed=True)
        E0[:,[1,0]] = E0[:,[0,1]]
        V1 = gpy.png2poly(filename)[2][:-1]
        E1 = gpy.edge_indices(V1.shape[0], closed=True)
        V = gpy.normalize_points(np.concatenate((V0,V1), axis=0))
        E = np.concatenate((E0,E1+V0.shape[0]), axis=0)

        def check(V,F,area=np.inf,angle=0):
            self.assertTrue(len(gpy.non_manifold_edges(F))==0)
            self.assertTrue((np.abs(V)<1.+1e-6).all())
            dA = gpy.doublearea(V,F)
            self.assertTrue((dA[np.isfinite(dA)]<2.*area+1e-6).all())
            self.assertTrue((gpy.tip_angles(V,F)>angle-1e-6).all())

        # Convex hull
        W,F = gpy.triangulate(V)
        check(W,F)

        # Including edges
        W,F = gpy.triangulate(V,E)
        check(W,F)
        

if __name__ == '__main__':
    unittest.main()