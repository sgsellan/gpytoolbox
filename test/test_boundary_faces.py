from .context import gpytoolbox as gpy
from .context import numpy as np
from .context import unittest

class TestBoundaryFaces(unittest.TestCase):

    def test_simple_cube_mesh(self):
        v,t = gpy.regular_cube_mesh(2)
        bf = gpy.boundary_faces(t)
        self.assertTrue(bf.shape[0] == 12)
        # if these are indeed the boundary faces, then the normals should be basis vectors in R^3
        n = gpy.per_face_normals(v,bf,unit_norm=True)
        l1_norm = np.linalg.norm(n,ord=1,axis=1)
        self.assertTrue(np.isclose(l1_norm,1).all())
        # t is oriented consistently, so the normals should be oriented consistently
        barycenters = (v[bf[:,0],:] + v[bf[:,1],:] + v[bf[:,2],:])/3.0
        points_slightly_outside = barycenters + 0.1*n
        # are they, really, outside?
        self.assertTrue((np.max(np.abs(points_slightly_outside - 0.5),axis=1)>0.5).all())
        # Same but inside
        points_slightly_inside = barycenters - 0.1*n
        self.assertTrue((np.max(np.abs(points_slightly_inside - 0.5),axis=1)<0.5).all())

    def test_larger_cube_mesh(self):
        v,t = gpy.regular_cube_mesh(5)
        bf = gpy.boundary_faces(t)
        self.assertTrue(bf.shape[0] == 192)
        n = gpy.per_face_normals(v,bf,unit_norm=True)
        l1_norm = np.linalg.norm(n,ord=1,axis=1)
        self.assertTrue(np.isclose(l1_norm,1).all())
        # t is oriented consistently, so the normals should be oriented consistently
        barycenters = (v[bf[:,0],:] + v[bf[:,1],:] + v[bf[:,2],:])/3.0
        points_slightly_outside = barycenters + 0.1*n
        # are they, really, outside?
        self.assertTrue((np.max(np.abs(points_slightly_outside - 0.5),axis=1)>0.5).all())
        # Same but inside
        points_slightly_inside = barycenters - 0.1*n
        self.assertTrue((np.max(np.abs(points_slightly_inside - 0.5),axis=1)<0.5).all())


if __name__ == '__main__':
    unittest.main()