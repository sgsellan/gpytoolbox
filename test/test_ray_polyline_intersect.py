import numpy as np
from .context import gpytoolbox
from .context import unittest

class TestRayPolylineIntersect(unittest.TestCase):
    def test_rays_against_square_2d(self):
        # Build a polyline; for example, a square
        V = np.array([ [-1.0, -1.0], [-1.0, 1.0], [1.0, 1.0], [1.0, -1.0] ])

        # Camera position and direction
        cam_pos = np.array([-0.5,-1.5])
        cam_dir = np.array([0.0,1.0])
        # Looking upwards: intersection should be [-0.5,-1.0], normal downwards
        x, n, ind = gpytoolbox.ray_polyline_intersect(cam_pos,cam_dir,V)
        self.assertTrue((np.isclose(x,np.array([-0.5,-1.0]))).all())
        self.assertTrue((np.isclose(n,np.array([0.0,-1.0]))).all())

        # Oblique direction, inside
        cam_pos = np.array([0.2,0.0])
        cam_dir = np.array([0.3,0.4])
        # Intersection should be [0.95,1.0], normal should be upwards
        x, n, ind = gpytoolbox.ray_polyline_intersect(cam_pos,cam_dir,V)
        self.assertTrue((np.isclose(x,np.array([0.95,1.0]))).all())
        self.assertTrue((np.isclose(n,np.array([0.0,1.0]))).all())

        # Degeneracies: Parallel without a hit
        cam_pos = np.array([1.1,1.1])
        cam_dir = np.array([0.0,-1.0])
        # There should be no intersection
        x, n, ind = gpytoolbox.ray_polyline_intersect(cam_pos,cam_dir,V)
        # if no intersection, x is infinity and n is zero
        self.assertTrue((np.isclose(x,np.array([np.Inf,np.Inf]))).all())
        self.assertTrue((np.isclose(n,np.array([0.0, 0.0]))).all())
        # Index is -1
        self.assertTrue(ind==-1)

        # Degeneracy: Coincident
        cam_pos = np.array([-1.0,-2.0])
        cam_dir = np.array([0.0,1.0])
        # Intersection should be [-1,-1], normal isn't well defined
        x, n, ind = gpytoolbox.ray_polyline_intersect(cam_pos,cam_dir,V)
        self.assertTrue((np.isclose(x,np.array([-1.0,-1.0]))).all())


if __name__ == '__main__':
    unittest.main()
