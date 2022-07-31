from .context import gpytoolbox
from .context import numpy as np
from .context import unittest

class TestRayMeshIntersect(unittest.TestCase):
    def test_simple_cube(self):
        # This is a cube, centered at the origin, with side length 1
        v,f = gpytoolbox.read_mesh("test/unit_tests_data/cube.obj")
        cam_pos = np.array([[1,0.1,0.1],[1,0.2,0.0]])
        cam_dir = np.array([[-1,0,0],[-1,0,0]])
        t, ids, l = gpytoolbox.ray_mesh_intersect(cam_pos.astype(np.float64),cam_dir.astype(np.float64),v.astype(np.float64),f.astype(np.int32))
        # There should only be two hits: let's check the output
        self.assertTrue(t.shape[0]==2)
        self.assertTrue(np.isclose(t[0],0.5))
        groundtruth_intersection = np.array([[0.5,0.1,0.1],[0.5,0.2,0.0]])
        intersection = cam_pos + t[:,None]*cam_dir
        self.assertTrue(np.isclose(groundtruth_intersection,intersection).all())

if __name__ == '__main__':
    unittest.main()