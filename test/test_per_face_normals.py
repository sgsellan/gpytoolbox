from .context import gpytoolbox
from .context import numpy as np
from .context import unittest
import igl

class TestPerFaceNormals(unittest.TestCase):
    def test_single_triangle(self):
        v = np.array([[0.0,0.0,0.0],[1.0,0.0,0.0],[0.0,1.0,0.0]])
        f_up = np.array([[0,1,2]],dtype=int)
        n = gpytoolbox.per_face_normals(v,f_up)
        n_up = np.array([[0,0,1]])
        # the normal should be pointing up
        self.assertTrue(np.isclose(n - n_up,0.0).all())
        # Now if we change the ordering convention, it should point down:
        f_down = np.array([[0,2,1]],dtype=int)
        n = gpytoolbox.per_face_normals(v,f_down)
        n_down = np.array([[0,0,-1]])
        self.assertTrue(np.isclose(n - n_down,0.0).all())

    def test_bunny(self):
        v,f = igl.read_triangle_mesh("test/unit_tests_data/bunny_oded.obj")
        n_gt = igl.per_face_normals(v,f,np.array([0.,0.,0.]))
        n_gt =  n_gt/np.tile(np.linalg.norm(n_gt,axis=1)[:,None],(1,3))
        n = gpytoolbox.per_face_normals(v,f,unit_norm=True)
        self.assertTrue(np.isclose(n-n_gt,0.0).all())


if __name__ == '__main__':
    unittest.main()