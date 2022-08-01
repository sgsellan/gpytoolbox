from .context import gpytoolbox
from .context import unittest
from .context import numpy as np

class TestRemoveDuplicateVertices(unittest.TestCase):
    def test_triangle_pairs(self):
        v = np.array([[0.0,0.0],[1.0,0.0],[0.0,1.0], # triangle 1
                      [1.0,0.0],[1.0,1.0],[0.0,1.0]]) # triangle 2
        f = np.array([[0,1,2],[3,4,5]],dtype=int)
        SV, SVI, SVJ = gpytoolbox.remove_duplicate_vertices(v)
        unique_verts = np.array([[0., 0.],
                                 [0., 1.],
                                 [1., 0.],
                                 [1., 1.]])
        svi_groundtruth = np.array([0,2,1,4])
        svj_groundtruth = np.array([0,2,1,2,3,1])
        self.assertTrue(np.isclose(SV - unique_verts,0.0).all())
        self.assertTrue(np.isclose(SVJ - svj_groundtruth,0.0).all())
        self.assertTrue(np.isclose(SVI - svi_groundtruth,0.0).all())
        SV, SVI, SVJ, SF = gpytoolbox.remove_duplicate_vertices(v,faces=f)
        f_groundtruth = np.array([[0,2,1],[2,3,1]])
        self.assertTrue(np.isclose(SV - unique_verts,0.0).all())
        self.assertTrue(np.isclose(SVJ - svj_groundtruth,0.0).all())
        self.assertTrue(np.isclose(SVI - svi_groundtruth,0.0).all())
        self.assertTrue(np.isclose(SF - f_groundtruth,0.0).all())
        # Now let's make it a bit noisy
        v = np.array([[0.0,0.0],[1.05,0.0],[0.0,1.001], # triangle 1
                      [1.0,0.0],[1.0,1.0],[0.0,1.0]]) # triangle 2
        # Running it now should not change anything
        SV, SVI, SVJ, SF = gpytoolbox.remove_duplicate_vertices(v,faces=f)
        self.assertTrue(SV.shape[0]==v.shape[0])
        # Running it with 0.1 tolerance we should get the same thing as earlier
        SV, SVI, SVJ, SF = gpytoolbox.remove_duplicate_vertices(v,faces=f,epsilon=0.1)
        self.assertTrue(np.isclose(SVJ - svj_groundtruth,0.0).all())
        self.assertTrue(np.isclose(SVI - svi_groundtruth,0.0).all())
        self.assertTrue(np.isclose(SF - f_groundtruth,0.0).all())
        # Running it with 0.01 tolerance should give us five points
        SV, SVI, SVJ, SF = gpytoolbox.remove_duplicate_vertices(v,faces=f,epsilon=0.01)
        self.assertTrue(SV.shape[0]==5)

if __name__ == '__main__':
    unittest.main()