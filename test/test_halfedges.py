from .context import gpytoolbox as gpy
from .context import numpy as np
from .context import unittest

class TestHalfedges(unittest.TestCase):
    # There is not much to test here that goes beyond just inputting the
    # definition of the function, but we can make sure that a few conditions
    # are fulfilled.

    def test_single_triangle(self):
        f = np.array([[0,1,2]],dtype=int)
        he = gpy.halfedges(f)
        he_groundtruth = np.array([[1,2], [2,0], [0,1]], dtype=int)
        self.assertTrue(np.all(he == he_groundtruth))
    
    def test_bunny(self):
        _,f = gpy.read_mesh("test/unit_tests_data/bunny_oded.obj")
        he = gpy.halfedges(f)
        self.assertTrue(he.shape==(f.shape[0],3,2))

    def test_mountain(self):
        _,f = gpy.read_mesh("test/unit_tests_data/mountain.obj")
        he = gpy.halfedges(f)
        self.assertTrue(he.shape==(f.shape[0],3,2))

    def test_airplane(self):
        _,f = gpy.read_mesh("test/unit_tests_data/airplane.obj")
        he = gpy.halfedges(f)
        self.assertTrue(he.shape==(f.shape[0],3,2))


        


if __name__ == '__main__':
    unittest.main()