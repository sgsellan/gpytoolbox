from .context import gpytoolbox as gpy
from .context import numpy as np
from .context import unittest

class TestBoundaryFaces(unittest.TestCase):

    def test_simple_cube_mesh(self):
        _,t = gpy.regular_cube_mesh(2)
        bf = gpy.boundary_faces(t)
        self.assertTrue(bf.shape[0] == 12)

    def test_larger_cube_mesh(self):
        _,t = gpy.regular_cube_mesh(5)
        bf = gpy.boundary_faces(t)
        self.assertTrue(bf.shape[0] == 192)


if __name__ == '__main__':
    unittest.main()