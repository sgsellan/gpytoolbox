from .context import gpytoolbox as gpy
from .context import numpy as np
from .context import unittest

class TestBoundaryVertices(unittest.TestCase):

    def test_single_triangle(self):
        f = np.array([[0,1,2]],dtype=int)
        b = gpy.boundary_vertices(f)
        self.assertTrue(b.shape[0] == 3)
    
    def test_bunny(self):
        v,f = gpy.read_mesh("test/unit_tests_data/bunny_oded.obj")

        b = gpy.boundary_vertices(f)

        #The bunny is closed
        self.assertTrue(len(b)==0)

    def test_mountain(self):
        v,f = gpy.read_mesh("test/unit_tests_data/mountain.obj")

        b = gpy.boundary_vertices(f)

        #The mountain has a boundary of length 636
        self.assertTrue(b.shape[0]==636)

        #Each vertex should appear exactly once.
        _,counts = np.unique(b, return_counts=True)
        self.assertTrue(np.all(counts==1))


    def test_airplane(self):
        v,f = gpy.read_mesh("test/unit_tests_data/airplane.obj")

        b = gpy.boundary_vertices(f)
        
        #The plane has a boundary of length 341
        self.assertTrue(b.shape[0]==341)

        #Each vertex should appear exactly twice.
        _,counts = np.unique(b, return_counts=True)
        self.assertTrue(np.all(counts==1))

    def test_2d_regular(self):
        n = 40
        v,f = gpy.regular_square_mesh(n)

        b = gpy.boundary_vertices(f)

        #The plane has a boundary of length (n-1)*4
        self.assertTrue(b.shape[0]==(n-1)*4)

        #Each vertex should appear exactly twice.
        _,counts = np.unique(b, return_counts=True)
        self.assertTrue(np.all(counts==1))


if __name__ == '__main__':
    unittest.main()