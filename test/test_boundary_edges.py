from .context import gpytoolbox as gpy
from .context import numpy as np
from .context import unittest

class TestBoundaryEdges(unittest.TestCase):

    def test_single_triangle(self):
        f = np.array([[0,1,2]],dtype=int)
        e = gpy.edges(f)
        be = gpy.boundary_edges(f)
        self.assertTrue(be.shape[0] == e.shape[0])
    
    def test_bunny(self):
        v,f = gpy.read_mesh("test/unit_tests_data/bunny_oded.obj")

        be = gpy.boundary_edges(f)

        #The bunny is closed
        self.assertTrue(len(be)==0)

    def test_mountain(self):
        v,f = gpy.read_mesh("test/unit_tests_data/mountain.obj")

        be = gpy.boundary_edges(f)

        #The mountain has a boundary of length 636
        b = 636
        self.assertTrue(be.shape[0]==b)

        #Each vertex should appear exactly twice.
        _,counts = np.unique(be, return_counts=True)
        self.assertTrue(np.all(counts==2))


    def test_airplane(self):
        v,f = gpy.read_mesh("test/unit_tests_data/airplane.obj")

        be = gpy.boundary_edges(f)
        
        #The plane has a boundary of length 341 with 27 components
        b = 341
        self.assertTrue(be.shape[0]==b)

        #Each vertex should appear exactly twice.
        _,counts = np.unique(be, return_counts=True)
        self.assertTrue(np.all(counts==2))

    def test_2d_regular(self):
        n = 40
        v,f = gpy.regular_square_mesh(n)

        be = gpy.boundary_edges(f)

        #The plane has a boundary of length (n-1)*4
        b = (n-1)*4
        self.assertTrue(be.shape[0]==b)

        #Each vertex should appear exactly twice.
        _,counts = np.unique(be, return_counts=True)
        self.assertTrue(np.all(counts==2))


if __name__ == '__main__':
    unittest.main()