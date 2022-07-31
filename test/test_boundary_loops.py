from .context import gpytoolbox as gpy
from .context import numpy as np
from .context import unittest

class TestBoundaryLoops(unittest.TestCase):

    def test_single_triangle(self):
        f = np.array([[0,1,2]],dtype=int)
        loops = gpy.boundary_loops(f)
        self.assertTrue(len(loops)==1)
        self.assertTrue(len(loops[0])==3)
    
    def test_bunny(self):
        v,f = gpy.read_mesh("test/unit_tests_data/bunny_oded.obj")

        loops = gpy.boundary_loops(f)
        #The bunny is closed
        self.assertTrue(len(loops)==0)

    def test_mountain(self):
        v,f = gpy.read_mesh("test/unit_tests_data/mountain.obj")

        #The mountain has a boundary of length 636 with one component
        b = 636
        c = 1

        loops = gpy.boundary_loops(f)
        self.assertTrue(len(loops)==c)
        self.assertTrue(len(loops[0])==b)

    def test_airplane(self):
        v,f = gpy.read_mesh("test/unit_tests_data/airplane.obj")
        
        #The plane has a boundary of length 341 with 27 components
        b = 341
        c = 27

        loops = gpy.boundary_loops(f)
        self.assertTrue(len(loops)==c)
        self.assertTrue(sum([len(l) for l in loops])==b)

    def test_2d_regular(self):
        n = 40
        v,f = gpy.regular_square_mesh(n)

        #The plane has a boundary of length (n-1)*4 with one component
        b = (n-1)*4

        loops = gpy.boundary_loops(f)
        self.assertTrue(len(loops)==1)
        self.assertTrue(len(loops[0])==b)


        


if __name__ == '__main__':
    unittest.main()