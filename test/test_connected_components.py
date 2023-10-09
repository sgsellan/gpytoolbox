from .context import gpytoolbox as gpy
from .context import numpy as np
from .context import unittest

class TestConnectedComponents(unittest.TestCase):

    def test_single_triangle(self):
        F = np.array([[0,1,2]], dtype=int)
        Cs = gpy.connected_components(F)
        C,CF = gpy.connected_components(F, return_face_indices=True)

        self.assertTrue(np.all(Cs==C))
        self.assertTrue(np.all(C==np.array([0,0,0])))
        self.assertTrue(np.all(CF==np.array([0])))

    def test_two_connected_triangles(self):
        F = np.array([[0,1,2],[2,1,3]], dtype=int)
        Cs = gpy.connected_components(F)
        C,CF = gpy.connected_components(F, return_face_indices=True)

        self.assertTrue(np.all(Cs==C))
        self.assertTrue(np.all(C==np.array([0,0,0,0])))
        self.assertTrue(np.all(CF==np.array([0,0])))

    def test_two_separate_triangles(self):
        F = np.array([[0,1,2],[3,4,5]], dtype=int)
        Cs = gpy.connected_components(F)
        C,CF = gpy.connected_components(F, return_face_indices=True)

        self.assertTrue(np.all(Cs==C))
        self.assertTrue(np.all(C==np.array([0,0,0,1,1,1])))
        self.assertTrue(np.all(CF==np.array([0,1])))
    
    def test_bunny(self):
        V,F = gpy.read_mesh("test/unit_tests_data/bunny_oded.obj")
        Cs = gpy.connected_components(F)
        C,CF = gpy.connected_components(F, return_face_indices=True)

        self.assertTrue(np.all(Cs==C))
        self.assertTrue(np.all(C==0))
        self.assertTrue(np.all(CF==0))
    
    def test_mountain(self):
        V,F = gpy.read_mesh("test/unit_tests_data/mountain.obj")
        Cs = gpy.connected_components(F)
        C,CF = gpy.connected_components(F, return_face_indices=True)

        self.assertTrue(np.all(Cs==C))
        self.assertTrue(np.all(C==0))
        self.assertTrue(np.all(CF==0))
    
    def test_split_mountain(self):
        V,F = gpy.read_mesh("test/unit_tests_data/split_mountain.obj")
        Cs = gpy.connected_components(F)
        C,CF = gpy.connected_components(F, return_face_indices=True)

        self.assertTrue(np.all(Cs==C))
        self.assertTrue(np.sum(C==0)==12005)
        self.assertTrue(np.sum(C==1)==13419)
        self.assertTrue(np.sum(C>1)==0)
        self.assertTrue(np.sum(CF==0)==23512)
        self.assertTrue(np.sum(CF==1)==26350)
        self.assertTrue(np.sum(CF>1)==0)

if __name__ == '__main__':
    unittest.main()
