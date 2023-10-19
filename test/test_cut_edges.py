from .context import gpytoolbox as gpy
from .context import numpy as np
from .context import unittest

class TestCutEdges(unittest.TestCase):

    def test_two_triangles(self):
        F = np.array([[0,1,2], [2,1,3]],dtype=int)
        E = np.array([[1,2]])
        G,I = gpy.cut_edges(F,E)
        self.assertTrue(np.all(G==np.array([[0,2,4],[1,3,5]])))
        self.assertTrue(np.all(I==np.array([0, 2, 1, 1, 2, 3])))

    def test_icosphere(self):
        V,F = gpy.icosphere(3)
        E = np.array([[371,573], [573,571], [571,219]])
        G,I = gpy.cut_edges(F,E)

        b = gpy.boundary_loops(G)
        print(b)
        self.assertTrue(b[0].size == 2*E.shape[0])
        self.assertTrue(np.all(np.sort(b[0]) == np.sort(
            np.array([
                557,
                278,
                641,
                437,
                507,
                642
                ]))))
    
    def test_bunny(self):
        V,F = gpy.read_mesh("test/unit_tests_data/bunny_oded.obj")
        path = np.array([1575,1482,1394,1309,1310,1225,1141])
        E = np.stack((path[:-1],path[1:]), axis=-1)
        G,I = gpy.cut_edges(F,E)

        b = gpy.boundary_loops(G)
        self.assertTrue(b[0].size == 2*E.shape[0])
        self.assertTrue(np.all(np.sort(b[0]) == np.sort(
            np.array([
                 761,
                 790,
                 811,
                 865,
                 927,
                 929,
                 952,
                1038,
                2501,
                2578,
                2596,
                2613
                ]))))
        

if __name__ == '__main__':
    unittest.main()