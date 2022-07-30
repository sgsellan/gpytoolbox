from .context import gpytoolbox as gpy
from .context import numpy as np
from .context import unittest

class TestTriangleTriangleAdjacency(unittest.TestCase):

    def test_two_triangles(self):
        f = np.array([[0,1,2],[2,1,3]],dtype=int)
        TT, TTi = gpy.triangle_triangle_adjacency(f)
        self.variety_of_asserts(f, TT, TTi)

    def test_three_triangles(self):
        f = np.array([[0,1,2],[2,1,3],[3,1,4]],dtype=int)
        TT, TTi = gpy.triangle_triangle_adjacency(f)
        self.variety_of_asserts(f, TT, TTi)
    
    def test_variety_of_meshes(self):
        meshes = ["airplane.obj", "armadillo.obj", "bunny.obj", "bunny_oded.obj", "mountain.obj", "wooden-chair-remesher-bug.obj"]
        for mesh in meshes:
            _,f = gpy.read_mesh("test/unit_tests_data/" + mesh)
            TT, TTi = gpy.triangle_triangle_adjacency(f)
            self.variety_of_asserts(f, TT, TTi)

    def variety_of_asserts(self, f, TT, TTi):
        self.assertTrue(TT.shape == f.shape)
        self.assertTrue(TTi.shape == f.shape)
        self.assertTrue(np.all((TT==-1) == (TTi==-1)))

        hes = gpy.halfedges(f)
        be = gpy.boundary_edges(f)

        for i in range(f.shape[0]):
            for j in range(f.shape[1]):
                he = hes[i,j,:]
                if TT[i,j] == -1:
                    #Is this really a boundary halfedge?
                    self.assertTrue((he[None,:] == be).all(-1).any() or (np.flip(he)[None,:] == be).all(-1).any())
                else:
                    #Is the adjacency info true?
                    self.assertTrue(np.all(np.flip(he) == hes[TT[i,j],TTi[i,j],:]))


if __name__ == '__main__':
    unittest.main()

