from .context import gpytoolbox as gpy
from .context import numpy as np
from .context import unittest

class TestEdges(unittest.TestCase):

    def test_single_triangle(self):
        f = np.array([[0,1,2]],dtype=int)
        he = gpy.halfedges(f)
        e = gpy.edges(f)
        self.assertTrue(he.shape[0]*he.shape[1] == e.shape[0])
    
    def test_bunny(self):
        v,f = gpy.read_mesh("test/unit_tests_data/bunny_oded.obj")

        e = gpy.edges(f)
        #The bunny is closed
        self.assertTrue(v.shape[0] - e.shape[0] + f.shape[0] == 2)
        #There can be no duplicate edges
        self.assertTrue(np.unique(e,axis=0).shape == e.shape)

        eb,boundary_indices = gpy.edges(f,return_boundary_indices=True)
        self.assertTrue(np.all(e==eb))
        #The bunny is closed
        self.assertTrue(len(boundary_indices)==0)

        ei,interior_indices = gpy.edges(f,return_interior_indices=True)
        self.assertTrue(np.all(e==ei))
        #The bunny is closed
        self.assertTrue(len(interior_indices) == e.shape[0])

        em,nonmanifold_indices = gpy.edges(f,return_nonmanifold_indices=True)
        self.assertTrue(np.all(e==em))
        #The bunny is manifold
        self.assertTrue(len(nonmanifold_indices)==0)

    def test_mountain(self):
        v,f = gpy.read_mesh("test/unit_tests_data/mountain.obj")

        #The mountain has a boundary of length 636
        b = 636

        e = gpy.edges(f)
        #The mountain has a single boundary
        self.assertTrue(v.shape[0] - e.shape[0] + f.shape[0] == 1)
        #There can be no duplicate edges
        self.assertTrue(np.unique(e,axis=0).shape == e.shape)

        eb,boundary_indices = gpy.edges(f,return_boundary_indices=True)
        self.assertTrue(np.all(e==eb))
        self.assertTrue(len(boundary_indices)==b)

        ei,interior_indices = gpy.edges(f,return_interior_indices=True)
        self.assertTrue(np.all(e==ei))
        #The mountain has a boundary
        self.assertTrue(len(interior_indices) == e.shape[0]-b)

        em,nonmanifold_indices = gpy.edges(f,return_nonmanifold_indices=True)
        self.assertTrue(np.all(e==em))
        #The mountain is manifold
        self.assertTrue(len(nonmanifold_indices)==0)

    def test_airplane(self):
        v,f = gpy.read_mesh("test/unit_tests_data/airplane.obj")
        
        #The plane has a boundary of length 341 with 27 components
        b = 341
        c = 27

        e = gpy.edges(f)
        #The plane has a 27 boundaries
        self.assertTrue(v.shape[0] - e.shape[0] + f.shape[0] == 2-c)
        #There can be no duplicate edges
        self.assertTrue(np.unique(e,axis=0).shape == e.shape)

        eb,boundary_indices = gpy.edges(f,return_boundary_indices=True)
        self.assertTrue(np.all(e==eb))
        self.assertTrue(len(boundary_indices)==b)

        ei,interior_indices = gpy.edges(f,return_interior_indices=True)
        self.assertTrue(np.all(e==ei))
        #The mountain has a boundary
        self.assertTrue(len(interior_indices) == e.shape[0]-b)

        em,nonmanifold_indices = gpy.edges(f,return_nonmanifold_indices=True)
        self.assertTrue(np.all(e==em))
        #The mountain is manifold
        self.assertTrue(len(nonmanifold_indices)==0)

    def test_2d_regular(self):
        n = 40
        v,f = gpy.regular_square_mesh(n)

        #The plane has a boundary of length (n-1)*4
        b = (n-1)*4

        e = gpy.edges(f)
        #The plane has a single boundary
        self.assertTrue(v.shape[0] - e.shape[0] + f.shape[0] == 1)
        #There can be no duplicate edges
        self.assertTrue(np.unique(e,axis=0).shape == e.shape)

        eb,boundary_indices = gpy.edges(f,return_boundary_indices=True)
        self.assertTrue(np.all(e==eb))
        self.assertTrue(len(boundary_indices)==b)

        ei,interior_indices = gpy.edges(f,return_interior_indices=True)
        self.assertTrue(np.all(e==ei))
        #The plane has a boundary
        self.assertTrue(len(interior_indices) == e.shape[0]-b)

        em,nonmanifold_indices = gpy.edges(f,return_nonmanifold_indices=True)
        self.assertTrue(np.all(e==em))
        #The plane is manifold
        self.assertTrue(len(nonmanifold_indices)==0)

        #Let's attach a nonmanifold triangle
        v = np.block([[v, np.zeros((v.shape[0],1))],[0.5,0.5,0.2]])
        f = np.block([[f], [3*n+2, 3*n+3, v.shape[0]-1]])
        em2,nonmanifold_indices2 = gpy.edges(f,return_nonmanifold_indices=True)
        self.assertTrue(em2.shape[0] == em.shape[0]+2)
        self.assertTrue(len(nonmanifold_indices2)==1)


        


if __name__ == '__main__':
    unittest.main()