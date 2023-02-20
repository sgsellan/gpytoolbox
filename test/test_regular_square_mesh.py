from .context import gpytoolbox
from .context import numpy as np
from .context import unittest

class TestRegularSquareMesh(unittest.TestCase):
    def test_valid_mesh(self):
        # Generate meshes of very diverse sizes
        for nx in range(5,50,5):
            for ny in range(5,50,5):
                V,F = gpytoolbox.regular_square_mesh(nx,ny)  
                # Check: correct number of vertices
                self.assertTrue(V.shape[0], nx*ny) 
                # Check: vertices are between minus one and one
                self.assertTrue(np.max(V)==1.0)
                self.assertTrue(np.min(V)==-1.0)
                # Check: all triangles are combinatorially connected
                self.assertTrue(len(gpytoolbox.boundary_loops(F))==1)
                # When we have adjacency matrix:
                # self.assertTrue(np.max(connected_components(adjacency_matrix(F))[0])==1)
                # Check: all faces properly oriented
                normals = np.cross( V[F[:,1],:] - V[F[:,0],:] , V[F[:,2],:] - V[F[:,0],:] , axis=1)
                self.assertTrue((normals>0).all())

if __name__ == '__main__':
    unittest.main()