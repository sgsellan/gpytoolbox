import numpy as np
# TODO: write all these functions and uncomment the parts of  the unit test
from .context import gpytoolbox
from .context import numpy as np
from .context import unittest


class TestRegularCubeMesh(unittest.TestCase):
    def test_six_tets(self):
        self.assertTrue(True)
        # Generate meshes of very diverse sizes
        for nx in range(4,50,5):
            for ny in range(4,50,5):
                for nz in range(4,50,5):
                    V,T = gpytoolbox.regular_cube_mesh(nx,ny,nz)    
                    # Check: correct number of vertices
                    self.assertTrue(V.shape[0], nx*ny*nz)
                    vols = gpytoolbox.volume(V,T)
                    # self.assertTrue all volumes are positive
                    self.assertTrue(np.all(vols>0))
                    # Check that all tets are combinatorially connected
                    # self.assertTrue(np.max(connected_components(adjacency_matrix(T))[0])==1)
                    # Check that the number of faces is six times the number of faces in each outer face of the cube, which is 2*(n-1)*(n-1)
                    # self.assertTrue(boundary_facets(T).shape[0]==2*(n-1)*(n-1)*6)
            
    def test_five_tets(self):
        for nx in range(4,50,5):
            for ny in range(4,50,5):
                for nz in range(4,50,5):
                    V,T = gpytoolbox.regular_cube_mesh(nx,ny,nz,type='five')   
                    # Check: correct number of vertices
                    self.assertTrue(V.shape[0], nx*ny*nz)
                    vols = gpytoolbox.volume(V,T)
                    # self.assertTrue all volumes are positive
                    self.assertTrue(np.all(vols>0))
                    # Check that all tets are combinatorially connected
                    # self.assertTrue(np.max(connected_components(adjacency_matrix(T))[0])==1)

    def test_reflectionally_symmetric(self):
        for nx in range(4,50,5):
            for ny in range(4,50,5):
                for nz in range(4,50,5):
                    V,T = gpytoolbox.regular_cube_mesh(nx,ny,nz,type='reflectionally-symmetric')   
                    # Check: correct number of vertices
                    self.assertTrue(V.shape[0], nx*ny*nz)
                    vols = gpytoolbox.volume(V,T)
                    # self.assertTrue all volumes are positive
                    self.assertTrue(np.all(vols>0))
                    # Check that all tets are combinatorially connected
                    # self.assertTrue(np.max(connected_components(adjacency_matrix(T))[0])==1)
                    # Check that the number of faces is six times the number of faces in each outer face of the cube, which is 2*(n-1)*(n-1)
                    # self.assertTrue(boundary_facets(T).shape[0]==2*(n-1)*(n-1)*6)

if __name__ == '__main__':
    unittest.main()