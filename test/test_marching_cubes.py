from .context import gpytoolbox
from .context import numpy as np
from .context import unittest

class TestMarchingCubes(unittest.TestCase):
    def test_meshes(self):
        meshes = ["bunny_oded.obj", "armadillo.obj"] # Closed meshes
        for mesh in meshes:
            # num_points = 1000
            V,F = gpytoolbox.read_mesh("test/unit_tests_data/" +mesh)
            # Normalize mesh
            V = gpytoolbox.normalize_points(V,center=np.array([0.5,0.5,0.5]))
            # Generate cube tet mesh
            GV,_ = gpytoolbox.regular_cube_mesh(100)
            # Get winding number at grid vertices
            w = gpytoolbox.fast_winding_number(GV,V,F)
            # Get isosurface
            U,G = gpytoolbox.marching_cubes(w,GV,100,100,100,0.5)
            # Now the claim is that U,G and V,F should be "similar"
            dists = gpytoolbox.squared_distance(U,V,F=F,use_cpp=True)[0]
            self.assertTrue(np.isclose(dists,0.0,atol=1e-2).all())
            # print(dists)
            

if __name__ == '__main__':
    unittest.main()
