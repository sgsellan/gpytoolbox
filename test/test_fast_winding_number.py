from .context import gpytoolbox
from .context import numpy as np
from .context import unittest

class TestFastWindingNumber(unittest.TestCase):
    def test_bunny(self):
        num_points = 1000
        V,F = gpytoolbox.read_mesh("test/unit_tests_data/" + "bunny_oded.obj")
        # Generate random points on mesh
        Q,I,_ = gpytoolbox.random_points_on_mesh(V,F,num_points,return_indices=True,rng=np.random.default_rng(5))
        # Per face normals
        N = gpytoolbox.per_face_normals(V,F,unit_norm=True)
        # Compute winding number
        eps = 1e-3
        points_out = Q + eps*N[I,:]
        points_in = Q - eps*N[I,:]
        wn_in = gpytoolbox.fast_winding_number(points_in,V,F)
        wn_out = gpytoolbox.fast_winding_number(points_out,V,F)
        # print(wn_in)
        # print(wn_out)
        # print(np.isclose(wn_out,0,atol=1e-2))
        self.assertTrue(np.isclose(wn_out,0,atol=1e-2).all())
        self.assertTrue(np.isclose(wn_in,1,atol=1e-2).all())
    def test_cube(self):
        num_points = 10000
        V,F = gpytoolbox.read_mesh("test/unit_tests_data/" + "cube.obj")
        # Generate random points on mesh
        Q,I,_ = gpytoolbox.random_points_on_mesh(V,F,num_points,return_indices=True,rng=np.random.default_rng(5))
        # Per face normals
        N = gpytoolbox.per_face_normals(V,F,unit_norm=True)
        # Compute winding number
        eps = 1e-3
        points_out = Q + eps*N[I,:]
        points_in = Q - eps*N[I,:]
        wn_in = gpytoolbox.fast_winding_number(points_in,V,F)
        wn_out = gpytoolbox.fast_winding_number(points_out,V,F)
        # print(wn_in)
        # print(wn_out)
        # print(np.isclose(wn_out,0,atol=1e-2))
        self.assertTrue(np.isclose(wn_out,0,atol=1e-2).all())
        self.assertTrue(np.isclose(wn_in,1,atol=1e-2).all())

if __name__ == '__main__':
    unittest.main()
