from .context import gpytoolbox
from .context import numpy as np
from .context import unittest
# import polyscope as ps
# import igl

class TestMarchingCubes(unittest.TestCase):
    def test_meshes(self):
        meshes = ["bunny_oded.obj", "armadillo.obj"] # Closed meshes
        for mesh in meshes:
            # num_points = 1000
            V,F = gpytoolbox.read_mesh("test/unit_tests_data/" +mesh)
            V = gpytoolbox.normalize_points(V,center=np.array([0.5,0.5,0.5]))
            # V, _, N, F, _, _ = igl.read_obj("test/unit_tests_data/" +mesh)
            # # ps.register_point_cloud("input",gpytoolbox.barycenters(V,F))
            # ps.show()
            # Normalize mesh
            # V = gpytoolbox.normalize_points(V,center=np.array([0.5,0.5,0.5]))
            # Generate cube tet mesh
            gs = 150
            GV,_ = gpytoolbox.regular_cube_mesh(gs)
            # Get winding number at grid vertices
            s1 = gpytoolbox.signed_distance(GV,V,F)[0]
            
            # Get isosurface
            U,G = gpytoolbox.marching_cubes(s1,GV,gs,gs,gs,0.0)
            # Get winding number now using the isosurface
            s2 = gpytoolbox.signed_distance(GV,U,G)[0]

            self.assertTrue(np.isclose(s1,s2,atol=1e-2).all())
            # Now the claim is that U,G and V,F should be "similar"
            dists = gpytoolbox.squared_distance(U,V,F=F,use_cpp=True)[0]
            self.assertTrue(np.isclose(dists,0.0,atol=1e-2).all())
            
            # ps.register_point_cloud("inside",GV[s2<0,:])
            # ps.register_surface_mesh("input",V,F)
            # ps.register_surface_mesh("output",U,G)
            # ps.show()
            # print(dists)
            

if __name__ == '__main__':
    unittest.main()
